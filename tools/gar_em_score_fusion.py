import argparse
import json
import os
import os.path as op
import sys
from collections import OrderedDict

import torch
import torch.nn.functional as F

REPO_ROOT = op.abspath(op.join(op.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from datasets import build_dataloader
from model.build_finetune import build_finetune_model
from utils.iotools import load_train_configs, mkdir_if_missing
from utils.metrics import rank


def load_checkpoint_state(path):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    cleaned = OrderedDict()
    for key, value in state.items():
        if key.startswith("module."):
            key = key[7:]
        cleaned[key] = value
    return cleaned


def load_expert_specs(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        experts = payload
    else:
        experts = payload.get("experts", [])
    if not experts:
        raise ValueError("Expert config must contain a non-empty `experts` list.")
    for index, expert in enumerate(experts):
        if "checkpoint" not in expert:
            raise ValueError(f"Expert {index} is missing `checkpoint`.")
        if "config_file" not in expert:
            raise ValueError(f"Expert {index} is missing `config_file`.")
        expert.setdefault("name", f"expert_{index}")
    return experts


def apply_overrides(args, expert, cli_args):
    args.training = False
    args.output_dir = cli_args.output_dir
    if cli_args.root_dir:
        args.root_dir = cli_args.root_dir
    if cli_args.test_batch_size:
        args.test_batch_size = cli_args.test_batch_size

    for key in (
        "dataset_name",
        "img_size",
        "stride_size",
        "loss_names",
        "pretrain_choice",
        "text_length",
        "val_dataset",
        "num_workers",
    ):
        if key in expert and expert[key] is not None:
            setattr(args, key, expert[key])
    return args


@torch.no_grad()
def compute_similarity(model, img_loader, txt_loader, device):
    model.eval()
    qids, gids, qfeats, gfeats = [], [], [], []

    for pid, caption in txt_loader:
        caption = caption.to(device)
        text_feat = model.encode_text(caption)
        qids.append(pid.view(-1).cpu())
        qfeats.append(text_feat.float().cpu())

    for pid, image in img_loader:
        image = image.to(device)
        image_feat = model.encode_image(image)
        gids.append(pid.view(-1).cpu())
        gfeats.append(image_feat.float().cpu())

    qids = torch.cat(qids, 0)
    gids = torch.cat(gids, 0)
    qfeats = F.normalize(torch.cat(qfeats, 0).to(device), p=2, dim=1)
    gfeats = F.normalize(torch.cat(gfeats, 0).to(device), p=2, dim=1)
    similarity = (qfeats @ gfeats.t()).float().cpu()
    return similarity, qids, gids


def evaluate_similarity(similarity, qids, gids):
    cmc, mean_ap, mean_inp, _ = rank(
        similarity=similarity,
        q_pids=qids,
        g_pids=gids,
        max_rank=10,
        get_mAP=True,
    )
    cmc = cmc.cpu().numpy()
    return {
        "R1": float(cmc[0]),
        "R5": float(cmc[4]),
        "R10": float(cmc[9]),
        "RSum": float(cmc[0] + cmc[4] + cmc[9]),
        "mAP": float(mean_ap.cpu().item()),
        "mINP": float(mean_inp.cpu().item()),
    }


def normalize_by_query(values, neutral=0.5, eps=1e-12):
    minimum = values.min(dim=0, keepdim=True).values
    maximum = values.max(dim=0, keepdim=True).values
    scale = maximum - minimum
    normalized = (values - minimum) / scale.clamp_min(eps)
    return torch.where(scale > eps, normalized, torch.full_like(values, neutral))


def pairwise_topk_overlap(first, second):
    return (first.unsqueeze(2) == second.unsqueeze(1)).any(dim=2).float().mean(dim=1)


def compute_adaptive_weights(
    similarities,
    topk,
    rank_weight,
    hardneg_weight,
    complement_weight,
    uncertainty_weight,
    temperature,
):
    expert_count = len(similarities)
    if expert_count == 1:
        weights = torch.ones(similarities[0].shape[0], 1)
        components = {
            "rank": weights.t().clone(),
            "hard_negative": weights.t().clone(),
            "complement": torch.zeros(1, similarities[0].shape[0]),
            "uncertainty": torch.zeros(1, similarities[0].shape[0]),
        }
        return weights, components

    k = min(topk, similarities[0].shape[1])
    top_values = []
    top_indices = []
    for similarity in similarities:
        values, indices = torch.topk(similarity, k=k, dim=1, largest=True, sorted=True)
        top_values.append(values)
        top_indices.append(indices)

    rank_scores = torch.zeros(expert_count, similarities[0].shape[0])
    for expert_index in range(expert_count):
        overlaps = []
        for other_index in range(expert_count):
            if expert_index == other_index:
                continue
            overlaps.append(pairwise_topk_overlap(top_indices[expert_index], top_indices[other_index]))
        rank_scores[expert_index] = torch.stack(overlaps, dim=0).mean(dim=0)

    hard_negative_scores = []
    uncertainty_scores = []
    for values in top_values:
        if k > 1:
            hard_negative_scores.append(values[:, 0] - values[:, 1:].mean(dim=1))
        else:
            hard_negative_scores.append(values[:, 0])
        probs = F.softmax(values, dim=1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
        uncertainty_scores.append(entropy)

    hard_negative_scores = torch.stack(hard_negative_scores, dim=0)
    uncertainty_scores = torch.stack(uncertainty_scores, dim=0)

    rank_norm = normalize_by_query(rank_scores)
    hardneg_norm = normalize_by_query(hard_negative_scores)
    uncertainty_norm = normalize_by_query(uncertainty_scores)
    complement_norm = (1.0 - rank_norm) * hardneg_norm

    raw = (
        rank_weight * rank_norm
        + hardneg_weight * hardneg_norm
        + complement_weight * complement_norm
        - uncertainty_weight * uncertainty_norm
    )
    weights = F.softmax((raw / max(temperature, 1e-6)).t(), dim=1)
    components = {
        "rank": rank_norm,
        "hard_negative": hardneg_norm,
        "complement": complement_norm,
        "uncertainty": uncertainty_norm,
        "raw": raw,
    }
    return weights, components


def fuse_similarities(similarities, weights):
    stacked = torch.stack(similarities, dim=0)
    return (stacked * weights.t().unsqueeze(-1)).sum(dim=0)


def parse_fixed_weights(raw_weights, expert_count):
    if not raw_weights:
        return None
    weights = [float(item) for item in raw_weights.split(",")]
    if len(weights) != expert_count:
        raise ValueError(f"Expected {expert_count} fixed weights, got {len(weights)}.")
    tensor = torch.tensor(weights, dtype=torch.float32)
    if torch.any(tensor < 0):
        raise ValueError("Fixed weights must be non-negative.")
    total = tensor.sum().item()
    if total <= 0:
        raise ValueError("At least one fixed weight must be positive.")
    return tensor / total


def save_markdown_report(path, payload):
    lines = [
        "# GAR-EM Score Fusion Report",
        "",
        f"- expert_config: `{payload['expert_config']}`",
        f"- topk: `{payload['topk']}`",
        f"- adaptive_temperature: `{payload['adaptive_temperature']}`",
        "",
        "## Experts",
        "",
        "| name | checkpoint | R1 | R5 | R10 | RSum | mAP | mINP |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for expert in payload["experts"]:
        metrics = expert["metrics"]
        lines.append(
            f"| {expert['name']} | `{expert['checkpoint']}` | "
            f"{metrics['R1']:.3f} | {metrics['R5']:.3f} | {metrics['R10']:.3f} | "
            f"{metrics['RSum']:.3f} | {metrics['mAP']:.3f} | {metrics['mINP']:.3f} |"
        )
    lines.extend(["", "## Fusion", "", "| method | R1 | R5 | R10 | RSum | mAP | mINP |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for name, metrics in payload["fusion"].items():
        lines.append(
            f"| {name} | {metrics['R1']:.3f} | {metrics['R5']:.3f} | {metrics['R10']:.3f} | "
            f"{metrics['RSum']:.3f} | {metrics['mAP']:.3f} | {metrics['mINP']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Adaptive Components",
            "",
            "The first GAR-EM implementation uses rank consistency, hard-negative separability, expert complementarity, and retrieval uncertainty. Cross-view cycle and local-global alignment are reserved for the next data-loader-aware version.",
        ]
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def build_args():
    parser = argparse.ArgumentParser(
        description="Ground-Aerial Retrieval-Aware Expert Merge score-level fusion."
    )
    parser.add_argument("--expert_config", required=True, help="JSON file describing expert configs and checkpoints.")
    parser.add_argument("--output_dir", required=True, help="Directory for report files.")
    parser.add_argument("--root_dir", default="", help="Override dataset root for all experts.")
    parser.add_argument("--test_batch_size", type=int, default=0, help="Override test batch size.")
    parser.add_argument("--device", default="cuda", help="cuda or cpu.")
    parser.add_argument("--topk", type=int, default=10, help="Top-k used for rank consistency and hard negatives.")
    parser.add_argument("--fixed_weights", default="", help="Comma-separated fixed fusion weights.")
    parser.add_argument("--rank_weight", type=float, default=1.0)
    parser.add_argument("--hardneg_weight", type=float, default=0.5)
    parser.add_argument("--complement_weight", type=float, default=0.3)
    parser.add_argument("--uncertainty_weight", type=float, default=0.2)
    parser.add_argument("--adaptive_temperature", type=float, default=0.5)
    return parser.parse_args()


def main():
    cli_args = build_args()
    mkdir_if_missing(cli_args.output_dir)
    device = torch.device(cli_args.device if torch.cuda.is_available() or cli_args.device == "cpu" else "cpu")
    expert_specs = load_expert_specs(cli_args.expert_config)

    similarities = []
    expert_rows = []
    reference_qids = None
    reference_gids = None

    for spec in expert_specs:
        args = load_train_configs(spec["config_file"])
        args = apply_overrides(args, spec, cli_args)
        img_loader, txt_loader, num_classes = build_dataloader(args)
        model = build_finetune_model(args, num_classes=num_classes).to(device)
        state = load_checkpoint_state(spec["checkpoint"])
        missing, unexpected = model.load_state_dict(state, strict=False)
        similarity, qids, gids = compute_similarity(model, img_loader, txt_loader, device)

        if reference_qids is None:
            reference_qids = qids
            reference_gids = gids
        elif not (torch.equal(reference_qids, qids) and torch.equal(reference_gids, gids)):
            raise ValueError("Experts must evaluate the same query/gallery pid order.")

        metrics = evaluate_similarity(similarity, qids, gids)
        similarities.append(similarity)
        expert_rows.append(
            {
                "name": spec["name"],
                "checkpoint": spec["checkpoint"],
                "config_file": spec["config_file"],
                "pretrain_choice": getattr(args, "pretrain_choice", ""),
                "loss_names": getattr(args, "loss_names", ""),
                "missing_keys": len(missing),
                "unexpected_keys": len(unexpected),
                "metrics": metrics,
            }
        )

    fusion_metrics = {}
    mean_weights = torch.full((similarities[0].shape[0], len(similarities)), 1.0 / len(similarities))
    fusion_metrics["mean"] = evaluate_similarity(
        fuse_similarities(similarities, mean_weights),
        reference_qids,
        reference_gids,
    )

    fixed = parse_fixed_weights(cli_args.fixed_weights, len(similarities))
    if fixed is not None:
        fixed_weights = fixed.unsqueeze(0).expand(similarities[0].shape[0], -1)
        fusion_metrics["fixed"] = evaluate_similarity(
            fuse_similarities(similarities, fixed_weights),
            reference_qids,
            reference_gids,
        )

    adaptive_weights, components = compute_adaptive_weights(
        similarities,
        cli_args.topk,
        cli_args.rank_weight,
        cli_args.hardneg_weight,
        cli_args.complement_weight,
        cli_args.uncertainty_weight,
        cli_args.adaptive_temperature,
    )
    fusion_metrics["gar_em_adaptive"] = evaluate_similarity(
        fuse_similarities(similarities, adaptive_weights),
        reference_qids,
        reference_gids,
    )

    payload = {
        "expert_config": cli_args.expert_config,
        "topk": cli_args.topk,
        "adaptive_temperature": cli_args.adaptive_temperature,
        "component_weights": {
            "rank": cli_args.rank_weight,
            "hard_negative": cli_args.hardneg_weight,
            "complement": cli_args.complement_weight,
            "uncertainty": cli_args.uncertainty_weight,
        },
        "experts": expert_rows,
        "fusion": fusion_metrics,
        "adaptive_weight_mean": adaptive_weights.mean(dim=0).tolist(),
        "adaptive_component_mean": {
            key: value.mean(dim=1).tolist()
            for key, value in components.items()
            if torch.is_tensor(value) and value.dim() == 2
        },
    }

    json_path = op.join(cli_args.output_dir, "gar_em_score_fusion.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    save_markdown_report(op.join(cli_args.output_dir, "gar_em_score_fusion.md"), payload)

    print(json.dumps({"report": json_path, "fusion": fusion_metrics}, indent=2))


if __name__ == "__main__":
    main()
