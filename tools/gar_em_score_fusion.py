import argparse
import json
import os
import os.path as op
import sys
from collections import OrderedDict

import torch
import torch.nn.functional as F

"""
Contribution 3: score-level GAR-EM fusion.

This tool evaluates individual experts, fixed score fusion, and retrieval-aware
adaptive/prior-adaptive fusion. The current paper result uses the five-expert
pool documented in `docs/merge/configs/currentbest_plus_hardneg.example.json`.
"""

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
        if "score_cache" in expert:
            if "checkpoint" in expert or "config_file" in expert:
                raise ValueError(f"Expert {index} should use either `score_cache` or model fields, not both.")
        else:
            if "checkpoint" not in expert:
                raise ValueError(f"Expert {index} is missing `checkpoint`.")
            if "config_file" not in expert:
                raise ValueError(f"Expert {index} is missing `config_file`.")
        expert.setdefault("name", f"expert_{index}")
    return experts


def load_score_cache(path):
    payload = torch.load(path, map_location="cpu")
    required = ("similarity", "qids", "gids")
    for key in required:
        if key not in payload:
            raise ValueError(f"Score cache {path} is missing `{key}`.")
    return payload


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


def save_score_cache(path, similarity, qids, gids, metadata):
    torch.save(
        {
            "similarity": similarity.cpu(),
            "qids": qids.cpu(),
            "gids": gids.cpu(),
            "metadata": metadata,
        },
        path,
    )


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
    prior_weights=None,
    prior_strength=0.0,
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
    logits = (raw / max(temperature, 1e-6)).t()
    if prior_weights is not None and prior_strength > 0:
        prior = prior_weights.to(logits.device).float().clamp_min(1e-12)
        prior = prior / prior.sum()
        logits = logits + prior_strength * torch.log(prior).unsqueeze(0)
    weights = F.softmax(logits, dim=1)
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


def parse_weight_vector(raw_weights, expert_count, name):
    if not raw_weights:
        return None
    weights = [float(item) for item in raw_weights.split(",")]
    if len(weights) != expert_count:
        raise ValueError(f"Expected {expert_count} {name}, got {len(weights)}.")
    tensor = torch.tensor(weights, dtype=torch.float32)
    if torch.any(tensor < 0):
        raise ValueError(f"{name} must be non-negative.")
    total = tensor.sum().item()
    if total <= 0:
        raise ValueError(f"At least one {name} value must be positive.")
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
    parser.add_argument("--prior_weights", default="", help="Comma-separated expert reliability prior for prior-adaptive fusion.")
    parser.add_argument("--prior_strength", type=float, default=1.0, help="Strength of prior_weights in GAR-EM prior-adaptive fusion.")
    parser.add_argument("--rank_weight", type=float, default=1.0)
    parser.add_argument("--hardneg_weight", type=float, default=0.5)
    parser.add_argument("--complement_weight", type=float, default=0.3)
    parser.add_argument("--uncertainty_weight", type=float, default=0.2)
    parser.add_argument("--adaptive_temperature", type=float, default=0.5)
    parser.add_argument(
        "--save_fusion_cache",
        default="",
        help="Comma-separated fusion methods to save as score caches, e.g. fixed,gar_em_prior_adaptive.",
    )
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
        if "score_cache" in spec:
            cache = load_score_cache(spec["score_cache"])
            similarity = cache["similarity"].float().cpu()
            qids = cache["qids"].cpu()
            gids = cache["gids"].cpu()
            metrics = evaluate_similarity(similarity, qids, gids)
            missing, unexpected = [], []
            pretrain_choice = spec.get("pretrain_choice", "score_cache")
            loss_names = spec.get("loss_names", "score_cache")
        else:
            args = load_train_configs(spec["config_file"])
            args = apply_overrides(args, spec, cli_args)
            img_loader, txt_loader, num_classes = build_dataloader(args)
            model = build_finetune_model(args, num_classes=num_classes).to(device)
            state = load_checkpoint_state(spec["checkpoint"])
            missing, unexpected = model.load_state_dict(state, strict=False)
            similarity, qids, gids = compute_similarity(model, img_loader, txt_loader, device)
            metrics = evaluate_similarity(similarity, qids, gids)
            pretrain_choice = getattr(args, "pretrain_choice", "")
            loss_names = getattr(args, "loss_names", "")

        if reference_qids is None:
            reference_qids = qids
            reference_gids = gids
        elif not (torch.equal(reference_qids, qids) and torch.equal(reference_gids, gids)):
            raise ValueError("Experts must evaluate the same query/gallery pid order.")

        similarities.append(similarity)
        expert_rows.append(
            {
                "name": spec["name"],
                "checkpoint": spec.get("checkpoint", spec.get("score_cache", "")),
                "config_file": spec.get("config_file", ""),
                "score_cache": spec.get("score_cache", ""),
                "pretrain_choice": pretrain_choice,
                "loss_names": loss_names,
                "missing_keys": len(missing),
                "unexpected_keys": len(unexpected),
                "metrics": metrics,
            }
        )

    fusion_metrics = {}
    fusion_similarities = {}
    mean_weights = torch.full((similarities[0].shape[0], len(similarities)), 1.0 / len(similarities))
    mean_similarity = fuse_similarities(similarities, mean_weights)
    fusion_similarities["mean"] = mean_similarity
    fusion_metrics["mean"] = evaluate_similarity(mean_similarity, reference_qids, reference_gids)

    fixed = parse_weight_vector(cli_args.fixed_weights, len(similarities), "fixed weights")
    if fixed is not None:
        fixed_weights = fixed.unsqueeze(0).expand(similarities[0].shape[0], -1)
        fixed_similarity = fuse_similarities(similarities, fixed_weights)
        fusion_similarities["fixed"] = fixed_similarity
        fusion_metrics["fixed"] = evaluate_similarity(fixed_similarity, reference_qids, reference_gids)

    adaptive_weights, components = compute_adaptive_weights(
        similarities,
        cli_args.topk,
        cli_args.rank_weight,
        cli_args.hardneg_weight,
        cli_args.complement_weight,
        cli_args.uncertainty_weight,
        cli_args.adaptive_temperature,
    )
    adaptive_similarity = fuse_similarities(similarities, adaptive_weights)
    fusion_similarities["gar_em_adaptive"] = adaptive_similarity
    fusion_metrics["gar_em_adaptive"] = evaluate_similarity(adaptive_similarity, reference_qids, reference_gids)

    prior = parse_weight_vector(cli_args.prior_weights, len(similarities), "prior weights")
    prior_adaptive_weights = None
    if prior is not None:
        prior_adaptive_weights, prior_components = compute_adaptive_weights(
            similarities,
            cli_args.topk,
            cli_args.rank_weight,
            cli_args.hardneg_weight,
            cli_args.complement_weight,
            cli_args.uncertainty_weight,
            cli_args.adaptive_temperature,
            prior_weights=prior,
            prior_strength=cli_args.prior_strength,
        )
        prior_adaptive_similarity = fuse_similarities(similarities, prior_adaptive_weights)
        fusion_similarities["gar_em_prior_adaptive"] = prior_adaptive_similarity
        fusion_metrics["gar_em_prior_adaptive"] = evaluate_similarity(
            prior_adaptive_similarity,
            reference_qids,
            reference_gids,
        )

    payload = {
        "expert_config": cli_args.expert_config,
        "topk": cli_args.topk,
        "adaptive_temperature": cli_args.adaptive_temperature,
        "prior_weights": prior.tolist() if prior is not None else None,
        "prior_strength": cli_args.prior_strength,
        "component_weights": {
            "rank": cli_args.rank_weight,
            "hard_negative": cli_args.hardneg_weight,
            "complement": cli_args.complement_weight,
            "uncertainty": cli_args.uncertainty_weight,
        },
        "experts": expert_rows,
        "fusion": fusion_metrics,
        "adaptive_weight_mean": adaptive_weights.mean(dim=0).tolist(),
        "prior_adaptive_weight_mean": prior_adaptive_weights.mean(dim=0).tolist()
        if prior_adaptive_weights is not None
        else None,
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

    cache_methods = [item.strip() for item in cli_args.save_fusion_cache.split(",") if item.strip()]
    saved_caches = {}
    for method in cache_methods:
        if method not in fusion_similarities:
            raise ValueError(
                f"Cannot save unknown fusion method `{method}`. "
                f"Available methods: {sorted(fusion_similarities.keys())}"
            )
        cache_path = op.join(cli_args.output_dir, f"{method}_score_cache.pth")
        save_score_cache(
            cache_path,
            fusion_similarities[method],
            reference_qids,
            reference_gids,
            {
                "method": method,
                "expert_config": cli_args.expert_config,
                "metrics": fusion_metrics[method],
                "experts": expert_rows,
                "fixed_weights": fixed.tolist() if fixed is not None else None,
                "prior_weights": prior.tolist() if prior is not None else None,
                "prior_strength": cli_args.prior_strength,
                "topk": cli_args.topk,
                "adaptive_temperature": cli_args.adaptive_temperature,
            },
        )
        saved_caches[method] = cache_path

    print(json.dumps({"report": json_path, "fusion": fusion_metrics, "score_caches": saved_caches}, indent=2))


if __name__ == "__main__":
    main()
