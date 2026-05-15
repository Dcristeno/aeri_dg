import argparse
import json
import os
import os.path as op
from collections import OrderedDict

import torch
import torch.nn.functional as F

from datasets import build_dataloader
from model.build_finetune import build_finetune_model
from utils.iotools import load_train_configs
from utils.logger import setup_logger


def normalize_loss_names(loss_names):
    tokens = [token.strip() for token in loss_names.split("+") if token.strip()]
    normalized = []
    alias_map = {
        "fa": "fta",
        "g2a": "bridge",
        "ga": "bridge",
        "ga_bridge": "bridge",
        "bridge_loss": "bridge",
    }
    finetune_alias_seen = any(
        token in {"fa", "fta", "cda", "bridge", "g2a", "ga", "ga_bridge"}
        for token in tokens
    )
    for token in tokens:
        token = alias_map.get(token, token)
        if finetune_alias_seen and token == "sdm":
            token = "cda"
        if token not in normalized:
            normalized.append(token)
    return "+".join(normalized)


def parse_alphas(alpha_values):
    if len(alpha_values) == 1 and "," in alpha_values[0]:
        alpha_values = alpha_values[0].split(",")
    alphas = sorted({float(value) for value in alpha_values})
    if not alphas:
        raise ValueError("At least one alpha is required.")
    for alpha in alphas:
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError(f"Alpha must be in [0, 1], got {alpha}.")
    return alphas


def alpha_tag(alpha):
    tag = f"{alpha:.4f}".rstrip("0").rstrip(".")
    return tag.replace("-", "m").replace(".", "p")


def load_model_state(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    cleaned = OrderedDict()
    for key, value in state.items():
        key = key[7:] if key.startswith("module.") else key
        cleaned[key] = value
    return cleaned


def merge_states(base_state, other_state, alpha):
    merged = OrderedDict()
    stats = {
        "merged_keys": 0,
        "merged_numel": 0,
        "kept_base_keys": 0,
        "missing_in_other": 0,
        "shape_mismatch": 0,
        "non_floating": 0,
        "other_only_keys": 0,
    }

    for key, base_value in base_state.items():
        other_value = other_state.get(key)
        can_merge = (
            other_value is not None
            and torch.is_tensor(base_value)
            and torch.is_tensor(other_value)
            and base_value.shape == other_value.shape
            and torch.is_floating_point(base_value)
            and torch.is_floating_point(other_value)
        )
        if can_merge:
            mixed = torch.lerp(base_value.float(), other_value.float(), alpha)
            merged[key] = mixed.to(dtype=base_value.dtype)
            stats["merged_keys"] += 1
            stats["merged_numel"] += base_value.numel()
            continue

        if torch.is_tensor(base_value):
            merged[key] = base_value.clone()
        else:
            merged[key] = base_value
        stats["kept_base_keys"] += 1
        if other_value is None:
            stats["missing_in_other"] += 1
        elif torch.is_tensor(base_value) and torch.is_tensor(other_value) and base_value.shape != other_value.shape:
            stats["shape_mismatch"] += 1
        else:
            stats["non_floating"] += 1

    stats["other_only_keys"] = len([key for key in other_state.keys() if key not in base_state])
    return merged, stats


@torch.no_grad()
def compute_retrieval_signature(model, img_loader, txt_loader, device, topk, max_text_batches=0, max_image_batches=0):
    model.eval()
    qfeats, gfeats = [], []

    for batch_idx, (_, caption) in enumerate(txt_loader):
        if max_text_batches and batch_idx >= max_text_batches:
            break
        caption = caption.to(device)
        qfeats.append(model.encode_text(caption).float().cpu())

    for batch_idx, (_, image) in enumerate(img_loader):
        if max_image_batches and batch_idx >= max_image_batches:
            break
        image = image.to(device)
        gfeats.append(model.encode_image(image).float().cpu())

    if not qfeats or not gfeats:
        raise RuntimeError("Empty retrieval features. Check dataloaders or max batch limits.")

    qfeats = F.normalize(torch.cat(qfeats, dim=0).to(device), p=2, dim=1)
    gfeats = F.normalize(torch.cat(gfeats, dim=0).to(device), p=2, dim=1)
    similarity = qfeats @ gfeats.t()
    k = min(topk, similarity.shape[1])
    values, indices = torch.topk(similarity, k=k, dim=1, largest=True, sorted=True)
    if k > 1:
        margin = (values[:, 0] - values[:, 1]).mean().item()
    else:
        margin = values[:, 0].mean().item()
    return indices.cpu(), margin


def topk_overlap(first, second):
    if first.shape != second.shape:
        raise ValueError(f"Top-k tensors must share shape, got {first.shape} and {second.shape}.")
    overlap = (first.unsqueeze(2) == second.unsqueeze(1)).any(dim=2).float().mean()
    return overlap.item()


def score_signatures(alphas, signatures, margins, confidence_weight):
    rows = []
    for index, alpha in enumerate(alphas):
        neighbor_scores = []
        if index > 0:
            neighbor_scores.append(topk_overlap(signatures[index], signatures[index - 1]))
        if index + 1 < len(alphas):
            neighbor_scores.append(topk_overlap(signatures[index], signatures[index + 1]))
        consistency = sum(neighbor_scores) / len(neighbor_scores) if neighbor_scores else 1.0
        score = consistency + confidence_weight * margins[index]
        rows.append(
            {
                "alpha": alpha,
                "consistency": consistency,
                "margin": margins[index],
                "score": score,
            }
        )
    return rows


def build_args():
    parser = argparse.ArgumentParser(
        description="AdaMMS-style adaptive checkpoint merge for CFAN/AERI retrieval models."
    )
    parser.add_argument("--config_file", required=True, help="Path to a saved configs.yaml.")
    parser.add_argument("--base_checkpoint", required=True, help="Checkpoint used as alpha=0.")
    parser.add_argument("--other_checkpoint", required=True, help="Checkpoint used as alpha=1.")
    parser.add_argument("--output_dir", required=True, help="Directory for merged checkpoint and score files.")
    parser.add_argument("--root_dir", default="", help="Override dataset root_dir from config.")
    parser.add_argument("--loss_names", default="", help="Override loss names used to build the finetune model.")
    parser.add_argument("--alphas", nargs="+", default=["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"])
    parser.add_argument("--topk", type=int, default=10, help="Top-k retrieval list used for consistency.")
    parser.add_argument("--confidence_weight", type=float, default=0.0, help="Optional weight for top1-top2 margin.")
    parser.add_argument("--max_text_batches", type=int, default=0, help="Use a subset for quick search; 0 means full test text set.")
    parser.add_argument("--max_image_batches", type=int, default=0, help="Use a subset for quick search; 0 means full test image set.")
    parser.add_argument("--device", default="cuda", help="Device for feature extraction.")
    parser.add_argument("--save_all", action="store_true", help="Save every alpha checkpoint, not only the selected one.")
    return parser.parse_args()


def main():
    cli_args = build_args()
    os.makedirs(cli_args.output_dir, exist_ok=True)

    args = load_train_configs(cli_args.config_file)
    args.training = False
    args.output_dir = cli_args.output_dir
    if cli_args.root_dir:
        args.root_dir = cli_args.root_dir
    if cli_args.loss_names:
        args.loss_names = cli_args.loss_names
    args.loss_names = normalize_loss_names(args.loss_names)

    logger = setup_logger("IRRA", save_dir=cli_args.output_dir, if_train=False)
    logger.info(args)
    logger.info(f"Base checkpoint: {cli_args.base_checkpoint}")
    logger.info(f"Other checkpoint: {cli_args.other_checkpoint}")

    device = torch.device(cli_args.device if torch.cuda.is_available() or cli_args.device == "cpu" else "cpu")
    alphas = parse_alphas(cli_args.alphas)
    base_state = load_model_state(cli_args.base_checkpoint)
    other_state = load_model_state(cli_args.other_checkpoint)

    img_loader, txt_loader, num_classes = build_dataloader(args)
    model = build_finetune_model(args, num_classes=num_classes).to(device)

    signatures, margins, merge_stats = [], [], {}
    for alpha in alphas:
        logger.info(f"Evaluating merged alpha={alpha:.4f}")
        state, stats = merge_states(base_state, other_state, alpha)
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(
            f"alpha={alpha:.4f}, merged_keys={stats['merged_keys']}, "
            f"missing_model_keys={len(missing)}, unexpected_checkpoint_keys={len(unexpected)}"
        )
        signature, margin = compute_retrieval_signature(
            model,
            img_loader,
            txt_loader,
            device,
            cli_args.topk,
            cli_args.max_text_batches,
            cli_args.max_image_batches,
        )
        signatures.append(signature)
        margins.append(margin)
        merge_stats[str(alpha)] = stats
        if cli_args.save_all:
            save_path = op.join(cli_args.output_dir, f"merged_alpha_{alpha_tag(alpha)}.pth")
            torch.save({"model": state, "alpha": alpha, "merge_stats": stats}, save_path)

    score_rows = score_signatures(alphas, signatures, margins, cli_args.confidence_weight)
    best_row = max(score_rows, key=lambda row: (row["score"], row["consistency"], -abs(row["alpha"] - 0.5)))
    best_alpha = best_row["alpha"]
    best_state, best_stats = merge_states(base_state, other_state, best_alpha)

    best_path = op.join(cli_args.output_dir, "best_adaptive_merge.pth")
    torch.save(
        {
            "model": best_state,
            "alpha": best_alpha,
            "base_checkpoint": cli_args.base_checkpoint,
            "other_checkpoint": cli_args.other_checkpoint,
            "scores": score_rows,
            "merge_stats": best_stats,
        },
        best_path,
    )

    payload = {
        "best_alpha": best_alpha,
        "best_checkpoint": best_path,
        "base_checkpoint": cli_args.base_checkpoint,
        "other_checkpoint": cli_args.other_checkpoint,
        "topk": cli_args.topk,
        "confidence_weight": cli_args.confidence_weight,
        "max_text_batches": cli_args.max_text_batches,
        "max_image_batches": cli_args.max_image_batches,
        "scores": score_rows,
        "merge_stats": merge_stats,
    }
    with open(op.join(cli_args.output_dir, "adaptive_merge_scores.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    logger.info(f"Best alpha: {best_alpha:.4f}")
    logger.info(f"Saved best adaptive merge checkpoint to {best_path}")


if __name__ == "__main__":
    main()
