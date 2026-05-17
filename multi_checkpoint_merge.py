import argparse
import json
import os
import os.path as op
from collections import OrderedDict

import torch


def load_model_state(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    cleaned = OrderedDict()
    for key, value in state.items():
        key = key[7:] if key.startswith("module.") else key
        cleaned[key] = value
    return cleaned


def should_merge_key(key, include_prefixes=None, exclude_prefixes=None, exclude_keywords=None):
    include_prefixes = include_prefixes or []
    exclude_prefixes = exclude_prefixes or []
    exclude_keywords = exclude_keywords or []
    if include_prefixes and not any(key.startswith(prefix) for prefix in include_prefixes):
        return False
    if exclude_prefixes and any(key.startswith(prefix) for prefix in exclude_prefixes):
        return False
    if exclude_keywords and any(keyword in key for keyword in exclude_keywords):
        return False
    return True


def parse_weight_list(value):
    if isinstance(value, str):
        parts = value.split(",") if "," in value else value.split()
    else:
        parts = value
    weights = [float(part) for part in parts if str(part).strip()]
    if not weights:
        raise ValueError("Weight list cannot be empty.")
    if any(weight < 0 for weight in weights):
        raise ValueError(f"Weights must be non-negative, got {weights}.")
    total = sum(weights)
    if total <= 0:
        raise ValueError(f"At least one weight must be positive, got {weights}.")
    return [weight / total for weight in weights]


def weight_tag(weights):
    tags = []
    for weight in weights:
        tag = f"{weight:.4f}".rstrip("0").rstrip(".")
        tags.append(tag.replace(".", "p"))
    return "w" + "_".join(tags)


def merge_multiple_states(states, weights, include_prefixes=None, exclude_prefixes=None, exclude_keywords=None):
    base_state = states[0]
    merged = OrderedDict()
    stats = {
        "merged_keys": 0,
        "merged_numel": 0,
        "kept_base_keys": 0,
        "skipped_by_scope": 0,
        "missing_in_checkpoint": 0,
        "shape_mismatch": 0,
        "non_floating": 0,
        "other_only_keys": 0,
    }
    base_keys = set(base_state.keys())
    all_other_keys = set().union(*(set(state.keys()) for state in states[1:])) if len(states) > 1 else set()
    stats["other_only_keys"] = len(all_other_keys - base_keys)

    for key, base_value in base_state.items():
        in_scope = should_merge_key(key, include_prefixes, exclude_prefixes, exclude_keywords)
        values = [state.get(key) for state in states]
        can_merge = in_scope and torch.is_tensor(base_value) and torch.is_floating_point(base_value)
        if can_merge:
            for value in values:
                if value is None:
                    stats["missing_in_checkpoint"] += 1
                    can_merge = False
                    break
                if not torch.is_tensor(value) or not torch.is_floating_point(value):
                    stats["non_floating"] += 1
                    can_merge = False
                    break
                if value.shape != base_value.shape:
                    stats["shape_mismatch"] += 1
                    can_merge = False
                    break
        if can_merge:
            mixed = torch.zeros_like(base_value, dtype=torch.float32)
            for weight, value in zip(weights, values):
                mixed.add_(value.float(), alpha=weight)
            merged[key] = mixed.to(dtype=base_value.dtype)
            stats["merged_keys"] += 1
            stats["merged_numel"] += base_value.numel()
            continue
        merged[key] = base_value.clone() if torch.is_tensor(base_value) else base_value
        stats["kept_base_keys"] += 1
        if not in_scope:
            stats["skipped_by_scope"] += 1
        elif torch.is_tensor(base_value) and not torch.is_floating_point(base_value):
            stats["non_floating"] += 1
    return merged, stats


def build_args():
    parser = argparse.ArgumentParser(description="Weighted multi-checkpoint merge for CFAN/AERI models.")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoints to merge. The first checkpoint supplies unmatched parameters.")
    parser.add_argument("--weights", nargs="*", default=[], help="One weight list. Accepts space-separated values or one comma-separated string.")
    parser.add_argument("--weight_sets", nargs="*", default=[], help="Multiple comma-separated weight lists, one merged checkpoint per list.")
    parser.add_argument("--output_dir", required=True, help="Directory for merged checkpoints and manifest.")
    parser.add_argument("--output_name", default="merged_multi", help="Base filename used when saving one weight set.")
    parser.add_argument("--include_prefix", nargs="*", default=[], help="Only merge checkpoint keys with these prefixes, e.g. base_model.")
    parser.add_argument("--exclude_prefix", nargs="*", default=[], help="Never merge checkpoint keys with these prefixes.")
    parser.add_argument("--exclude_keyword", nargs="*", default=[], help="Never merge checkpoint keys containing these substrings.")
    return parser.parse_args()


def main():
    args = build_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.weight_sets:
        weight_sets = [parse_weight_list(item) for item in args.weight_sets]
    elif args.weights:
        weight_sets = [parse_weight_list(args.weights)]
    else:
        weight_sets = [[1.0 / len(args.checkpoints)] * len(args.checkpoints)]
    for weights in weight_sets:
        if len(weights) != len(args.checkpoints):
            raise ValueError(f"Expected {len(args.checkpoints)} weights, got {len(weights)}: {weights}")

    states = [load_model_state(path) for path in args.checkpoints]
    manifest = {
        "checkpoints": args.checkpoints,
        "include_prefix": args.include_prefix,
        "exclude_prefix": args.exclude_prefix,
        "exclude_keyword": args.exclude_keyword,
        "outputs": [],
    }
    for index, weights in enumerate(weight_sets):
        merged_state, stats = merge_multiple_states(
            states,
            weights,
            include_prefixes=args.include_prefix,
            exclude_prefixes=args.exclude_prefix,
            exclude_keywords=args.exclude_keyword,
        )
        if len(weight_sets) == 1:
            output_path = op.join(args.output_dir, f"{args.output_name}.pth")
        else:
            output_path = op.join(args.output_dir, f"{args.output_name}_{weight_tag(weights)}.pth")
        torch.save(
            {
                "model": merged_state,
                "checkpoints": args.checkpoints,
                "weights": weights,
                "include_prefix": args.include_prefix,
                "exclude_prefix": args.exclude_prefix,
                "exclude_keyword": args.exclude_keyword,
                "merge_stats": stats,
            },
            output_path,
        )
        row = {"index": index, "weights": weights, "checkpoint": output_path, "merge_stats": stats}
        manifest["outputs"].append(row)
        print(json.dumps(row, indent=2))
    with open(op.join(args.output_dir, "multi_checkpoint_merge_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    main()
