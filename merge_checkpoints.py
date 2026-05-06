import argparse
import json
import os
import re
from collections import OrderedDict

import torch


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu")
    if "model" not in checkpoint:
        raise KeyError(f"{path} does not contain a 'model' state dict")
    return checkpoint


def should_skip(key, patterns):
    return any(re.search(pattern, key) for pattern in patterns)


def merge_state_dicts(base_state, other_state, alpha, skip_patterns):
    merged = OrderedDict()
    stats = {
        "merged": 0,
        "kept_base_missing": 0,
        "kept_base_shape": 0,
        "kept_base_dtype": 0,
        "kept_base_skipped": 0,
    }

    for key, base_value in base_state.items():
        if should_skip(key, skip_patterns):
            merged[key] = base_value
            stats["kept_base_skipped"] += 1
            continue

        if key not in other_state:
            merged[key] = base_value
            stats["kept_base_missing"] += 1
            continue

        other_value = other_state[key]
        if base_value.shape != other_value.shape:
            merged[key] = base_value
            stats["kept_base_shape"] += 1
            continue

        if not torch.is_floating_point(base_value) or not torch.is_floating_point(other_value):
            merged[key] = base_value
            stats["kept_base_dtype"] += 1
            continue

        merged[key] = (1.0 - alpha) * base_value + alpha * other_value.to(base_value.dtype)
        stats["merged"] += 1

    return merged, stats


def main():
    parser = argparse.ArgumentParser(description="Linearly merge two compatible checkpoints.")
    parser.add_argument("--base", required=True, help="Checkpoint kept as the target structure")
    parser.add_argument("--other", required=True, help="Checkpoint interpolated into --base")
    parser.add_argument("--alpha", required=True, type=float, help="Interpolation coefficient for --other")
    parser.add_argument("--output", required=True, help="Output checkpoint path")
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        help="Regex pattern for parameter keys to keep from --base without merging; can be passed multiple times",
    )
    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError(f"--alpha must be in [0, 1], got {args.alpha}")

    base_checkpoint = load_checkpoint(args.base)
    other_checkpoint = load_checkpoint(args.other)
    merged_state, stats = merge_state_dicts(
        base_checkpoint["model"],
        other_checkpoint["model"],
        args.alpha,
        args.skip,
    )

    output_checkpoint = dict(base_checkpoint)
    output_checkpoint["model"] = merged_state
    output_checkpoint["merge_info"] = {
        "base": os.path.abspath(args.base),
        "other": os.path.abspath(args.other),
        "alpha": args.alpha,
        "skip": args.skip,
        "stats": stats,
    }

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(output_checkpoint, args.output)
    print(json.dumps(output_checkpoint["merge_info"], indent=2))


if __name__ == "__main__":
    main()
