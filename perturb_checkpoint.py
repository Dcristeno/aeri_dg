import argparse
import os
import os.path as op
from collections import OrderedDict

import torch


def parse_float_list(values):
    if len(values) == 1 and "," in values[0]:
        values = values[0].split(",")
    return [float(value) for value in values]


def parse_int_list(values):
    if len(values) == 1 and "," in values[0]:
        values = values[0].split(",")
    return [int(value) for value in values]


def value_tag(value):
    return f"{value:.6g}".replace("-", "m").replace(".", "p")


def load_model_state(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    cleaned = OrderedDict()
    for key, value in state.items():
        key = key[7:] if key.startswith("module.") else key
        cleaned[key] = value
    return cleaned


def should_perturb_key(key, include_prefixes, exclude_prefixes, exclude_keywords):
    if include_prefixes and not any(key.startswith(prefix) for prefix in include_prefixes):
        return False
    if exclude_prefixes and any(key.startswith(prefix) for prefix in exclude_prefixes):
        return False
    if exclude_keywords and any(keyword in key for keyword in exclude_keywords):
        return False
    return True


def perturb_state(
    state,
    sigma,
    seed,
    include_prefixes=None,
    exclude_prefixes=None,
    exclude_keywords=None,
    skip_1d=True,
):
    include_prefixes = include_prefixes or []
    exclude_prefixes = exclude_prefixes or []
    exclude_keywords = exclude_keywords or []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    perturbed = OrderedDict()
    stats = {
        "perturbed_keys": 0,
        "perturbed_numel": 0,
        "skipped_by_scope": 0,
        "skipped_non_floating": 0,
        "skipped_1d": 0,
        "skipped_zero_std": 0,
    }

    for key, value in state.items():
        if not torch.is_tensor(value):
            perturbed[key] = value
            continue

        if not should_perturb_key(key, include_prefixes, exclude_prefixes, exclude_keywords):
            perturbed[key] = value.clone()
            stats["skipped_by_scope"] += 1
            continue

        if not torch.is_floating_point(value):
            perturbed[key] = value.clone()
            stats["skipped_non_floating"] += 1
            continue

        if skip_1d and value.ndim <= 1:
            perturbed[key] = value.clone()
            stats["skipped_1d"] += 1
            continue

        value_float = value.float()
        scale = value_float.std(unbiased=False)
        if scale.item() == 0.0:
            perturbed[key] = value.clone()
            stats["skipped_zero_std"] += 1
            continue

        noise = torch.randn(value_float.shape, generator=generator, dtype=value_float.dtype) * (sigma * scale)
        perturbed[key] = (value_float + noise).to(dtype=value.dtype)
        stats["perturbed_keys"] += 1
        stats["perturbed_numel"] += value.numel()

    return perturbed, stats


def build_args():
    parser = argparse.ArgumentParser(description="Add small random perturbations to checkpoint weights.")
    parser.add_argument("--checkpoint", required=True, help="Input checkpoint path.")
    parser.add_argument("--output_dir", required=True, help="Directory for perturbed checkpoints.")
    parser.add_argument("--sigmas", nargs="+", required=True, help="Relative noise strengths, e.g. 0.0005 0.001.")
    parser.add_argument("--seeds", nargs="+", required=True, help="Noise random seeds, e.g. 1 2 3.")
    parser.add_argument("--include_prefix", nargs="*", default=[], help="Only perturb keys with these prefixes, e.g. base_model.")
    parser.add_argument("--exclude_prefix", nargs="*", default=[], help="Never perturb keys with these prefixes.")
    parser.add_argument("--exclude_keyword", nargs="*", default=[], help="Never perturb keys containing these substrings.")
    parser.add_argument("--include_1d", action="store_true", help="Also perturb 1D tensors such as bias and LayerNorm weights.")
    return parser.parse_args()


def main():
    args = build_args()
    os.makedirs(args.output_dir, exist_ok=True)

    state = load_model_state(args.checkpoint)
    sigmas = parse_float_list(args.sigmas)
    seeds = parse_int_list(args.seeds)

    for sigma in sigmas:
        for seed in seeds:
            perturbed, stats = perturb_state(
                state,
                sigma=sigma,
                seed=seed,
                include_prefixes=args.include_prefix,
                exclude_prefixes=args.exclude_prefix,
                exclude_keywords=args.exclude_keyword,
                skip_1d=not args.include_1d,
            )
            output_path = op.join(args.output_dir, f"noise_sigma_{value_tag(sigma)}_seed_{seed}.pth")
            torch.save(
                {
                    "model": perturbed,
                    "source_checkpoint": args.checkpoint,
                    "sigma": sigma,
                    "seed": seed,
                    "include_prefix": args.include_prefix,
                    "exclude_prefix": args.exclude_prefix,
                    "exclude_keyword": args.exclude_keyword,
                    "include_1d": args.include_1d,
                    "stats": stats,
                },
                output_path,
            )
            print(f"saved {output_path} | {stats}")


if __name__ == "__main__":
    main()
