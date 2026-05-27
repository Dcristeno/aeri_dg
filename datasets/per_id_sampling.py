import random

import numpy as np

from utils.iotools import read_image


"""
Contribution 1: per-ID training sampling.

The paper's k-random module is the `strategy="random"` path with
`samples_per_id=2`. The heuristic strategies below are retained as internal
controls, but the main reported contribution should use random k sampling.
"""

_IMAGE_SHARPNESS_SCORE_CACHE = {}
_IMAGE_DESCRIPTOR_CACHE = {}


def _compute_image_sharpness_score(img_path):
    cached_score = _IMAGE_SHARPNESS_SCORE_CACHE.get(img_path)
    if cached_score is not None:
        return cached_score

    gray = np.asarray(read_image(img_path).convert("L"), dtype=np.float32)
    if gray.size == 0:
        score = 0.0
    else:
        dx = gray[:, 1:] - gray[:, :-1]
        dy = gray[1:, :] - gray[:-1, :]
        score = float(dx.var() + dy.var())

    _IMAGE_SHARPNESS_SCORE_CACHE[img_path] = score
    return score


def _compute_image_descriptor(img_path, descriptor_size=(32, 16)):
    cached_descriptor = _IMAGE_DESCRIPTOR_CACHE.get(img_path)
    if cached_descriptor is not None:
        return cached_descriptor

    image = read_image(img_path).resize(descriptor_size)
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0

    gray_flat = gray.reshape(-1)
    gray_flat = gray_flat - gray_flat.mean()
    gray_norm = np.linalg.norm(gray_flat)
    if gray_norm > 0:
        gray_flat = gray_flat / gray_norm

    color_stats = np.concatenate([rgb.mean(axis=(0, 1)), rgb.std(axis=(0, 1))], axis=0)
    descriptor = np.concatenate([gray_flat, color_stats], axis=0).astype(np.float32)
    descriptor_norm = np.linalg.norm(descriptor)
    if descriptor_norm > 0:
        descriptor = descriptor / descriptor_norm

    _IMAGE_DESCRIPTOR_CACHE[img_path] = descriptor
    return descriptor


def _cosine_similarity(vec_a, vec_b):
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + 1e-12
    return float(np.dot(vec_a, vec_b) / denom)


def _weighted_random_choice(items, weights, rng):
    total_weight = sum(weights)
    if total_weight <= 0:
        return rng.choice(items)

    threshold = rng.random() * total_weight
    cumulative = 0.0
    for item, weight in zip(items, weights):
        cumulative += weight
        if cumulative >= threshold:
            return item
    return items[-1]


def _sample_pid_samples_random_diverse(pid_samples, samples_per_id, rng, diversity_weight=1.0):
    if len(pid_samples) <= samples_per_id:
        if len(pid_samples) == samples_per_id:
            return list(pid_samples)
        expanded = list(pid_samples)
        while len(expanded) < samples_per_id:
            expanded.append(rng.choice(pid_samples))
        return expanded

    descriptors = {sample[1]: _compute_image_descriptor(sample[1]) for sample in pid_samples}

    selected = [rng.choice(pid_samples)]
    remaining = [sample for sample in pid_samples if sample != selected[0]]

    while remaining and len(selected) < samples_per_id:
        weights = []
        for sample in remaining:
            max_similarity = max(
                _cosine_similarity(descriptors[sample[1]], descriptors[selected_sample[1]])
                for selected_sample in selected
            )
            diversity_score = np.clip((1.0 - max_similarity) / 2.0, 1e-6, 1.0)
            weights.append(float(diversity_score ** diversity_weight))

        chosen = _weighted_random_choice(remaining, weights, rng)
        selected.append(chosen)
        remaining.remove(chosen)

    while len(selected) < samples_per_id:
        selected.append(rng.choice(selected))

    return selected


def _select_mid_sharpness_diverse_samples(
    pid_samples,
    samples_per_id,
    mid_ratio=0.6,
    representative_weight=1.0,
    diversity_weight=0.5,
    mid_weight=0.5,
):
    scored_samples = []
    for sample in pid_samples:
        sharpness = _compute_image_sharpness_score(sample[1])
        descriptor = _compute_image_descriptor(sample[1])
        scored_samples.append({
            "sample": sample,
            "sharpness": sharpness,
            "descriptor": descriptor,
        })

    scored_samples.sort(key=lambda item: item["sharpness"])
    num_samples = len(scored_samples)
    candidate_count = max(samples_per_id, int(round(num_samples * mid_ratio)))
    candidate_count = min(candidate_count, num_samples)
    start = max(0, (num_samples - candidate_count) // 2)
    candidates = scored_samples[start:start + candidate_count]

    all_descriptors = np.stack([item["descriptor"] for item in scored_samples], axis=0)
    mean_descriptor = all_descriptors.mean(axis=0)
    mean_descriptor_norm = np.linalg.norm(mean_descriptor)
    if mean_descriptor_norm > 0:
        mean_descriptor = mean_descriptor / mean_descriptor_norm

    median_sharpness = float(np.median([item["sharpness"] for item in scored_samples]))
    sharpness_values = [item["sharpness"] for item in scored_samples]
    sharpness_span = max(max(sharpness_values) - min(sharpness_values), 1e-6)

    for item in candidates:
        item["representative_score"] = _cosine_similarity(item["descriptor"], mean_descriptor)
        sharpness_deviation = abs(item["sharpness"] - median_sharpness) / sharpness_span
        item["mid_score"] = max(0.0, 1.0 - sharpness_deviation)

    selected = []
    remaining = list(candidates)
    while remaining and len(selected) < samples_per_id:
        best_item = None
        best_score = None
        for item in remaining:
            redundancy_penalty = 0.0
            if selected:
                redundancy_penalty = max(
                    _cosine_similarity(item["descriptor"], selected_item["descriptor"])
                    for selected_item in selected
                )

            total_score = (
                representative_weight * item["representative_score"]
                + mid_weight * item["mid_score"]
                - diversity_weight * redundancy_penalty
            )

            if best_score is None or total_score > best_score:
                best_score = total_score
                best_item = item

        selected.append(best_item)
        remaining.remove(best_item)

    if not selected:
        return [sample for sample in pid_samples[:samples_per_id]]

    while len(selected) < samples_per_id:
        selected.append(selected[len(selected) % len(selected)])

    return [item["sample"] for item in selected]


def sample_train_dataset_per_pid(
    dataset,
    samples_per_id,
    epoch_seed=None,
    strategy="random",
    mid_ratio=0.6,
    representative_weight=1.0,
    diversity_weight=0.5,
    mid_weight=0.5,
):
    if samples_per_id <= 0:
        return dataset

    pid_to_samples = {}
    for sample in dataset:
        pid = sample[0]
        pid_to_samples.setdefault(pid, []).append(sample)

    rng = random.Random(epoch_seed) if epoch_seed is not None else random
    sampled_dataset = []
    for pid in sorted(pid_to_samples.keys()):
        pid_samples = pid_to_samples[pid]

        if strategy == "random":
            if len(pid_samples) >= samples_per_id:
                sampled_dataset.extend(rng.sample(pid_samples, samples_per_id))
            else:
                sampled_dataset.extend(rng.choices(pid_samples, k=samples_per_id))
        elif strategy == "random_diverse":
            sampled_dataset.extend(
                _sample_pid_samples_random_diverse(
                    pid_samples,
                    samples_per_id,
                    rng,
                    diversity_weight=diversity_weight,
                )
            )
        elif strategy == "sharpness_topk":
            ranked_pid_samples = sorted(pid_samples, key=lambda sample: _compute_image_sharpness_score(sample[1]), reverse=True)
            if len(ranked_pid_samples) >= samples_per_id:
                sampled_dataset.extend(ranked_pid_samples[:samples_per_id])
            else:
                for idx in range(samples_per_id):
                    sampled_dataset.append(ranked_pid_samples[idx % len(ranked_pid_samples)])
        elif strategy == "mid_sharpness_diverse":
            sampled_dataset.extend(
                _select_mid_sharpness_diverse_samples(
                    pid_samples,
                    samples_per_id,
                    mid_ratio=mid_ratio,
                    representative_weight=representative_weight,
                    diversity_weight=diversity_weight,
                    mid_weight=mid_weight,
                )
            )
        else:
            raise ValueError(f"Unsupported train sampling strategy: {strategy}")

    return sampled_dataset
