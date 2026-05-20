import logging
import os.path as op
import random

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .AGData import AGData, AGDataAttr, AGSGData
from .aeripedes import AERIPEDES
from .agtbpr import AG_ReID
from .bases import ImageDataset, ImageTextMLMDataset, TextDataset
from utils.iotools import read_json


__factory = {
    "AGTBPR": AG_ReID,
    "AERI-PEDES": AERIPEDES,
    "AGData": AGData,
    "AGDataAttr": AGDataAttr,
    "AGSGData": AGSGData,
}


def build_transforms(img_size=(384, 128), aug=False, is_train=True):
    height, width = img_size
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    if not is_train:
        return T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    if aug:
        return T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(scale=(0.02, 0.4), value=mean),
        ])

    return T.Compose([
        T.Resize((height, width)),
        T.RandomHorizontalFlip(0.5),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def collate(batch):
    keys = set(key for item in batch for key in item.keys())
    grouped = {key: [item[key] if key in item else None for item in batch] for key in keys}

    output = {}
    for key, values in grouped.items():
        if isinstance(values[0], int):
            output[key] = torch.tensor(values)
        elif torch.is_tensor(values[0]):
            output[key] = torch.stack(values)
        else:
            raise TypeError(f"Unexpected data type: {type(values[0])} in batch key {key}.")
    return output


def _caption_cherry_candidates(caption):
    caption = " ".join(str(caption).split())
    if not caption:
        return []

    candidates = [
        f"An aerial-to-ground person retrieval description: {caption}",
        f"A pedestrian identity with the following visual attributes: {caption}",
        f"The same person is described as: {caption}",
        f"A cross-view image of a person matching this description: {caption}",
        f"From another camera view, this person can be recognized by: {caption}",
    ]
    seen = set()
    unique = []
    for candidate in candidates:
        normalized = candidate.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(candidate)
    return unique


def _caption_cherry_score(caption, candidate, text_length):
    caption_words = set(str(caption).lower().replace(".", " ").replace(",", " ").split())
    candidate_words = str(candidate).lower().replace(".", " ").replace(",", " ").split()
    candidate_set = set(candidate_words)
    if not caption_words or not candidate_words:
        return 0.0
    retention = len(caption_words & candidate_set) / len(caption_words)
    length_penalty = max(0.0, 1.0 - max(0, len(candidate_words) - text_length + 8) / max(text_length, 1))
    view_bonus = 0.05 if any(token in candidate_set for token in ("aerial", "ground", "cross-view", "camera")) else 0.0
    return retention + length_penalty + view_bonus


def build_caption_cherry_dataset(dataset, mode="off", extra_per_sample=0, cherry_weight=1.0, text_length=77):
    if mode == "off" or extra_per_sample <= 0:
        return dataset
    if mode != "template":
        raise ValueError(f"Unsupported caption_cherry_mode: {mode}")

    expanded = []
    for sample in dataset:
        if len(sample) >= 5 and isinstance(sample[1], int):
            pid, img_path, g_path, caption = sample[0], sample[2], sample[3], sample[4]
        else:
            pid, img_path, g_path, caption = sample[:4]
        expanded.append((pid, img_path, g_path, caption, 1.0))
        candidates = _caption_cherry_candidates(caption)
        candidates = sorted(
            candidates,
            key=lambda candidate: _caption_cherry_score(caption, candidate, text_length),
            reverse=True,
        )
        for candidate in candidates[:extra_per_sample]:
            expanded.append((pid, img_path, g_path, candidate, cherry_weight))
    return expanded


def _resolve_synthetic_path(path, root_dir, dataset_name):
    if op.isabs(path):
        return path
    dataset_dir = op.join(root_dir, dataset_name)
    candidate = op.join(dataset_dir, path)
    if op.exists(candidate):
        return candidate
    return op.join(root_dir, path)


def build_synthetic_cherry_dataset(args, dataset):
    manifest_path = getattr(args, "synthetic_cherry_manifest", "")
    if not manifest_path:
        return dataset, 0
    if not op.exists(manifest_path):
        raise RuntimeError(f"Synthetic cherry manifest not found: {manifest_path}")

    records = read_json(manifest_path)
    if isinstance(records, dict):
        records = records.get("samples", [])
    if not isinstance(records, list):
        raise ValueError("Synthetic cherry manifest must be a list or a dict with a 'samples' list.")

    min_score = float(args.synthetic_cherry_min_score)
    max_per_pid = int(args.synthetic_cherry_max_per_pid)
    cherry_weight = float(args.synthetic_cherry_weight)

    pid_to_records = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        score = float(record.get("score", record.get("clis", 1.0)))
        if score < min_score:
            continue
        if "pid" not in record:
            continue
        pid = int(record["pid"])
        aerial_path = record.get("aerial_img") or record.get("aerial_img_path") or record.get("image_path")
        ground_path = record.get("ground_img") or record.get("ground_img_path") or record.get("image_path")
        caption = record.get("caption") or record.get("text") or record.get("prompt")
        if not aerial_path or not ground_path or not caption:
            continue
        pid_to_records.setdefault(pid, []).append((score, aerial_path, ground_path, caption))

    synthetic_samples = []
    for pid in sorted(pid_to_records.keys()):
        ranked = sorted(pid_to_records[pid], key=lambda item: item[0], reverse=True)
        if max_per_pid > 0:
            ranked = ranked[:max_per_pid]
        for score, aerial_path, ground_path, caption in ranked:
            sample_weight = cherry_weight * max(score, 1e-6)
            synthetic_samples.append((
                pid,
                _resolve_synthetic_path(aerial_path, args.root_dir, args.dataset_name),
                _resolve_synthetic_path(ground_path, args.root_dir, args.dataset_name),
                caption,
                sample_weight,
            ))

    return list(dataset) + synthetic_samples, len(synthetic_samples)


def _build_eval_loaders(args, split, transforms, num_workers):
    img_set = ImageDataset(split["image_pids"], split["img_paths"], transforms)
    txt_set = TextDataset(split["caption_pids"], split["captions"], text_length=args.text_length)
    img_loader = DataLoader(img_set, batch_size=args.test_batch_size, shuffle=False, num_workers=num_workers)
    txt_loader = DataLoader(txt_set, batch_size=args.test_batch_size, shuffle=False, num_workers=num_workers)
    return img_loader, txt_loader


def sample_train_dataset_per_pid(dataset, samples_per_id, epoch_seed=None, strategy="random"):
    if samples_per_id <= 0:
        return dataset
    if strategy not in {"random", "cherry_weighted"}:
        raise ValueError(f"Unsupported train sampling strategy: {strategy}")

    pid_to_samples = {}
    for sample in dataset:
        pid_to_samples.setdefault(sample[0], []).append(sample)

    rng = random.Random(epoch_seed) if epoch_seed is not None else random
    sampled_dataset = []
    for pid in sorted(pid_to_samples.keys()):
        pid_samples = pid_to_samples[pid]
        if strategy == "cherry_weighted":
            weights = [
                float(sample[4]) if len(sample) >= 5 and not isinstance(sample[1], int) else 1.0
                for sample in pid_samples
            ]
            sampled_dataset.extend(rng.choices(pid_samples, weights=weights, k=samples_per_id))
        elif len(pid_samples) >= samples_per_id:
            sampled_dataset.extend(rng.sample(pid_samples, samples_per_id))
        else:
            sampled_dataset.extend(rng.choices(pid_samples, k=samples_per_id))
    return sampled_dataset


def relabel_train_dataset_pids(dataset):
    pids = sorted({sample[0] for sample in dataset})
    pid_to_label = {pid: label for label, pid in enumerate(pids)}
    relabeled = []
    for sample in dataset:
        relabeled.append((pid_to_label[sample[0]], *sample[1:]))
    return relabeled, pid_to_label


def build_finetune_train_loader(args, train_dataset, epoch=None):
    train_transforms = build_transforms(img_size=args.img_size, aug=args.img_aug, is_train=True)
    train_dataset = build_caption_cherry_dataset(
        train_dataset,
        mode=args.caption_cherry_mode,
        extra_per_sample=args.caption_cherry_extra_per_sample,
        cherry_weight=args.caption_cherry_weight,
        text_length=args.text_length,
    )
    sampled_dataset = sample_train_dataset_per_pid(
        train_dataset,
        args.train_samples_per_id,
        epoch_seed=epoch,
        strategy=args.train_sample_strategy,
    )
    train_set = ImageTextMLMDataset(sampled_dataset, train_transforms, text_length=args.text_length)
    return DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )


def build_dataloader(args, tranforms=None):
    dataset = __factory[args.dataset_name](root=args.root_dir)
    num_classes = len(dataset.train_id_container)
    num_workers = args.num_workers

    if args.training:
        train_transforms = build_transforms(img_size=args.img_size, aug=args.img_aug, is_train=True)
        train_set = ImageTextMLMDataset(dataset.train, train_transforms, text_length=args.text_length)
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate,
        )

        eval_transforms = build_transforms(img_size=args.img_size, is_train=False)
        split = dataset.val if args.val_dataset == "val" else dataset.test
        val_img_loader, val_txt_loader = _build_eval_loaders(args, split, eval_transforms, num_workers)
        return train_loader, val_img_loader, val_txt_loader, num_classes

    test_transforms = tranforms or build_transforms(img_size=args.img_size, is_train=False)
    test_img_loader, test_txt_loader = _build_eval_loaders(args, dataset.test, test_transforms, num_workers)
    return test_img_loader, test_txt_loader, num_classes


def build_zero_shot_loader(args, finetune=False):
    logger = logging.getLogger("IRRA.dataset")
    dataset_name = args.dataset_name if finetune else args.pretrain
    dataset = __factory[dataset_name](root=args.root_dir)
    train_dataset = dataset.train
    num_workers = args.num_workers

    train_transforms = build_transforms(img_size=args.img_size, aug=args.img_aug, is_train=True)
    eval_transforms = build_transforms(img_size=args.img_size, is_train=False)

    eval_mode = getattr(args, "finetune_eval_mode", "test").lower()
    val_img_loader = None
    val_txt_loader = None

    if finetune and eval_mode == "heldout":
        train_dataset, val_split = split_finetune_train_and_val(
            dataset.train,
            args.finetune_val_ratio,
            args.finetune_val_seed,
        )
        val_img_loader, val_txt_loader = _build_eval_loaders(args, val_split, eval_transforms, num_workers)
        logger.info(
            f"using held-out finetune validation split: train_samples={len(train_dataset)}, "
            f"val_images={len(val_split['img_paths'])}, val_texts={len(val_split['captions'])}"
        )
    elif finetune and eval_mode == "none":
        logger.info("skipping intermediate finetune validation and training on the full train split")
    elif finetune and eval_mode == "test":
        val_img_loader, val_txt_loader = _build_eval_loaders(args, dataset.test, eval_transforms, num_workers)
        logger.info("using official test split for per-epoch finetune evaluation")
    elif finetune:
        raise ValueError(f"Unsupported finetune_eval_mode: {eval_mode}")
    else:
        val_img_loader, val_txt_loader = _build_eval_loaders(args, dataset.test, eval_transforms, num_workers)

    train_dataset, synthetic_count = build_synthetic_cherry_dataset(args, train_dataset)
    if synthetic_count > 0:
        logger.info(
            f"using synthetic image cherry-picking: manifest={args.synthetic_cherry_manifest}, "
            f"samples={synthetic_count}, min_score={args.synthetic_cherry_min_score}, "
            f"max_per_pid={args.synthetic_cherry_max_per_pid}, weight={args.synthetic_cherry_weight}"
        )

    train_dataset, train_pid_to_label = relabel_train_dataset_pids(train_dataset)
    if args.caption_cherry_mode != "off" and args.caption_cherry_extra_per_sample > 0:
        logger.info(
            f"using caption cherry-picking: mode={args.caption_cherry_mode}, "
            f"extra_per_sample={args.caption_cherry_extra_per_sample}, "
            f"weight={args.caption_cherry_weight}"
        )
    logger.info(
        f"using contiguous finetune training labels: classes={len(train_pid_to_label)}, "
        f"raw_pid_min={min(train_pid_to_label)}, raw_pid_max={max(train_pid_to_label)}"
    )

    if getattr(args, "train_samples_per_id", 0) > 0:
        train_loader = build_finetune_train_loader(args, train_dataset, epoch=1)
        logger.info(
            f"using per-id epoch sampling: strategy={args.train_sample_strategy}, "
            f"k={args.train_samples_per_id}, samples={len(train_loader.dataset)}"
        )
    else:
        train_dataset = build_caption_cherry_dataset(
            train_dataset,
            mode=args.caption_cherry_mode,
            extra_per_sample=args.caption_cherry_extra_per_sample,
            cherry_weight=args.caption_cherry_weight,
            text_length=args.text_length,
        )
        train_set = ImageTextMLMDataset(train_dataset, train_transforms, text_length=args.text_length)
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate,
        )
        logger.info("using full training split with random shuffle")

    num_classes = len(train_pid_to_label)
    return train_dataset, train_loader, val_img_loader, val_txt_loader, num_classes


def split_finetune_train_and_val(train_dataset, val_ratio, val_seed):
    pid_to_samples = {}
    for sample in train_dataset:
        pid_to_samples.setdefault(sample[0], []).append(sample)

    all_pids = sorted(pid_to_samples.keys())
    if len(all_pids) < 2:
        raise ValueError("Need at least two training identities to create a held-out finetune validation split.")

    val_ratio = float(val_ratio)
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"finetune_val_ratio must be in (0, 1), but got {val_ratio}")

    rng = random.Random(val_seed)
    shuffled_pids = list(all_pids)
    rng.shuffle(shuffled_pids)
    num_val_pids = max(1, int(round(len(shuffled_pids) * val_ratio)))
    num_val_pids = min(num_val_pids, len(shuffled_pids) - 1)
    val_pids = set(shuffled_pids[:num_val_pids])

    train_split = []
    val_split = {"image_pids": [], "img_paths": [], "caption_pids": [], "captions": []}
    for sample in train_dataset:
        pid, img_path, _, caption = sample[:4]
        if pid in val_pids:
            val_split["image_pids"].append(pid)
            val_split["img_paths"].append(img_path)
            val_split["caption_pids"].append(pid)
            val_split["captions"].append(caption)
        else:
            train_split.append(sample)

    if not train_split or not val_split["img_paths"] or not val_split["captions"]:
        raise ValueError("Failed to create a non-empty finetune train/val split.")
    return train_split, val_split
