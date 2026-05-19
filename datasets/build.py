import logging
import random
import torch
import torchvision.transforms as T
import numpy as np
from torch.utils.data import DataLoader
from datasets.luperson import LuPerson_PEDES
from datasets.sampler import RandomIdentitySampler
from datasets.sampler_ddp import RandomIdentitySampler_DDP
from torch.utils.data.distributed import DistributedSampler

from utils.comm import get_world_size
from utils.iotools import read_image

from .bases import FilterDataset, ImageDataset, TextDataset, ImageTextDataset, ImageTextMLMDataset

from .cuhkpedes import CUHKPEDES
from .icfgpedes import ICFGPEDES
from .rstpreid import RSTPReid
from .aeripedes import AERIPEDES

from .agtbpr import AG_ReID
from .AGData import AGData,AGSGData,AGDataAttr

# __factory = {'CUHK-PEDES': CUHKPEDES, 'ICFG-PEDES': ICFGPEDES, 'RSTPReid': RSTPReid,
#             'LuPerson_PEDES':LuPerson_PEDES,}

# __factory = {'AERI-PEDES': AERIPEDES}

__factory = {
    'CUHK-PEDES': CUHKPEDES, 
    'ICFG-PEDES': ICFGPEDES, 
    'RSTPReid': RSTPReid, 
    "AGTBPR":AG_ReID,
    'AERI-PEDES': AERIPEDES,
    "AGData":AGData,
    "AGDataAttr":AGDataAttr, 
    "AGSGData":AGSGData
    }

def build_transforms(img_size=(384, 128), aug=False, is_train=True):
    height, width = img_size

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    if not is_train:
        transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return transform

    # transform for training
    if aug:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(scale=(0.02, 0.4), value=mean),
        ])
    else:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    return transform


def collate(batch):
    keys = set([key for b in batch for key in b.keys()])
    # turn list of dicts data structure to dict of lists data structure
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

    batch_tensor_dict = {}
    for k, v in dict_batch.items():
        if isinstance(v[0], int):
            batch_tensor_dict.update({k: torch.tensor(v)})
        elif torch.is_tensor(v[0]):
             batch_tensor_dict.update({k: torch.stack(v)})
        else:
            raise TypeError(f"Unexpect data type: {type(v[0])} in a batch.")

    return batch_tensor_dict


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


def _mlm_dataset_kwargs(args):
    return {
        "pclip_noise_ratio": getattr(args, "pclip_noise_ratio", 0.0),
        "vocab_size": getattr(args, "vocab_size", 49408),
    }


def _uses_ground_line(args):
    loss_names = getattr(args, "loss_names", "")
    tasks = {token.strip() for token in loss_names.split("+") if token.strip()}
    return bool(tasks & {"cda", "bridge", "ga_bridge"})


def build_finetune_train_loader(args, train_dataset, epoch=None):
    train_transforms = build_transforms(img_size=args.img_size,
                                        aug=args.img_aug,
                                        is_train=True)
    sampled_dataset = sample_train_dataset_per_pid(
        train_dataset,
        args.train_samples_per_id,
        epoch_seed=epoch,
        strategy=args.train_sample_strategy,
        mid_ratio=args.train_sample_mid_ratio,
        representative_weight=args.train_sample_representative_weight,
        diversity_weight=args.train_sample_diversity_weight,
        mid_weight=args.train_sample_mid_weight,
    )
    train_set = ImageTextMLMDataset(sampled_dataset,
                                    train_transforms,
                                    text_length=args.text_length,
                                    tile_mix_grid=args.train_tile_mix_grid,
                                    tile_mix_prob=args.train_tile_mix_prob,
                                    use_ground=_uses_ground_line(args),
                                    **_mlm_dataset_kwargs(args))
    return DataLoader(train_set,
                      batch_size=args.batch_size,
                      shuffle=True,
                      num_workers=args.num_workers,
                      collate_fn=collate)

def _uses_id_loss(args):
    return "id" in [token.strip() for token in getattr(args, "loss_names", "").split("+")]


def _set_id_class_mapping(args, train_dataset):
    pid_classes = sorted({int(sample[0]) for sample in train_dataset})
    args.id_pid_classes = []
    return max(pid_classes, default=-1) + 1


def build_dataloader(args, tranforms=None):
    logger = logging.getLogger("IRRA.dataset")

    num_workers = args.num_workers
    dataset = __factory[args.dataset_name](root=args.root_dir)
    num_classes = len(dataset.train_id_container)
    if _uses_id_loss(args):
        num_classes = _set_id_class_mapping(args, dataset.train)
    
    if args.training:
        train_transforms = build_transforms(img_size=args.img_size,
                                            aug=args.img_aug,
                                            is_train=True)
        val_transforms = build_transforms(img_size=args.img_size,
                                          is_train=False)

        if args.MLM:
            if args.pretrain:
                syn_dataset = __factory[args.pretrain](root=args.root_dir)
                train_set = ImageTextMLMDataset(syn_dataset.train,
                                        train_transforms,
                                        text_length=args.text_length,
                                        use_ground=_uses_ground_line(args),
                                        **_mlm_dataset_kwargs(args))
                num_classes = len(syn_dataset.train)
            else:
                train_set = ImageTextMLMDataset(dataset.train,
                                        train_transforms,
                                        text_length=args.text_length,
                                        use_ground=_uses_ground_line(args),
                                        **_mlm_dataset_kwargs(args))
        else:
            train_set = ImageTextDataset(dataset.train,
                                     train_transforms,
                                     text_length=args.text_length)

        if args.sampler == 'identity':
            if args.distributed:
                logger.info('using ddp random identity sampler')
                logger.info('DISTRIBUTED TRAIN START')
                mini_batch_size = args.batch_size // get_world_size()
                # TODO wait to fix bugs
                data_sampler = RandomIdentitySampler_DDP(
                    dataset.train, args.batch_size, args.num_instance)
                batch_sampler = torch.utils.data.sampler.BatchSampler(
                    data_sampler, mini_batch_size, True)

            else:
                logger.info(
                    f'using random identity sampler: batch_size: {args.batch_size}, id: {args.batch_size // args.num_instance}, instance: {args.num_instance}'
                )
                train_loader = DataLoader(train_set,
                                          batch_size=args.batch_size,
                                          sampler=RandomIdentitySampler(
                                              dataset.train, args.batch_size,
                                              args.num_instance),
                                          num_workers=num_workers,
                                          collate_fn=collate)
        elif args.sampler == 'random':
            # TODO add distributed condition
            logger.info('using random sampler')
            train_loader = DataLoader(train_set,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      collate_fn=collate)
        else:
            logger.error('unsupported sampler! expected softmax or triplet but got {}'.format(args.sampler))

        # use test set as validate set
        ds = dataset.val if args.val_dataset == 'val' else dataset.test
        val_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                   val_transforms)
        val_txt_set = TextDataset(ds['caption_pids'],
                                  ds['captions'],
                                  text_length=args.text_length)

        val_img_loader = DataLoader(val_img_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)
        val_txt_loader = DataLoader(val_txt_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)

        return train_loader, val_img_loader, val_txt_loader, num_classes
    else:
        # build dataloader for testing
        if tranforms:
            test_transforms = tranforms
        else:
            test_transforms = build_transforms(img_size=args.img_size,
                                               is_train=False)

        ds = dataset.test
        test_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                    test_transforms)
        test_txt_set = TextDataset(ds['caption_pids'],
                                   ds['captions'],
                                   text_length=args.text_length)

        test_img_loader = DataLoader(test_img_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)
        test_txt_loader = DataLoader(test_txt_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)
        return test_img_loader, test_txt_loader, num_classes


def build_zero_shot_loader(args, finetune=False):
    logger = logging.getLogger("IRRA.dataset")

    num_workers = args.num_workers

    train_transforms = build_transforms(img_size=args.img_size,
                                            aug=args.img_aug,
                                            is_train=True)
    val_transforms = build_transforms(img_size=args.img_size,
                                          is_train=False)
    
    if finetune:
        syn_dataset = __factory[args.dataset_name](root=args.root_dir)
    else:
        syn_dataset = __factory[args.pretrain](root=args.root_dir)

    if finetune:
        eval_mode = getattr(args, "finetune_eval_mode", "test").lower()
        train_dataset = syn_dataset.train
        train_set = ImageTextMLMDataset(train_dataset,
                                train_transforms,
                                text_length=args.text_length,
                                use_ground=_uses_ground_line(args),
                                **_mlm_dataset_kwargs(args))
        num_classes = len(syn_dataset.train)

        if eval_mode == "heldout":
            train_dataset, val_dataset = split_finetune_train_and_val(
                syn_dataset.train,
                getattr(args, "finetune_val_ratio", 0.1),
                getattr(args, "finetune_val_seed", 1),
            )
            val_img_set = ImageDataset(val_dataset['image_pids'], val_dataset['img_paths'],
                                        val_transforms)
            val_txt_set = TextDataset(val_dataset['caption_pids'],
                                        val_dataset['captions'],
                                        text_length=args.text_length)
            train_set = ImageTextMLMDataset(train_dataset,
                                    train_transforms,
                                    text_length=args.text_length,
                                    use_ground=_uses_ground_line(args),
                                    **_mlm_dataset_kwargs(args))
            num_classes = max((sample[0] for sample in train_dataset), default=-1) + 1
            logger.info(
                f'using held-out finetune validation split: train_samples={len(train_dataset)}, '
                f'val_images={len(val_dataset["img_paths"])}, val_texts={len(val_dataset["captions"])}, '
                f'val_ratio={getattr(args, "finetune_val_ratio", 0.1)}, val_seed={getattr(args, "finetune_val_seed", 1)}'
            )
        elif eval_mode == "none":
            val_img_loader = None
            val_txt_loader = None
            logger.info('skipping intermediate finetune validation and training on the full train split')
        elif eval_mode == "test":
            ds = syn_dataset.test
            val_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                        val_transforms)
            val_txt_set = TextDataset(ds['caption_pids'],
                                        ds['captions'],
                                        text_length=args.text_length)
            logger.info('using official test split for per-epoch finetune evaluation')
        else:
            raise ValueError(f"Unsupported finetune_eval_mode: {eval_mode}")
    else:
        ds = syn_dataset.test
        val_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                    val_transforms)
        val_txt_set = TextDataset(ds['caption_pids'],
                                    ds['captions'],
                                    text_length=args.text_length)
        train_set = ImageTextMLMDataset(syn_dataset.train,
                                train_transforms,
                                text_length=args.text_length,
                                use_ground=_uses_ground_line(args),
                                **_mlm_dataset_kwargs(args))
        train_dataset = syn_dataset.train
        num_classes = len(syn_dataset.train)

    if _uses_id_loss(args):
        num_classes = _set_id_class_mapping(args, train_dataset)

    if not (finetune and getattr(args, "finetune_eval_mode", "test").lower() == "none"):
        val_img_loader = DataLoader(val_img_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)
        val_txt_loader = DataLoader(val_txt_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)

    logger.info('using random sampler')
    if getattr(args, "train_samples_per_id", 0) > 0:
        logger.info(
            f'using per-id epoch sampling: strategy={args.train_sample_strategy}, {args.train_samples_per_id} samples per id before shuffle'
        )
        train_loader = build_finetune_train_loader(args, train_dataset, epoch=1)
    else:
        train_loader = DataLoader(train_set,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=num_workers,
                                    )

    return train_dataset, train_loader, val_img_loader, val_txt_loader, num_classes


def split_finetune_train_and_val(train_dataset, val_ratio, val_seed):
    pid_to_samples = {}
    for sample in train_dataset:
        pid = sample[0]
        pid_to_samples.setdefault(pid, []).append(sample)

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
    val_img_paths = []
    val_img_pids = []
    val_captions = []
    val_caption_pids = []

    for sample in train_dataset:
        pid, img_path, _, caption = sample[:4]
        if pid in val_pids:
            val_img_paths.append(img_path)
            val_img_pids.append(pid)
            val_captions.append(caption)
            val_caption_pids.append(pid)
        else:
            train_split.append(sample)

    if not train_split or not val_img_paths or not val_captions:
        raise ValueError("Failed to create a non-empty finetune train/val split.")

    val_split = {
        "image_pids": val_img_pids,
        "img_paths": val_img_paths,
        "caption_pids": val_caption_pids,
        "captions": val_captions,
    }
    return train_split, val_split

def build_filter_loader(args, dataset):
    logger = logging.getLogger("IRRA.dataset")

    num_workers = args.num_workers

    train_transforms = build_transforms(img_size=args.img_size,
                                            aug=args.img_aug,
                                            is_train=True)
    train_set = FilterDataset(dataset,
                            train_transforms,
                            text_length=args.text_length,
                            use_ground=_uses_ground_line(args))
    train_loader = DataLoader(train_set,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=num_workers)

    return train_loader
