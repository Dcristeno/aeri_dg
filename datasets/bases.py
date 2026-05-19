from typing import List
import numpy as np
from torch.utils.data import Dataset
import os.path as osp
import logging
import torch
from collections import defaultdict
from PIL import Image
from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer
from prettytable import PrettyTable
import random
import regex as re
import copy


class BaseDataset(object):
    """
    Base class of text to image reid dataset
    """
    logger = logging.getLogger("IRRA.dataset")

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = len(
            self.test_id_container), len(self.test_annos), len(
                self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = len(
            self.val_id_container), len(self.val_annos), len(
                self.val['captions'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info('\n' + str(table))


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result


class ImageTextDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        ret = {
            'img_path': img_path,
            'caption': caption,
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': tokens,
        }

        return ret


class ImageDataset(Dataset):
    def __init__(self, image_pids, img_paths, transform=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_path = self.image_pids[index], self.img_paths[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return pid, img


class TextDataset(Dataset):
    def __init__(self,
                 caption_pids,
                 captions,
                 text_length: int = 77,
                 truncate: bool = True):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption = self.caption_pids[index], self.captions[index]

        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        return pid, caption


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class ImageTextMLMDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True,
                 tile_mix_grid: int = 0,
                 tile_mix_prob: float = 0.0,
                 pclip_noise_ratio: float = 0.0,
                 vocab_size: int = 49408,
                 use_ground: bool = False):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tile_mix_grid = int(tile_mix_grid)
        self.tile_mix_prob = float(tile_mix_prob)
        self.pclip_noise_ratio = float(pclip_noise_ratio)
        self.vocab_size = int(vocab_size)
        self.use_ground = bool(use_ground)

        self.tokenizer = SimpleTokenizer()
        self.pid_to_indices = defaultdict(list)
        for idx, sample in enumerate(self.dataset):
            self.pid_to_indices[sample[0]].append(idx)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, g_path, caption = self._parse_sample(self.dataset[index], index)
        img = read_image(img_path)
        if self.tile_mix_grid > 1 and random.random() < self.tile_mix_prob:
            img = self._build_same_pid_tile_mix(index, pid, img)
        g = read_image(g_path) if self.use_ground and g_path is not None else None
        if self.transform is not None:
            img = self.transform(img)
            if g is not None:
                g = self.transform(g)
        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        caption_np = caption_tokens.cpu().numpy().copy()
        mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(caption_np.copy())
        pclip_noisy_tokens = self._build_random_replaced_tokens(caption_np.copy())
        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': caption_tokens,
            'mlm_ids': mlm_tokens,
            'mlm_labels': mlm_labels,
            'pclip_noisy_caption_ids': pclip_noisy_tokens,
        }
        if g is not None:
            ret['ground_imgs'] = g

        return ret

    def _parse_sample(self, sample, index):
        if len(sample) >= 4 and isinstance(sample[1], str) and isinstance(sample[2], str):
            pid, img_path, g_path, caption = sample[:4]
            return pid, index, img_path, g_path, caption

        pid, image_id, img_path, caption = sample[:4]
        return pid, image_id, img_path, None, caption

    def _build_same_pid_tile_mix(self, index, pid, base_img):
        candidate_indices = self.pid_to_indices.get(pid, [])
        if not candidate_indices:
            return base_img

        partner_index = random.choice(candidate_indices)
        if len(candidate_indices) > 1:
            while partner_index == index:
                partner_index = random.choice(candidate_indices)

        _, _, partner_img_path, _, _ = self._parse_sample(self.dataset[partner_index], partner_index)
        partner_img = read_image(partner_img_path)
        if partner_img.size != base_img.size:
            partner_img = partner_img.resize(base_img.size, Image.BILINEAR)

        width, height = base_img.size
        grid = self.tile_mix_grid
        mixed = Image.new('RGB', (width, height))
        x_points = [round(i * width / grid) for i in range(grid + 1)]
        y_points = [round(i * height / grid) for i in range(grid + 1)]

        for gy in range(grid):
            top = y_points[gy]
            bottom = y_points[gy + 1]
            for gx in range(grid):
                left = x_points[gx]
                right = x_points[gx + 1]
                box = (left, top, right, bottom)
                source_img = base_img if random.random() < 0.5 else partner_img
                mixed.paste(source_img.crop(box), box)

        return mixed

    def _build_random_masked_tokens_and_labels(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3))  # 1 ~ 49405

        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = mask

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)

                    # append current token to output (we will predict these later)
                    labels.append(token)
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)

        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)


    def _build_random_replaced_tokens(self, tokens):
        """
        P-CLIP lite text perturbation: replace a small portion of normal words
        with random vocabulary tokens while preserving SOT/EOT/padding.
        """
        if self.pclip_noise_ratio <= 0:
            return torch.tensor(tokens)

        max_random_token = min(self.vocab_size, len(self.tokenizer.encoder) - 3, 49405)
        token_range = list(range(1, max_random_token))
        candidate_positions = [idx for idx, token in enumerate(tokens) if 0 < token < 49405]
        if not candidate_positions or not token_range:
            return torch.tensor(tokens)

        replaced = False
        for idx in candidate_positions:
            if random.random() < self.pclip_noise_ratio:
                tokens[idx] = random.choice(token_range)
                replaced = True

        if not replaced:
            idx = random.choice(candidate_positions)
            tokens[idx] = random.choice(token_range)

        return torch.tensor(tokens)


class FilterDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True,
                 use_ground: bool = False):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.use_ground = bool(use_ground)

        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, ground_path, caption, sim = self._parse_sample(self.dataset[index], index)
        img = read_image(img_path)
        ground_img = read_image(ground_path) if self.use_ground and ground_path is not None else None
        if self.transform is not None:
            img = self.transform(img)
            if ground_img is not None:
                ground_img = self.transform(ground_img)

        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(caption_tokens.cpu().numpy(), sim)
        ori_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': caption_tokens,
            'mlm_ids': mlm_tokens,
            'mlm_labels': mlm_labels,
            'caption_ids_ori': ori_tokens
        }
        if ground_img is not None:
            ret['ground_imgs'] = ground_img

        return ret

    def _parse_sample(self, sample, index):
        sim = np.full(self.text_length, 0.85, dtype=np.float32)
        sample_core = sample
        if len(sample) >= 5 and not isinstance(sample[-1], str):
            sample_core = sample[:-1]
            sim = np.asarray(sample[-1], dtype=np.float32)

        if len(sample_core) >= 4 and isinstance(sample_core[1], str) and isinstance(sample_core[2], str):
            pid, img_path, ground_path, caption = sample_core[:4]
            return pid, index, img_path, ground_path, caption, sim

        pid, image_id, img_path, caption = sample_core[:4]
        return pid, image_id, img_path, None, caption, sim

    def _build_random_masked_tokens_and_labels(self, tokens, sim):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3))  # 1 ~ 49405

        labels = []

        if tokens[-1] == 0:
            valid_token_num = np.where(tokens == 0)[0][0]
        else:
            valid_token_num = len(tokens)
        ori_sim = np.array(sim)
        ori_pro = 1 - ori_sim
        if ori_pro[-1] != 0.15:
            valid_prob = ori_pro[1:valid_token_num-1]
            # normalize the probisibility to match E = 0.15
            mean_prob = np.mean(valid_prob)
            normed_prob = valid_prob - mean_prob
            normalized_prob = normed_prob + 0.15
            normalized_prob = np.clip(normalized_prob, 0, 1)
            ori_pro[1:valid_token_num-1] = normalized_prob

        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability
                if prob < ori_pro[i]:
                    prob /= ori_pro[i]

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = mask

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)

                    # append current token to output (we will predict these later)
                    labels.append(token)
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)

        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)
