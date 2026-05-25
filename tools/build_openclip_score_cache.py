import argparse
import json
import os.path as op
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = op.abspath(op.join(op.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from datasets.build import __factory
from utils.iotools import mkdir_if_missing, read_image
from utils.metrics import rank


class RawImageDataset(Dataset):
    def __init__(self, image_pids, img_paths, transform):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid = self.image_pids[index]
        image = read_image(self.img_paths[index])
        if self.transform is not None:
            image = self.transform(image)
        return pid, image


class RawTextDataset(Dataset):
    def __init__(self, caption_pids, captions):
        self.caption_pids = caption_pids
        self.captions = captions

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        return self.caption_pids[index], self.captions[index]


def text_collate(batch):
    pids, captions = zip(*batch)
    return torch.tensor(pids), list(captions)


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


@torch.no_grad()
def compute_openclip_similarity(model, tokenizer, image_loader, text_loader, device):
    model.eval()
    qids, gids, qfeats, gfeats = [], [], [], []

    for pid, captions in text_loader:
        tokens = tokenizer(captions).to(device)
        text_feat = model.encode_text(tokens)
        qids.append(pid.view(-1).cpu())
        qfeats.append(text_feat.float().cpu())

    for pid, images in image_loader:
        images = images.to(device)
        image_feat = model.encode_image(images)
        gids.append(pid.view(-1).cpu())
        gfeats.append(image_feat.float().cpu())

    qids = torch.cat(qids, 0)
    gids = torch.cat(gids, 0)
    qfeats = F.normalize(torch.cat(qfeats, 0).to(device), dim=1)
    gfeats = F.normalize(torch.cat(gfeats, 0).to(device), dim=1)
    similarity = (qfeats @ gfeats.t()).float().cpu()
    return similarity, qids, gids


def build_args():
    parser = argparse.ArgumentParser(description="Build a score cache from an OpenCLIP/HF model.")
    parser.add_argument("--model_name", default="hf-hub:joaodaniel/RS-M-CLIP")
    parser.add_argument("--pretrained", default="", help="Optional OpenCLIP pretrained tag.")
    parser.add_argument("--dataset_name", default="AERI-PEDES")
    parser.add_argument("--root_dir", default="/home/wuyong/datasets")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True, help="Output .pth score cache path.")
    parser.add_argument("--report", default="", help="Optional JSON report path.")
    return parser.parse_args()


def main():
    args = build_args()
    try:
        import open_clip
    except ImportError as exc:
        raise RuntimeError("Install OpenCLIP first: `pip install -U open_clip_torch`.") from exc

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    pretrained = args.pretrained or None
    model, _, preprocess = open_clip.create_model_and_transforms(args.model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model = model.to(device)

    dataset = __factory[args.dataset_name](root=args.root_dir)
    split = dataset.test if args.split == "test" else dataset.val
    image_set = RawImageDataset(split["image_pids"], split["img_paths"], preprocess)
    text_set = RawTextDataset(split["caption_pids"], split["captions"])
    image_loader = DataLoader(
        image_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    text_loader = DataLoader(
        text_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=text_collate,
    )

    similarity, qids, gids = compute_openclip_similarity(model, tokenizer, image_loader, text_loader, device)
    metrics = evaluate_similarity(similarity, qids, gids)
    metadata = {
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "dataset_name": args.dataset_name,
        "root_dir": args.root_dir,
        "split": args.split,
        "metrics": metrics,
    }

    mkdir_if_missing(op.dirname(args.output))
    torch.save(
        {
            "similarity": similarity.cpu(),
            "qids": qids.cpu(),
            "gids": gids.cpu(),
            "metadata": metadata,
        },
        args.output,
    )

    report_path = args.report or op.splitext(args.output)[0] + ".json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(json.dumps({"score_cache": args.output, "report": report_path, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
