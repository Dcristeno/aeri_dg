import argparse
import os.path as op

import torch

from datasets import build_dataloader
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
from model.build_finetune import build_finetune_model
from utils.iotools import load_train_configs

try:
    import swanlab
except ImportError:
    swanlab = None


def normalize_finetune_loss_names(loss_names):
    tokens = [token.strip() for token in loss_names.split('+') if token.strip()]
    normalized = []
    alias_map = {
        "fa": "fta",
        "g2a": "bridge",
        "ga": "bridge",
        "ga_bridge": "bridge",
        "bridge_loss": "bridge",
        "hn": "hardneg",
        "hard": "hardneg",
        "hard_negative": "hardneg",
        "hard-negative": "hardneg",
        "hardneg_loss": "hardneg",
    }
    finetune_alias_seen = any(
        token in {"fa", "fta", "cda", "bridge", "g2a", "ga", "ga_bridge", "hn", "hard", "hard_negative", "hard-negative", "hardneg"}
        for token in tokens
    )
    for token in tokens:
        token = alias_map.get(token, token)
        if finetune_alias_seen and token == "sdm":
            token = "cda"
        if token not in normalized:
            normalized.append(token)
    return "+".join(normalized)


def checkpoint_uses_finetune_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    return any(key.endswith("query") or key.endswith("mlp_logsigma2.0.weight") for key in state_dict.keys())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CFAN Test")
    parser.add_argument("--config_file", required=True, help="Path to saved configs.yaml")
    parser.add_argument("--checkpoint", default="", help="Optional explicit checkpoint path")
    parser.add_argument("--root_dir", default="", help="Override dataset root_dir from config")
    parser.add_argument("--output_dir", default="", help="Override output_dir from config for local logs")
    parser.add_argument("--loss_names", default="", help="Override loss names from config")
    parser.add_argument("--use_finetune_model", action="store_true", help="Force build_finetune_model for evaluation")
    parser.add_argument("--use_swanlab", action="store_true", help="Log test metrics to SwanLab")
    parser.add_argument("--swanlab_project", default="CFAN", help="SwanLab project name")
    parser.add_argument("--swanlab_experiment", default="", help="Optional SwanLab experiment name")
    parser.add_argument("--swanlab_mode", default="cloud", help="SwanLab mode, e.g. cloud or local")
    args_cli = parser.parse_args()

    args = load_train_configs(args_cli.config_file)
    args.training = False

    if args_cli.root_dir:
        args.root_dir = args_cli.root_dir
    if args_cli.output_dir:
        args.output_dir = args_cli.output_dir
    if args_cli.loss_names:
        args.loss_names = args_cli.loss_names

    if args_cli.checkpoint:
        checkpoint_path = args_cli.checkpoint
    else:
        best_ckpt = op.join(args.output_dir, 'best0.pth')
        final_ckpt = op.join(args.output_dir, 'final.pth')
        checkpoint_path = best_ckpt if op.exists(best_ckpt) else final_ckpt
    args.loss_names = normalize_finetune_loss_names(args.loss_names)

    logger = setup_logger('IRRA', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    use_finetune_model = args_cli.use_finetune_model or checkpoint_uses_finetune_model(checkpoint_path)
    if use_finetune_model:
        logger.info("Building finetune CFAN model for evaluation")
        model = build_finetune_model(args, num_classes=num_classes)
    else:
        logger.info("Building baseline IRRA/HAM model for evaluation")
        model = build_model(args, num_classes=num_classes)

    checkpointer = Checkpointer(model)
    checkpointer.load(f=checkpoint_path)
    model.to("cuda")

    swanlab_run = None
    if args_cli.use_swanlab:
        if swanlab is None:
            raise ImportError("SwanLab is not installed. Please run `pip install swanlab` before using --use_swanlab.")
        swanlab_run = swanlab.init(
            project=args_cli.swanlab_project,
            experiment_name=args_cli.swanlab_experiment or f"{args.name}_eval",
            config={
                "config_file": args_cli.config_file,
                "checkpoint": checkpoint_path,
                "dataset_name": args.dataset_name,
                "root_dir": args.root_dir,
                "loss_names": args.loss_names,
            },
            mode=args_cli.swanlab_mode,
            logdir=args.output_dir,
        )

    do_inference(model, test_img_loader, test_txt_loader, swanlab_run=swanlab_run)
