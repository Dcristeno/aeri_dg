import collections
import os
import os.path as op
from model.build_finetune import build_finetune_model
import torch
import numpy as np
import random
import time
import torch.nn as nn

from datasets import build_dataloader
from datasets.bases import ImageTextMLMDataset
from datasets.build import  build_zero_shot_loader
from processor.processor_finetune import do_train
from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from solver import build_optimizer, build_lr_scheduler
from model import build_model
from utils.metrics import Evaluator
from utils.options import get_args
from utils.comm import get_rank, synchronize

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
    }
    finetune_alias_seen = any(token in {"fa", "fta", "cda", "bridge", "g2a", "ga", "ga_bridge"} for token in tokens)
    for token in tokens:
        token = alias_map.get(token, token)
        if finetune_alias_seen and token == "sdm":
            token = "cda"
        if token not in normalized:
            normalized.append(token)
    return "+".join(normalized)


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    args = get_args()
    args.loss_names = normalize_finetune_loss_names(args.loss_names)
    set_seed(args.seed + get_rank())
    name = args.name

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    
    device = "cuda"
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.output_dir = op.join(args.output_dir, args.dataset_name, f'{cur_time}_{name}')
    logger = setup_logger('IRRA', save_dir=args.output_dir, if_train=args.training, distributed_rank=get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(str(args).replace(',', '\n'))
    save_train_configs(args.output_dir, args)

    # get image-text pair datasets dataloader
    trainset ,train_loader, val_img_loader, val_txt_loader, num_classes = build_zero_shot_loader(args,finetune=True)
    args.track_memory_num_pids = max((sample[0] for sample in trainset), default=-1) + 1
    model = build_finetune_model(args, num_classes)
    logger.info('Total params: %2.fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    if args.finetune:
        logger.info("loading {} model".format(args.finetune))
        param_dict = torch.load(args.finetune,map_location='cpu')['model']
        for k in list(param_dict.keys()):
            refine_k = k.replace('module.','')
            param_dict[refine_k] = param_dict[k].detach().clone()
            del param_dict[k]
        model.load_state_dict(param_dict, False)
    model = model.float()

    model.cuda()
    model = nn.DataParallel(model)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)

    is_master = get_rank() == 0
    checkpointer = Checkpointer(model, optimizer, scheduler, args.output_dir, is_master)
    evaluator = Evaluator(val_img_loader, val_txt_loader) if val_img_loader is not None and val_txt_loader is not None else None
    swanlab_run = None
    if args.use_swanlab and is_master:
        if swanlab is None:
            raise ImportError("SwanLab is not installed. Please run `pip install swanlab` before using --use_swanlab.")
        swanlab_run = swanlab.init(
            project=args.swanlab_project,
            experiment_name=args.swanlab_experiment or name,
            config=vars(args),
            mode=args.swanlab_mode,
            logdir=args.output_dir,
        )

    start_epoch = 1
    if args.resume:
        checkpoint = checkpointer.resume(args.resume_ckpt_file)
        start_epoch = checkpoint['epoch']


    do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer, trainset, swanlab_run=swanlab_run)
