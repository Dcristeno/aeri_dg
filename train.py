import collections
import os
import os.path as op
import torch
import numpy as np
import random
import time
import torch.nn as nn

from datasets import build_dataloader
from datasets.bases import ImageTextMLMDataset
from datasets.build import build_zero_shot_loader
from processor.processor import do_pretrain
from processor.processor_finetune import do_train
from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from solver import build_optimizer, build_lr_scheduler
from model import build_model,build_finetune_model
from utils.metrics import Evaluator
from utils.options import get_args
from utils.comm import get_rank, synchronize

try:
    import swanlab
except ImportError:
    swanlab = None


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
    loader_outputs = build_zero_shot_loader(args)
    if len(loader_outputs) == 5:
        trainset, train_loader, val_img_loader0, val_txt_loader0, num_classes = loader_outputs
        val_img_loader1 = val_txt_loader1 = None
        val_img_loader2 = val_txt_loader2 = None
    else:
        trainset, train_loader, val_img_loader0, val_txt_loader0, val_img_loader1, val_txt_loader1, val_img_loader2, val_txt_loader2, num_classes = loader_outputs
    if args.nam:
        model = build_model(args, num_classes)
    else:
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
    # model = model.float()
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
    evaluator0 = Evaluator(val_img_loader0, val_txt_loader0) if val_img_loader0 is not None and val_txt_loader0 is not None else None
    evaluator1 = Evaluator(val_img_loader1, val_txt_loader1) if val_img_loader1 is not None and val_txt_loader1 is not None else None
    evaluator2 = Evaluator(val_img_loader2, val_txt_loader2) if val_img_loader2 is not None and val_txt_loader2 is not None else None
    swanlab_run = None
    if args.use_swanlab and get_rank() == 0:
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

    if args.nam:
        do_pretrain(start_epoch, args, model, train_loader, evaluator0,evaluator1,evaluator2, optimizer, scheduler, checkpointer, trainset, swanlab_run=swanlab_run)
    else:
        do_train(start_epoch, args, model, train_loader, evaluator0,evaluator1,evaluator2, optimizer, scheduler, checkpointer, trainset)
