import torch

from .lr_scheduler import LRSchedulerWithWarmup


def build_optimizer(args, model):
    params = []
    for key, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lr = args.lr
        if "fta_" in key:
            lr = args.lr2 * args.lr_factor
        params.append({"params": [param], "lr": lr, "weight_decay": args.weight_decay})

    if args.optimizer == "SGD":
        return torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)
    if args.optimizer == "Adam":
        return torch.optim.Adam(params, lr=args.lr, betas=(args.alpha, args.beta), eps=1e-3)
    if args.optimizer == "AdamW":
        return torch.optim.AdamW(params, lr=args.lr, betas=(args.alpha, args.beta), eps=1e-8)
    raise NotImplementedError(f"Unsupported optimizer: {args.optimizer}")


def build_lr_scheduler(args, optimizer):
    return LRSchedulerWithWarmup(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma,
        warmup_factor=args.warmup_factor,
        warmup_epochs=args.warmup_epochs,
        warmup_method=args.warmup_method,
        total_epochs=args.num_epoch,
        mode=args.lrscheduler,
        target_lr=args.target_lr,
        power=args.power,
    )
