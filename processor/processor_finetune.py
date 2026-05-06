import logging
import time

import torch
from datasets.build import build_finetune_train_loader
from torch.utils.tensorboard import SummaryWriter

from utils.comm import get_rank
from utils.meter import AverageMeter


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer, trainset, swanlab_run=None):
    log_period = args.log_period
    eval_period = args.eval_period
    num_epoch = args.num_epoch
    arguments = {"num_epoch": num_epoch, "iteration": 0}

    logger = logging.getLogger("IRRA.train")
    logger.info("start training")
    if evaluator is None:
        logger.info("intermediate validation is disabled; training will save final.pth only")

    meters = {
        "loss": AverageMeter(),
        "base_loss": AverageMeter(),
        "base_aerial_text": AverageMeter(),
        "base_ground_text": AverageMeter(),
        "id_loss": AverageMeter(),
        "rep_reg_loss": AverageMeter(),
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)
    best_r1 = 0.0
    best_epoch = None

    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()

        if getattr(args, "train_samples_per_id", 0) > 0:
            train_loader = build_finetune_train_loader(args, trainset, epoch=epoch)
            logger.info(
                f"Epoch[{epoch}] rebuilt train loader with per-id sampling: "
                f"strategy={args.train_sample_strategy}, k={args.train_samples_per_id}, "
                f"samples={len(train_loader.dataset)}"
            )

        for n_iter, batch in enumerate(train_loader):
            batch = {key: value.cuda() for key, value in batch.items()}

            ret = model(batch)
            ret = {key: value.mean() for key, value in ret.items()}
            total_loss = sum(value for key, value in ret.items() if key.endswith("_loss"))
            batch_size = batch["images"].shape[0]

            meters["loss"].update(total_loss.item(), batch_size)
            meters["base_loss"].update(ret["base_loss"], batch_size)
            meters["base_aerial_text"].update(ret["base_aerial_text"], batch_size)
            meters["base_ground_text"].update(ret["base_ground_text"], batch_size)
            meters["id_loss"].update(ret.get("id_loss", 0), batch_size)
            meters["rep_reg_loss"].update(ret.get("rep_reg_loss", 0), batch_size)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                for key, meter in meters.items():
                    if meter.avg > 0:
                        info_str += f", {key}: {meter.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)

        tb_writer.add_scalar("lr", scheduler.get_lr()[0], epoch)
        for key, meter in meters.items():
            if meter.avg > 0:
                tb_writer.add_scalar(key, meter.avg, epoch)
        if swanlab_run is not None:
            payload = {"epoch": epoch, "lr": scheduler.get_lr()[0]}
            for key, meter in meters.items():
                if meter.avg > 0:
                    payload[f"train/{key}"] = float(meter.avg)
            swanlab_run.log(payload)

        scheduler.step()

        if get_rank() == 0:
            end_time = time.time()
            time_per_epoch = (end_time - start_time) / 60
            logger.info(
                "Epoch {} done. Time per epoch: {:.3f}[min] Speed: {:.1f}[samples/s]".format(
                    epoch,
                    time_per_epoch,
                    train_loader.batch_size / max(time_per_epoch, 1e-12),
                )
            )

        if evaluator is not None and eval_period > 0 and epoch % eval_period == 0:
            logger.info(f"best R1: {best_r1}")
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                eval_model = model.module if hasattr(model, "module") else model
                eval_metrics = evaluator.eval(eval_model.eval(), return_details=True)
                r1 = eval_metrics["t2i_R1"]
                if swanlab_run is not None:
                    swanlab_run.log({"epoch": epoch, **{f"val/{key}": value for key, value in eval_metrics.items()}})
                torch.cuda.empty_cache()
                if best_r1 < r1:
                    best_r1 = r1
                    best_epoch = epoch
                    arguments["epoch"] = epoch
                    checkpointer.save("best0", **arguments)

    arguments["epoch"] = num_epoch
    if get_rank() == 0:
        checkpointer.save("final", **arguments)
        if best_epoch is not None:
            logger.info(f"best R1: {best_r1} at epoch {best_epoch}")
        else:
            logger.info("No intermediate validation was run. Saved final checkpoint as final.pth")
