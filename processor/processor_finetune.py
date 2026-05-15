import collections
import logging
import random
import time
import torch
from datasets.build import build_filter_loader
from datasets.build import build_finetune_train_loader
from model import objectives
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
import torch.nn.functional as F

def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer, trainset, swanlab_run=None):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("IRRA.train")
    # if get_rank() == 0:
    #     logger.info("Validation before training - Epoch: {}".format(-1))
    #     top1 = evaluator.eval(model.module.eval())
    logger.info('start training')
    if evaluator is None:
        logger.info('intermediate validation is disabled; training will save final.pth only')

    meters = {
        "loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "cda_loss": AverageMeter(),
        "bridge_loss": AverageMeter(),
        "bridge_pair_loss": AverageMeter(),
        "bridge_distill_loss": AverageMeter(),
        "prototype_loss": AverageMeter(),
        "track_loss": AverageMeter(),
        "track_image_loss": AverageMeter(),
        "track_text_loss": AverageMeter(),
        "cvpr_loss": AverageMeter(),
        "fta_loss": AverageMeter(),
        "entropy_loss": AverageMeter(),
        "fa_triplet_loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter()
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_rsum = 0.0
    best_epoch = None

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()

        if getattr(args, "train_samples_per_id", 0) > 0:
            train_loader = build_finetune_train_loader(args, trainset, epoch=epoch)
            logger.info(
                f"Epoch[{epoch}] rebuilt train loader with per-id sampling: strategy={args.train_sample_strategy}, k={args.train_samples_per_id}, samples={len(train_loader.dataset)}"
            )

        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.cuda() for k, v in batch.items()}
           
            ret = model(batch)
            aux_ret = {key: values for key, values in ret.items() if key.startswith('_')}
            ret = {key: values.mean() for key, values in ret.items() if not key.startswith('_')}
            total_loss = sum([v for k, v in ret.items() if "loss" in k])

            batch_size = batch['images'].shape[0]
            
            meters['loss'].update(total_loss.item(), batch_size)
            meters['sdm_loss'].update(ret.get('sdm_loss', 0), batch_size)
            meters['cda_loss'].update(ret.get('cda_loss', 0), batch_size)
            meters['bridge_loss'].update(ret.get('bridge_loss', 0), batch_size)
            meters['bridge_pair_loss'].update(ret.get('bridge_pair_loss', 0), batch_size)
            meters['bridge_distill_loss'].update(ret.get('bridge_distill_loss', 0), batch_size)
            meters['prototype_loss'].update(ret.get('prototype_loss', 0), batch_size)
            meters['track_loss'].update(ret.get('track_loss', 0), batch_size)
            meters['track_image_loss'].update(ret.get('track_image_loss', 0), batch_size)
            meters['track_text_loss'].update(ret.get('track_text_loss', 0), batch_size)
            meters['cvpr_loss'].update(ret.get('cvpr_loss', 0), batch_size)
            meters['fta_loss'].update(ret.get('fta_loss', 0), batch_size)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if '_track_memory_pids' in aux_ret and '_track_memory_image_feats' in aux_ret:
                track_model = model.module if hasattr(model, "module") else model
                track_model.update_track_memory(
                    aux_ret['_track_memory_pids'],
                    aux_ret['_track_memory_image_feats'],
                )
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        # tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)
        if swanlab_run is not None:
            log_payload = {"epoch": epoch, "lr": scheduler.get_lr()[0]}
            for k, v in meters.items():
                if v.avg > 0:
                    log_payload[f"train/{k}"] = float(v.avg)
            swanlab_run.log(log_payload)


        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / 60
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[min] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if evaluator is not None and eval_period > 0 and epoch % eval_period == 0:
            logger.info(f"best RSum: {best_rsum}")
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    eval_metrics = evaluator.eval(model.module.eval(), return_details=True)
                else:
                    eval_metrics = evaluator.eval(model.module.eval(), return_details=True)
                rsum = eval_metrics["t2i_RSum"]
                if swanlab_run is not None:
                    swanlab_run.log({"epoch": epoch, **{f"val/{k}": v for k, v in eval_metrics.items()}})
                torch.cuda.empty_cache()
                if best_rsum < rsum:
                    best_rsum = rsum
                    arguments["epoch"] = epoch
                    best_epoch = epoch
                    checkpointer.save("best0", **arguments)
    arguments["epoch"] = num_epoch
    if get_rank() == 0:
        checkpointer.save("final", **arguments)
        if best_epoch is not None:
            logger.info(f"best RSum: {best_rsum} at epoch {best_epoch}")
        else:
            logger.info("No intermediate validation was run. Saved final checkpoint as final.pth")


def do_inference(model, test_img_loader, test_txt_loader):

    logger = logging.getLogger("IRRA.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())
