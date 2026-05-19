import collections
import logging
import random
import time
import torch
from datasets.build import build_filter_loader
from model import objectives
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
import torch.nn.functional as F

def do_pretrain(start_epoch, args, model, train_loader, evaluator0,evaluator1,evaluator2, optimizer,
             scheduler, checkpointer, trainset, swanlab_run=None):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("IRRA.train")
    if get_rank() == 0:
        logger.info("Validation before training - Epoch: {}".format(-1))
        # top1 = evaluator0.eval(model.module.eval())
        # top1 = evaluator1.eval(model.module.eval())
        # top1 = evaluator2.eval(model.module.eval())
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "cda_loss": AverageMeter(),
        "fta_loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter()
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_r1_0 = 0.0
    best_r1_1 = 0.0
    best_r1_2 = 0.0
    best_epoch_0 = None
    best_epoch_1 = None
    best_epoch_2 = None

    # train
    active_tasks = {token.strip() for token in args.loss_names.split("+") if token.strip()}
    if not active_tasks:
        active_tasks = {"sdm"}
    if "base" in active_tasks:
        active_tasks.add("sdm")
    use_ground_line = bool(active_tasks & {"cda", "bridge", "ga_bridge"})

    for epoch in range(start_epoch, num_epoch + 1):
        with torch.no_grad():
            if epoch % 1 == 0: 
                logger.info('Reconstruct the train loader')
                train_loader = build_filter_loader(args, trainset, epoch=epoch)
        
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()

        for n_iter, batch in enumerate(train_loader):
            # batch = {k: v.cuda() for k, v in batch.items()}

            image = batch['images'].cuda()
            text = batch['caption_ids'].cuda()
            ori_text = batch['caption_ids_ori'].cuda()
            ground_image = None
            if use_ground_line:
                if 'ground_imgs' not in batch:
                    raise ValueError("loss_names includes cda/bridge, but the train loader did not provide ground_imgs.")
                ground_image = batch['ground_imgs'].cuda()

            model_outputs = model(image, text, ori_text, ground_image)
            if len(model_outputs) == 5:
                i_feats, text_feats, fu_i_feats, fu_t_feats, ground_feats = model_outputs
            else:
                i_feats, text_feats, fu_i_feats, fu_t_feats = model_outputs
                ground_feats = None

            caption_ids = text
            t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
            i_cls_feats = i_feats[:, 0, :].float()
            logit_scale = model.module.logit_scale.to(i_cls_feats.device)
            
            total_loss = i_cls_feats.sum() * 0.0
            loss_sdm = None
            if 'sdm' in active_tasks:
                loss_sdm = objectives.compute_sdm(i_cls_feats, t_feats, batch['pids'].cuda(), logit_scale)
                total_loss = total_loss + loss_sdm
            
            loss_cda = None
            if 'cda' in active_tasks:
                if ground_feats is None:
                    raise ValueError("cda loss requires ground image features.")
                loss_cda = objectives.compute_selective_align_loss(
                    i_cls_feats,
                    ground_feats[:, 0, :].float(),
                    t_feats,
                    batch['pids'].cuda(),
                    logit_scale,
                ) * getattr(args, "cda_loss_weight", 1.0)
                total_loss = total_loss + loss_cda

            loss_fta = None
            if 'fta' in active_tasks:
                loss_fta = model.module.compute_fta_loss(
                    i_feats,
                    text_feats,
                    i_cls_feats,
                    t_feats,
                    batch['pids'].cuda(),
                    logit_scale,
                )
                total_loss = total_loss + loss_fta

            with torch.no_grad():
                similarity_matrix = torch.einsum('nld,nkd->nlk', [F.normalize(fu_t_feats,dim=-1), F.normalize(fu_i_feats[:,1:,:],dim=-1)])
                similarity_matrix = similarity_matrix.max(-1)[0]
                for idx, sim in zip(batch['image_ids'].data, similarity_matrix):
                    sample_idx = int(idx)
                    sim_value = sim.data.cpu().numpy()
                    try:
                        trainset[sample_idx][-1] = sim_value
                    except TypeError:
                        trainset[sample_idx] = tuple(list(trainset[sample_idx]) + [sim_value])

            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            if loss_sdm is not None:
                meters['sdm_loss'].update(loss_sdm, batch_size)
            if loss_cda is not None:
                meters['cda_loss'].update(loss_cda, batch_size)
            if loss_fta is not None:
                meters['fta_loss'].update(loss_fta, batch_size)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
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
            swanlab_payload = {
                "epoch": epoch,
                "lr": scheduler.get_lr()[0],
            }
            for k, v in meters.items():
                if v.avg > 0:
                    swanlab_payload[f"train/{k}"] = v.avg
            swanlab_run.log(swanlab_payload)


        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / 60
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[min] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            logger.info(f"best R1: val0 {best_r1_0}, val1 {best_r1_1}, val2 {best_r1_2}")
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                eval_model = model.module.eval()
                metrics0 = evaluator0.eval(eval_model, return_details=True) if evaluator0 is not None else None
                metrics1 = evaluator1.eval(eval_model, return_details=True) if evaluator1 is not None else None
                metrics2 = evaluator2.eval(eval_model, return_details=True) if evaluator2 is not None else None
                r1_0 = metrics0["t2i_R1"] if metrics0 is not None else best_r1_0
                r1_1 = metrics1["t2i_R1"] if metrics1 is not None else best_r1_1
                r1_2 = metrics2["t2i_R1"] if metrics2 is not None else best_r1_2
                if swanlab_run is not None:
                    eval_payload = {"epoch": epoch}
                    if metrics0 is not None:
                        eval_payload.update({f"val0/{k}": v for k, v in metrics0.items()})
                    if metrics1 is not None:
                        eval_payload.update({f"val1/{k}": v for k, v in metrics1.items()})
                    if metrics2 is not None:
                        eval_payload.update({f"val2/{k}": v for k, v in metrics2.items()})
                    swanlab_run.log(eval_payload)
                torch.cuda.empty_cache()
                if best_r1_0 < r1_0:
                    best_r1_0 = r1_0
                    arguments["epoch"] = epoch
                    best_epoch_0 = epoch
                    checkpointer.save("best0", **arguments)
                if best_r1_1 < r1_1:
                    best_r1_1 = r1_1
                    arguments["epoch"] = epoch
                    best_epoch_1 = epoch
                    checkpointer.save("best1", **arguments)
                if best_r1_2 < r1_2:
                    best_r1_2 = r1_2
                    arguments["epoch"] = epoch
                    best_epoch_2 = epoch
                    checkpointer.save("best2", **arguments)
    if get_rank() == 0:
        logger.info(
            f"best R1: val0 {best_r1_0} at epoch {best_epoch_0}, "
            f"val1 {best_r1_1} at epoch {best_epoch_1}, "
            f"val2 {best_r1_2} at epoch {best_epoch_2}"
        )


def do_inference(model, test_img_loader, test_txt_loader, swanlab_run=None):

    logger = logging.getLogger("IRRA.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    metrics = evaluator.eval(model.eval(), return_details=True)
    if swanlab_run is not None:
        swanlab_run.log({f"test/{k}": v for k, v in metrics.items()})
    return metrics
