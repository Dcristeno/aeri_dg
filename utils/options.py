import argparse


def get_args():
    parser = argparse.ArgumentParser(description="IRRA Args")
    ######################## general settings ########################
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--name", default="baseline", help="experiment name to save")
    parser.add_argument("--seed", default=1, type=int, help="random seed used for training and per-rank seeding")
    parser.add_argument("--output_dir", default="logs")
    parser.add_argument("--log_period", default=100)
    parser.add_argument("--eval_period", default=1)
    parser.add_argument("--val_dataset", default="test") # use val set when evaluate, if test use test set
    parser.add_argument("--resume", default=False, action='store_true')
    parser.add_argument("--resume_ckpt_file", default="", help='resume from ...')
    parser.add_argument("--finetune_eval_mode", default="test", type=str, help="finetune evaluation mode: test, heldout, or none")
    parser.add_argument("--use_swanlab", default=False, action='store_true', help="whether to log metrics to SwanLab")
    parser.add_argument("--swanlab_project", default="CFAN", help="SwanLab project name")
    parser.add_argument("--swanlab_experiment", default="", help="Optional SwanLab experiment name")
    parser.add_argument("--swanlab_mode", default="cloud", help="SwanLab mode, e.g. cloud or local")
    parser.add_argument("--save_optimizer", default=False, action='store_true', help="save optimizer and scheduler state in finetune checkpoints")

    parser.add_argument("--finetune", type=str, default="pretrain/HAMbest0.pth")
    parser.add_argument("--pretrain", type=str, default="")
    parser.add_argument("--nam", default=False, action='store_true')

    ######################## model general settings ########################
    parser.add_argument("--pretrain_choice", default='ViT-B/16') # whether use pretrained model
    parser.add_argument("--temperature", type=float, default=0.02, help="initial temperature value, if 0, don't use temperature")
    parser.add_argument("--img_aug", default=True, action='store_true')

    ## cross modal transfomer setting
    parser.add_argument("--cmt_depth", type=int, default=4, help="cross modal transformer self attn layers")
    parser.add_argument("--fta_num_query", type=int, default=4, help="number of learned query slots used by FTA")
    parser.add_argument("--fta_query_mode", type=str, default="static", help="FTA query mode: static or conditioned")
    parser.add_argument("--fta_query_condition_scale", type=float, default=1.0, help="scale applied to the instance-conditioned FTA query deltas")
    parser.add_argument("--masked_token_rate", type=float, default=0.8, help="masked token rate for mlm task")
    parser.add_argument("--masked_token_unchanged_rate", type=float, default=0.1, help="masked token unchanged rate")
    parser.add_argument("--lr_factor", type=float, default=5.0, help="lr factor for random init self implement module")
    parser.add_argument("--MLM", default=True, action='store_true', help="whether to use Mask Language Modeling dataset")

    ######################## loss settings ########################
    parser.add_argument("--loss_names", default='sdm', help="which loss to use ['mlm', 'cmpm', 'id', 'itc', 'sdm']")
    parser.add_argument("--mlm_loss_weight", type=float, default=1.0, help="mlm loss weight")
    parser.add_argument("--id_loss_weight", type=float, default=1.0, help="id loss weight")
    parser.add_argument("--bridge_loss_weight", type=float, default=1.0, help="overall weight for the explicit ground-to-aerial bridge loss")
    parser.add_argument("--bridge_pair_weight", type=float, default=1.0, help="weight of the pair-level aerial-to-ground bridge term")
    parser.add_argument("--bridge_distill_weight", type=float, default=1.0, help="weight of the ground-to-text -> aerial-to-text relation distillation term")
    parser.add_argument("--bridge_distill_temp", type=float, default=0.07, help="temperature for the relation distillation bridge term")
    parser.add_argument("--aerial_prototype_path", type=str, default="", help="path to a fixed pid-indexed aerial prototype tensor file")
    parser.add_argument("--aerial_prototype_weight", type=float, default=0.0, help="weight for the fixed aerial trajectory prototype pull loss")
    parser.add_argument("--track_memory_loss_weight", type=float, default=0.0, help="overall weight for the online track-memory alignment loss")
    parser.add_argument("--track_memory_image_weight", type=float, default=1.0, help="weight for aligning sampled aerial features to their online track memory")
    parser.add_argument("--track_memory_text_weight", type=float, default=1.0, help="weight for aligning text features to their online track memory")
    parser.add_argument("--track_memory_momentum", type=float, default=0.8, help="EMA momentum used to update each pid track memory")
    
    ######################## vison trainsformer settings ########################
    parser.add_argument("--img_size", type=tuple, default=(384, 128))
    parser.add_argument("--stride_size", type=int, default=16)

    ######################## text transformer settings ########################
    parser.add_argument("--text_length", type=int, default=77)
    parser.add_argument("--vocab_size", type=int, default=49408)

    ######################## solver ########################
    parser.add_argument("--optimizer", type=str, default="Adam", help="[SGD, Adam, Adamw]")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr2", type=float, default=1e-5)
    parser.add_argument("--bias_lr_factor", type=float, default=2.)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--weight_decay_bias", type=float, default=0.)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.999)
    
    ######################## scheduler ########################
    parser.add_argument("--num_epoch", type=int, default=60)
    parser.add_argument("--milestones", type=int, nargs='+', default=(20, 40))
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warmup_factor", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_method", type=str, default="linear")
    parser.add_argument("--lrscheduler", type=str, default="cosine")
    parser.add_argument("--target_lr", type=float, default=0)
    parser.add_argument("--power", type=float, default=0.9)

    ######################## dataset ########################
    parser.add_argument("--dataset_name", default="AERI-PEDES", help="[CUHK-PEDES, AERI-PEDES, AGDataAttr]")
    parser.add_argument("--sampler", default="random", help="choose sampler from [idtentity, random]")
    parser.add_argument("--num_instance", type=int, default=4)
    parser.add_argument("--root_dir", default="/data1/Datasets/ReID/")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--finetune_val_ratio", type=float, default=0.1, help="fraction of training identities held out as validation during finetuning when finetune_eval_mode=heldout")
    parser.add_argument("--finetune_val_seed", type=int, default=1, help="seed used to split finetune train identities into train/val when finetune_eval_mode=heldout")
    parser.add_argument("--train_samples_per_id", type=int, default=0, help="if > 0, resample the finetune train set each epoch with k samples per identity")
    parser.add_argument("--train_sample_strategy", type=str, default="random", help="per-id sampling strategy: random, random_diverse, sharpness_topk, or mid_sharpness_diverse")
    parser.add_argument("--train_sample_mid_ratio", type=float, default=0.6, help="fraction of per-id images kept as the middle-sharpness candidate pool")
    parser.add_argument("--train_sample_representative_weight", type=float, default=1.0, help="weight for representative-score in heuristic per-id sampling")
    parser.add_argument("--train_sample_diversity_weight", type=float, default=0.5, help="weight for redundancy penalty in heuristic per-id sampling")
    parser.add_argument("--train_sample_mid_weight", type=float, default=0.5, help="weight for preferring mid-sharpness images in heuristic per-id sampling")
    parser.add_argument("--train_tile_mix_grid", type=int, default=0, help="if > 1, randomly mosaic same-pid aerial images using an NxN grid during training")
    parser.add_argument("--train_tile_mix_prob", type=float, default=0.0, help="probability of applying same-pid aerial tile mixing to each training sample")
    parser.add_argument("--test", dest='training', default=True, action='store_false')

    args = parser.parse_args()

    return args
