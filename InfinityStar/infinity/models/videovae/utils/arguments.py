# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def add_model_specific_args(args, parser):
    from infinity.models.videovae.models import CVIVIT_VQGAN, CNN_VQGAN, FLUX_VAE, MS_VAE, CogVAE, SlowFastVAE, HunYuanVAE, CogVAEL
    
    if args.tokenizer == "cvivit":
        parser = CVIVIT_VQGAN.add_model_specific_args(parser)
        vae_model = CVIVIT_VQGAN
    elif args.tokenizer == "cnn":
        parser = CNN_VQGAN.add_model_specific_args(parser)
        vae_model = CNN_VQGAN
    elif args.tokenizer in ["flux"]:  # add cogvideo here to align evaluation configs
        parser = CNN_VQGAN.add_model_specific_args(parser) # align with cnn config
        parser = FLUX_VAE.add_model_specific_args(parser) # flux config
        vae_model = FLUX_VAE
    elif args.tokenizer == "ms":
        parser = CNN_VQGAN.add_model_specific_args(parser) # align with cnn config
        parser = FLUX_VAE.add_model_specific_args(parser) # align with flux config
        vae_model = MS_VAE
    elif args.tokenizer in ["sd", "sd-vq", "mar", "cogvideox_origin", "vidtok", "open-sora-plan", "step-fun", "hunyuan_origin"]:
        vae_model = None
        pass
    elif args.tokenizer in ["cogvideox"]:
        parser = CogVAE.add_model_specific_args(parser)
        parser = FLUX_VAE.add_model_specific_args(parser)
        vae_model = CogVAE
    elif args.tokenizer in ["cogvideoxl"]:
        parser = CogVAEL.add_model_specific_args(parser)
        parser = FLUX_VAE.add_model_specific_args(parser)
        vae_model = CogVAEL
    elif args.tokenizer in ["slow-fast"]:
        parser = SlowFastVAE.add_model_specific_args(parser)
        parser = FLUX_VAE.add_model_specific_args(parser)
        vae_model = SlowFastVAE
    elif args.tokenizer in ["hunyuan"]:
        parser = HunYuanVAE.add_model_specific_args(parser)
        parser = FLUX_VAE.add_model_specific_args(parser)
        vae_model = HunYuanVAE
    else:
        raise NotImplementedError
    return args, parser, vae_model

class MainArgs:
    @staticmethod
    def add_main_args(parser):
        # training
        parser.add_argument('--max_steps', type=int, default=1e6)
        parser.add_argument('--log_every', type=int, default=1)
        parser.add_argument('--ckpt_every', type=int, default=1000)
        parser.add_argument('--default_root_dir', type=str, required=True)
        parser.add_argument('--compile', type=str, default="no", choices=["no", "yes"])
        parser.add_argument('--ema', type=str, default="no", choices=["no", "yes"])
        parser.add_argument('--mfu_logging', type=str, default="no", choices=["no", "yes"])
        parser.add_argument('--dataloader_init_epoch', type=int, default=-1)
        parser.add_argument('--context_parallel_size', type=int, default=0)

        # optimization
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--beta1', type=float, default=0.9)
        parser.add_argument('--beta2', type=float, default=0.95)
        parser.add_argument('--optim_type', type=str, default="Adam", choices=["Adam", "AdamW"])
        parser.add_argument('--disc_optim_type', type=str, default=None, choices=[None, "rmsprop"])
        parser.add_argument('--max_grad_norm', type=float, default=1.0)
        parser.add_argument('--max_grad_norm_disc', type=float, default=1.0)
        parser.add_argument('--disable_sch', action="store_true") # deprecated option
        parser.add_argument('--scheduler', type=str, default="no", choices=["no", "linear"])
        parser.add_argument('--warmup_steps', type=int, default=0)
        parser.add_argument('--lr_min', type=float, default=0.)
        parser.add_argument('--warmup_lr_init', type=float, default=0.)

        # basic vae config
        parser.add_argument('--patch_size', type=int, default=8)
        parser.add_argument('--temporal_patch_size', type=int, default=4)
        parser.add_argument('--embedding_dim', type=int, default=256)
        parser.add_argument('--codebook_dim', type=int, default=16)
        parser.add_argument('--use_vae', action="store_true")
        parser.add_argument('--fix_model', type=str, default='no', choices=['no', 'encoder', 'encoder_decoder'])

        # discrete vae config
        parser.add_argument('--use_stochastic_depth', action="store_true")
        parser.add_argument("--drop_rate", type=float, default=0.0)
        parser.add_argument('--schedule_mode', type=str, default="original", choices=["original", "dynamic", "dense", "same1", "same2", "same3", "half", "dense_f8", "dense_f8_double"])
        parser.add_argument('--lr_drop', nargs='*', type=int, default=None, help="A list of numeric values. Example: --values 270 300")
        parser.add_argument('--lr_drop_rate', type=float, default=0.1)
        parser.add_argument('--keep_first_quant', action="store_true")
        parser.add_argument('--keep_last_quant', action="store_true")
        parser.add_argument('--remove_residual_detach', action="store_true")
        parser.add_argument('--use_out_phi', action="store_true")
        parser.add_argument('--use_out_phi_res', action="store_true")
        parser.add_argument('--use_lecam_reg', action="store_true")
        parser.add_argument('--lecam_weight', type=float, default=0.05)
        parser.add_argument('--perceptual_model', type=str, default="vgg16", choices=["vgg16", "resnet50", "resnet50_v2"])
        parser.add_argument('--base_ch_disc', type=int, default=64)
        parser.add_argument('--random_flip', action="store_true")
        parser.add_argument('--flip_prob', type=float, default=0.5)
        parser.add_argument('--flip_mode', type=str, default="stochastic", choices=["stochastic", "deterministic", "stochastic_dynamic"])
        parser.add_argument('--max_flip_lvl', type=int, default=1)
        parser.add_argument('--not_load_optimizer', action="store_true")
        parser.add_argument('--use_lecam_reg_zero', action="store_true")
        parser.add_argument('--freeze_encoder', action="store_true")
        parser.add_argument('--rm_downsample', action="store_true")
        parser.add_argument('--random_flip_1lvl', action="store_true")
        parser.add_argument('--flip_lvl_idx', type=int, default=0)
        parser.add_argument('--drop_when_test', action="store_true")
        parser.add_argument('--drop_lvl_idx', type=int, default=None)
        parser.add_argument('--drop_lvl_num', type=int, default=0)
        parser.add_argument('--compute_all_commitment', action="store_true")
        parser.add_argument('--disable_codebook_usage', action="store_true")
        parser.add_argument('--freeze_enc_main', action="store_true")
        parser.add_argument('--freeze_dec_main', action="store_true")
        parser.add_argument('--random_short_schedule', action="store_true")
        parser.add_argument('--short_schedule_prob', type=float, default=0.5)
        parser.add_argument('--use_bernoulli', action="store_true")
        parser.add_argument('--use_rot_trick', action="store_true")
        parser.add_argument('--disable_flip_prob', type=float, default=0.0)
        parser.add_argument('--dino_disc', action="store_true")
        parser.add_argument('--quantizer_type', type=str, default='MultiScaleBSQ')
        parser.add_argument('--lfq_weight', type=float, default=0.)
        parser.add_argument('--entropy_loss_weight', type=float, default=0.1)
        parser.add_argument('--visu_every', type=int, default=1000)
        parser.add_argument('--commitment_loss_weight', type=float, default=0.25)
        parser.add_argument('--bsq_version', type=str, default="v1", choices=["v1", "v2"])
        parser.add_argument('--diversity_gamma', type=float, default=1)
        parser.add_argument('--bs1_for1024', action="store_true")
        parser.add_argument('--casual_multi_scale', action="store_true")
        parser.add_argument('--double_compress_t', action="store_true")
        parser.add_argument('--temporal_slicing', action="store_true")
        parser.add_argument('--latent_adjust_type', type=str, default=None)
        parser.add_argument('--compute_latent_loss', action="store_true")
        parser.add_argument('--latent_loss_weight', type=float, default=0.0)

        # discriminator config
        parser.add_argument('--disc_version', type=str, default="v1")
        parser.add_argument('--magvit_disc', action="store_true") # deprecated
        parser.add_argument('--disc_type', type=str, default="patchgan", choices=["patchgan", "stylegan"])
        parser.add_argument('--sigmoid_in_disc', action="store_true")
        parser.add_argument('--activation_in_disc', type=str, default="leaky_relu")
        parser.add_argument('--apply_blur', action="store_true")
        parser.add_argument('--apply_noise', action="store_true")
        parser.add_argument('--dis_warmup_steps', type=int, default=0)
        parser.add_argument('--dis_lr_multiplier', type=float, default=1.)
        parser.add_argument('--dis_minlr_multiplier', action="store_true")
        parser.add_argument('--disc_channels', type=int, default=64)
        parser.add_argument('--disc_layers', type=int, default=3)
        parser.add_argument('--discriminator_iter_start', type=int, default=0)
        parser.add_argument('--disc_pretrain_iter', type=int, default=0)
        parser.add_argument('--disc_optim_steps', type=int, default=1)
        parser.add_argument('--disc_warmup', type=int, default=0)
        parser.add_argument('--disc_pool', type=str, default="no", choices=["no", "yes"])
        parser.add_argument('--disc_pool_size', type=int, default=100)
        parser.add_argument('--disc_temporal_compress', type=str, default="yes", choices=["no", "yes"])
        parser.add_argument('--disc_use_blur', type=str, default="yes", choices=["no", "yes"])
        parser.add_argument('--disc_stylegan_downsample_base', type=int, default=2)

        parser = MainArgs.add_loss_args(parser)
        parser = MainArgs.add_accelerate_args(parser)

        # initialization
        parser.add_argument('--tokenizer', type=str, required=True)
        parser.add_argument('--pretrained', type=str, default=None)
        parser.add_argument('--pretrained_mode', type=str, default="full")
        parser.add_argument('--pretrained_ema', type=str, default="no")
        parser.add_argument('--inflation_pe', action="store_true")
        parser.add_argument('--init_vgen', type=str, default='no', choices=['no', 'keep', 'average'])
        parser.add_argument('--no_init_idis', action="store_true") # deprecated option
        parser.add_argument('--init_idis', type=str, default='keep', choices=['no', 'keep']) # use keep by default following previous settings
        parser.add_argument('--init_vdis', type=str, default="no")

        # misc
        parser.add_argument('--enable_nan_detector', action='store_true')
        parser.add_argument('--turn_on_profiler', action='store_true')
        parser.add_argument('--profiler_scheduler_wait_steps', type=int, default=10)
        parser.add_argument('--debug', action='store_true')
        parser.add_argument('--video_logger', action='store_true') # deprecated option
        parser.add_argument('--bytenas', type=str, default="sg")
        parser.add_argument('--username', type=str, default="zhufengda")
        parser.add_argument('--seed', type=int, default=1234)
        parser.add_argument('--vq_to_vae', action='store_true')
        parser.add_argument('--load_not_strict', action='store_true')
        parser.add_argument('--zero', type=int, default=0, choices=[0, 1, 2, 3]) # 1 hybrid shard, 2 shard grad_op, 3 full shard
        parser.add_argument('--bucket_cap_mb', type=int, default=40) # DDP
        parser.add_argument('--manual_gc_interval', type=int, default=10000) # DDP

        return parser
    
    @staticmethod
    def add_loss_args(parser):
        parser.add_argument("--recon_loss_type", type=str, default='l1', choices=['l1', 'l2'])
        parser.add_argument('--video_perceptual_weight', type=float, default=0.)
        parser.add_argument('--image_gan_weight', type=float, default=1.0)
        parser.add_argument('--video_gan_weight', type=float, default=1.0)
        parser.add_argument('--image_disc_weight', type=float, default=0.)
        parser.add_argument('--video_disc_weight', type=float, default=0.)
        parser.add_argument('--l1_weight', type=float, default=4.0)
        parser.add_argument('--gan_feat_weight', type=float, default=0.0)
        parser.add_argument('--lpips_model', type=str, default='vgg', choices=['vgg', 'resnet50'])
        parser.add_argument('--perceptual_weight', type=float, default=0.0)
        parser.add_argument('--kl_weight', type=float, default=0.)
        parser.add_argument('--norm_type', type=str, default='group', choices=['batch', 'group', "no"])
        parser.add_argument('--disc_loss_type', type=str, default='hinge', choices=['hinge', 'vanilla'])
        parser.add_argument('--gan_image4video', type=str, default='yes', choices=['no', 'yes'])
        return parser 

    @staticmethod
    def add_accelerate_args(parser):
        parser.add_argument('--use_checkpoint', action="store_true")
        parser.add_argument('--precision', type=str, default="fp32", choices=['fp32', 'bf16']) # disable fp16
        parser.add_argument('--encoder_dtype', type=str, default="fp32", choices=['fp32', 'bf16']) # disable fp16
        parser.add_argument('--decoder_dtype', type=str, default="fp32", choices=['fp32', 'bf16']) # disable fp16
        parser.add_argument('--upcast_attention', type=str, default="", choices=["qk", "qkv"])
        parser.add_argument('--upcast_tf32', action="store_true")
        return parser

def format_args(args):
    # Start building the script string
    script_content = "#!/bin/bash\n\n"
    script_content += "torchrun \\\n"
    script_content += "    --nproc_per_node=$ARNOLD_WORKER_GPU \\\n"
    script_content += "    --nnodes=$ARNOLD_WORKER_NUM --master_addr=$ARNOLD_WORKER_0_HOST \\\n"
    script_content += "    --node_rank=$ARNOLD_ID --master_port=$port \\\n"
    script_content += "    train.py \\\n"

    # Iterate over each key-value pair and append it to the command
    for k, v in args.__dict__.items():
        script_content += f"    --{k} {v} \\\n"

    # Remove the last backslash and newline
    script_content = script_content.rstrip(" \\\n") + "\n"
    return script_content

def init_resolution(resolution, num_datasets):
    if len(resolution) == 1:
        resolution = [(resolution[0], resolution[0])] * num_datasets
    elif len(resolution) == num_datasets:
        resolution = [(resolution[i], resolution[i]) for i in range(len(resolution))]
    elif len(resolution) == num_datasets * 2:
        resolution = [(resolution[i], resolution[i+1]) for i in range(0, len(resolution), 2)]
    else:
        raise NotImplementedError
    return resolution
