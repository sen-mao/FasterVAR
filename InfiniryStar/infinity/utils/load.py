# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
#!/usr/bin/python3
import torch

from infinity.models import Infinity
from infinity.utils import arg_util

def load_visual_tokenizer(args, device=None):
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.vae_type in [8,12,14,16,18,20,24,32,48,64,128]:
        schedule_mode = "dynamic"
        codebook_dim = args.vae_type # 18
        print(f'Load VAE from {args.vae_path}')

        if args.videovae == 10: # absorb patchify
            from infinity.models.videovae.models.load_vae_bsq_wan_absorb_patchify import video_vae_model
            vae_local = video_vae_model(args.vae_path, schedule_mode, codebook_dim, global_args=args, test_mode=True).to(device)
        else:
            raise ValueError(f"vae_type {args.vae_type} not supported")
    else:
        raise ValueError(f"vae_type {args.vae_type} not supported")
    return vae_local

def build_vae_gpt(args: arg_util.Args, force_flash=False, device='cuda'):
    vae_local = load_visual_tokenizer(args, device)

    if force_flash: args.flash = True
    gpt_kw = dict(
        text_channels=args.Ct5, 
        text_maxlen=args.tlen,
        norm_eps=args.norm_eps, 
        rms_norm=args.rms_norm,
        cond_drop_rate=args.cfg, 
        rand_uncond=args.rand_uncond,
        raw_scale_schedule=args.scale_schedule,
        top_p=args.topp,
        top_k=args.topk,
        checkpointing=args.enable_checkpointing,
        pad_to_multiplier=args.pad_to_multiplier,
        use_flex_attn=args.use_flex_attn,
        add_lvl_embeding_on_first_block=args.add_lvl_embeding_on_first_block,
        num_of_label_value=args.num_of_label_value,
        rope2d_each_sa_layer=args.rope2d_each_sa_layer,
        rope2d_normalized_by_hw=args.rope2d_normalized_by_hw,
        pn=args.pn,
        train_h_div_w_list=None,
        apply_spatial_patchify=args.apply_spatial_patchify,
        video_frames=args.video_frames,
        other_args=args,
    )
    
    print(f'[create gpt_wo_ddp] constructor kw={gpt_kw}\n')
    gpt_kw['vae_local'] = vae_local
    
    model_str = args.model.replace('vgpt', 'infinity')   # legacy
    print(f"{model_str=}")
    if model_str.rsplit('c', maxsplit=1)[-1].isdecimal():
        model_str, _ = model_str.rsplit('c', maxsplit=1)    
    from timm.models import create_model
    gpt_wo_ddp: Infinity = create_model(model_str, **gpt_kw)
    vae_local = vae_local.to('cuda')
    assert all(not p.requires_grad for p in vae_local.parameters())
    assert all(p.requires_grad for n, p in gpt_wo_ddp.named_parameters())
    return vae_local, gpt_wo_ddp
