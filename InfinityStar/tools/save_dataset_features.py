# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
import os
import os.path as osp
import time
import gc
import json
import math
import random
import sys
import argparse
import copy
import traceback
import collections
from collections import deque
from contextlib import nullcontext
from functools import partial
from typing import List, Optional, Tuple
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'
import threading

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.distributed as tdist
import tqdm

from tools.run_infinity import *
from infinity.dataset.dataset_joint_vi import JointViIterableDataset
from infinity.schedules.dynamic_resolution import get_dynamic_resolution_meta



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument('--reweight_loss_by_scale', type=int, default=1, choices=[0,1])
    parser.add_argument('--vis_model_flop_param', type=int, default=0, choices=[0,1])
    parser.add_argument('--image_data_path', type=str, default='')
    parser.add_argument('--video_data_path', type=str, default='')
    parser.add_argument('--video_batch_size', type=int, default=1)
    parser.add_argument('--image_batch_size', type=int, default=1)
    parser.add_argument('--dataloader_workers', type=int, default=12)
    parser.add_argument('--noise_apply_layers', type=int, default=20)
    parser.add_argument('--noise_apply_requant', type=int, default=1, choices=[0,1])
    parser.add_argument('--noise_apply_strength', type=float, default=0.2)
    parser.add_argument('--debug_bsc', type=int, default=0, choices=[0,1])
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--video_fps', type=int, default=24)
    parser.add_argument('--steps_per_frame', type=int, default=4)
    parser.add_argument('--video_tower_style', type=str, default='bottom')
    parser.add_argument('--use_slice', type=int, default=1, choices=[0,1])
    parser.add_argument('--use_vae_token_cache', type=int, default=1, choices=[0,1])
    parser.add_argument('--allow_online_vae_feature_extraction', type=int, default=1, choices=[0,1])
    parser.add_argument('--use_text_token_cache', type=int, default=1, choices=[0,1])
    parser.add_argument('--image_batches_multiply', type=float, default=1)
    parser.add_argument('--token_cache_dir', type=str, default='/mnt/bn/genai-data2/hanjian.thu123/vae_features')
    parser.add_argument('--down_size_limit', type=int, default=10000)
    parser.add_argument('--addition_pn006M', type=int, default=1, choices=[0,1])
    parser.add_argument('--addition_pn025M', type=int, default=1, choices=[0,1])
    parser.add_argument('--video_caption_type', type=str, default='tarsier2_caption')
    parser.add_argument('--only_images4extract_feats', type=int, default=1, choices=[0,1])
    parser.add_argument('--temporal_compress_rate', type=int, default=4)
    parser.add_argument('--cached_video_frames', type=int, default=81)
    parser.add_argument('--duration_resolution', type=float, default=0.005)
    parser.add_argument('--train_max_token_len', type=int, default=20480)
    parser.add_argument('--cache_check_mode', type=int, default=0)
    parser.add_argument('--seq_pack_bucket', type=int, default=1)
    parser.add_argument('--drop_long_video', type=int, default=0)
    parser.add_argument('--append_duration2caption', type=int, default=0)
    parser.add_argument('--min_video_frames', type=int, default=32)
    parser.add_argument('--addition_pn_list', type=str, default='[]')
    parser.add_argument('--semantic_scale_dim', type=int, default=16)
    parser.add_argument('--detail_scale_dim', type=int, default=64)
    parser.add_argument('--use_learnable_dim_proj', type=int, default=0)
    parser.add_argument('--detail_scale_min_tokens', type=int, default=80)
    parser.add_argument('--semantic_scales', type=int, default=80)

    parser.add_argument('--tlen', type=int, default=512)
    parser.add_argument('--manual_parallel', action="store_true")
    parser.add_argument('--num_replicas', type=int, default=-1) # only valid when manual_parallel is True
    parser.add_argument('--rank', type=int, default=-1) # only valid when manual_parallel is True
    parser.add_argument('--restrict_data_size', type=int, default=-1)
    parser.add_argument('--allow_less_one_elem_in_seq', type=int, default=1)
    parser.add_argument('--use_feat_proj', type=int, default=2)
    parser.add_argument('--use_two_stage_lfq', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=0)

    args = parser.parse_args()

    if args.manual_parallel:
        device = "cuda:0"
        num_replicas = args.num_replicas
        rank = args.rank
        assert num_replicas > 0 and rank >= 0
    else:
        tdist.init_process_group(backend='nccl')
        device = torch.device(tdist.get_rank() % torch.cuda.device_count())
        num_replicas = tdist.get_world_size()
        rank=tdist.get_rank()
    args.device = device
    args.text_tokenizer = None
    args.duration_resolution = 4 / args.video_fps

    # load vae
    vae = load_visual_tokenizer(args, device=device)

    dataset = JointViIterableDataset(
        image_meta_folder=args.image_data_path, 
        video_meta_folder=args.video_data_path, 
        max_caption_len=512,
        short_prob=0.0, 
        load_vae_instead_of_image=False, 
        pn=args.pn,
        seed=args.seed,
        video_fps=args.video_fps,
        num_frames=args.video_frames,
        online_t5=True,
        num_replicas=num_replicas, # 1,
        rank=rank, # 0
        dataloader_workers=args.dataloader_workers,
        dynamic_resolution_across_gpus=0,
        enable_dynamic_length_prompt=0,
        dynamic_scale_schedule=args.dynamic_scale_schedule,
        add_motion_score2caption=0,
        other_args=args,
    )
    dataloader = DataLoader(dataset, batch_size=None, num_workers=args.dataloader_workers, pin_memory=True)
    print(f'len(dataloader): {len(dataloader)}, len(dataset): {len(dataset)}')
    t1 = time.time()
    dataloader.dataset.set_epoch(0)
    pbar = tqdm.tqdm(total=len(dataloader))
    accumulate_res = collections.defaultdict(list)
    dynamic_resolution_h_w, h_div_w_templates = get_dynamic_resolution_meta(args.dynamic_scale_schedule, args.video_frames)
    
    print(device)
    vae.to(device)

    def save_token():
        while True:
            try:
                raw_features, feature_cache_files4images = save_token_queue.get()
                for i in range(len(feature_cache_files4images)):
                    if not osp.exists(feature_cache_files4images[i]):
                        os.makedirs(osp.dirname(feature_cache_files4images[i]), exist_ok=True)
                        torch.save(raw_features[i], feature_cache_files4images[i])
                        print(f'Save to {feature_cache_files4images[i]}')
                    else:
                        print(f'{feature_cache_files4images[i]} exists, skip')
            except Exception as e:
                print(f"Error saving token: {e}")
            finally:
                save_token_queue.task_done()

    import queue
    save_token_queue = queue.Queue()
    saver = threading.Thread(target=save_token, daemon=True)
    saver.start()

    data_time = time.time()
    iter_time = time.time()

    pn_list = [args.pn] + json.loads(args.addition_pn_list)
    pn_list = list(set(pn_list))
    
    for i, data in enumerate(iter(dataloader)):
        pbar.update(1)
        # print(f"[step {i}]: iter time: {time.time() - iter_time:.2f}s, data time:{time.time() - data_time:.2f}s")
        iter_time = time.time()
        # print("data time", iter_time - data_time)

        captions, feature_cache_files4images, raw_features_bcthw = data['captions'], data['feature_cache_files4images'], data['raw_features_bcthw']
        # print(len(feature_cache_files4images))
        if args.only_images4extract_feats:
            assert len(raw_features_bcthw) == 0
        if not len(feature_cache_files4images):
            continue

        for pn_ind, pn in enumerate(pn_list):
            if pn == args.pn:
                inp_B3HW = data['images']
            else:
                inp_B3HW = data['addition_pn_images'][f'img_T3HW_{pn}']
            try:
                # print(f"args.pn:{args.pn}, pn:{pn}")
                cur_feature_cache_files4images = [item.replace(f'pn_{args.pn}', f'pn_{pn}') for item in feature_cache_files4images]
            except Exception as e:
                import pdb; pdb.set_trace()
            assert len(inp_B3HW) == len(cur_feature_cache_files4images)

            for images_CTHW, feature_save_file in zip(inp_B3HW, cur_feature_cache_files4images):
                try:
                    pt = images_CTHW.shape[-3]
                    h_div_w = images_CTHW.shape[-2] / images_CTHW.shape[-1]
                    h_div_w_templates = np.array(list(dynamic_resolution_h_w.keys()))
                    h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w-h_div_w_templates))]
                    # [forward]
                    with torch.amp.autocast('cuda', enabled=False):
                        with torch.no_grad():
                            raw_features, _, _ = vae.encode_for_raw_features(images_CTHW.unsqueeze(0).to(device), scale_schedule=None, slice=args.use_slice)
                            raw_features = raw_features.cpu().data
                            save_token_queue.put((raw_features, [feature_save_file]))
                except Exception as e:
                    print(e)
            data_time = time.time()
            # print("iter time", data_time - iter_time)
    save_token_queue.join()
