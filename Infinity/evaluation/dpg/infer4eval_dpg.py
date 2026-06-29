import os
import os.path as osp
import hashlib
import time
import argparse
import json
import shutil
import glob
import re
import sys

import cv2
import tqdm
import torch
import numpy as np
from pytorch_lightning import seed_everything

import os
import sys

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(parent_path)

from infinity.utils.csv_util import load_csv_as_dicts, write_dicts2csv_file
from tools.run_infinity import *
from conf import HF_TOKEN, HF_HOME

# set environment variables
os.environ['HF_TOKEN'] = HF_TOKEN
os.environ['HF_HOME'] = HF_HOME
os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'

rank_ratio_path = "data.jsonl"
open(rank_ratio_path, "w").close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument('--outdir', type=str, default='dpg_outputs')
    parser.add_argument('--n_samples', type=int, default=4)
    parser.add_argument('--prompt_path', type=str, default='Infinity/evaluation/dpg/dpg_bench/prompts')
    args = parser.parse_args()

    # parse cfg
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]

    prompt_files = os.listdir(args.prompt_path)
    prompt_files.sort()
    prompts = []
    for prompt_file in prompt_files:
        prompt_path = os.path.join(args.prompt_path, prompt_file)
        with open(prompt_path, 'r', encoding='utf-8') as file:
            prompt = file.read()
        prompts += [(prompt_file.split('.')[0], prompt)]

    assert 'infinity' in args.model_type
    if 'infinity' in args.model_type:
        # load text encoder
        text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
        # load vae
        vae = load_visual_tokenizer(args)
        # load infinity
        infinity = load_transformer(vae, args)

    for index, (prompti, prompt) in enumerate(prompts):
        seed_everything(args.seed, verbose=False)
        outpath = os.path.join(args.outdir)
        os.makedirs(outpath, exist_ok=True)

        tau = args.tau
        cfg = args.cfg
        images = []
        for sample_j in range(args.n_samples):
            if 'infinity' in args.model_type:
                h_div_w_template = 1.000
                scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
                scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
                tgt_h, tgt_w = dynamic_resolution_h_w[h_div_w_template][args.pn]['pixel']
                image = gen_one_img(infinity, vae, text_tokenizer, text_encoder, prompt, tau_list=tau, cfg_sc=3,
                                    cfg_list=cfg, scale_schedule=scale_schedule,
                                    cfg_insertion_layer=[args.cfg_insertion_layer], vae_type=args.vae_type)
            else:
                raise ValueError
            images.append(image)

        grid = [[images[0], images[1]], [images[2], images[3]]]
        # 水平拼接每一行
        row1 = torch.cat(grid[0], dim=0)  # 水平拼接：dim=2 (宽度维度)
        row2 = torch.cat(grid[1], dim=0)
        # 垂直拼接两行
        combined = torch.cat([row1, row2], dim=1)  # 垂直拼接：dim=1 (高度维度)

        prompt_name = f'partiprompts{prompti}'
        save_file = os.path.join(outpath, f"{prompt_name}.jpg")

        if 'infinity' in args.model_type:
            cv2.imwrite(save_file, combined.cpu().numpy())



