#!/usr/bin/env bash

set -x

# Configure different distributed environment variables according to your platform.
# nccl setting
unset NCCL_NET_PLUGIN
unset NCCL_FASTRAK_ENABLE
unset NCCL_FASTRAK_USE_SNAP
unset NCCL_FASTRAK_NUM_FLOWS
unset NCCL_FASTRAK_*
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1
export NCCL_NET_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_NET_PLUGIN=NONE
export NCCL_NVLS_ENABLE=0

# arnold setting
unset ARNOLD_WORKER_0_HOST

TORCHRUN_RDZV_READ_TIMEOUT=${TORCHRUN_RDZV_READ_TIMEOUT:-600}
ARNOLD_ID=${ARNOLD_ID:-0}
ARNOLD_WORKER_0_HOST=${ARNOLD_WORKER_0_HOST:-'localhost'}
ARNOLD_WORKER_0_PORT=${ARNOLD_WORKER_0_PORT:-'9591'}
RUN_COMMAND=${@:2}
PORT=$(echo "$ARNOLD_WORKER_0_PORT" | cut -d "," -f 1)
########---------------------------------------------------------------------------------------------------


export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHINDUCTOR_COMPILE_THREADS=1
export OMP_NUM_THREADS=8

# set exp args
video_data_path='./data/infinitystar_toy_data/split_jsonls'

checkpoints_dir='./'
vae_path="${checkpoints_dir}infinitystar_videovae.pth"
token_cache_dir=./checkpoints/local_cache/cached_visual_tokens_720p

# --pn: 0.90M for 720p, 0.40M for 480p \


torchrun --nproc_per_node=$ARNOLD_WORKER_GPU \
    --nnodes=$ARNOLD_WORKER_NUM \
    --master_addr=$ARNOLD_WORKER_0_HOST \
    --node_rank=$ARNOLD_ID \
    --master_port=$PORT \
    --rdzv_conf=read_timeout=$TORCHRUN_RDZV_READ_TIMEOUT \
tools/save_dataset_features.py \
--pn 0.90M \
--video_data_path=${video_data_path} \
--video_frames 81 \
--video_fps 16 \
--vae_type=64 \
--videovae=10 \
--apply_spatial_patchify 1 \
--image_batch_size=16 \
--video_batch_size=1 \
--use_slice 1 \
--dataloader_workers=16 \
--vae_path=${vae_path} \
--dynamic_scale_schedule infinity_star_extract_features \
--token_cache_dir=${token_cache_dir} \
--video_caption_type='tarsier2_caption' \
--only_images4extract_feats 0 \
--train_max_token_len=-1 \
--drop_long_video=0 \
--min_video_frames=-1 \
--cache_check_mode=-2 \
--restrict_data_size=-1 \
--use_feat_proj=2 \
--use_two_stage_lfq=1 \
--seed=1452 \


