#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/share/data/drive_5/senmao/FasterVAR2/InfinityStar"
VIDEO_DIR="$REPO_DIR/output/vbench_original_full/videos"
EVAL_DIR="$REPO_DIR/output/vbench_original_full/eval"
LOG_DIR="$REPO_DIR/output/vbench_original_full/logs"
EXPECTED_VIDEOS=4730
EVAL_GPUS="${EVAL_GPUS:-4,5}"
IFS="," read -r -a EVAL_GPU_LIST <<< "$EVAL_GPUS"
NGPUS="${#EVAL_GPU_LIST[@]}"

mkdir -p "$EVAL_DIR" "$LOG_DIR"
cd "$REPO_DIR"

generation_running() {
  tmux list-sessions -F '#S' 2>/dev/null | grep -Eq '^vbench_orig_r[0-9]'
}

echo "[$(date '+%F %T')] waiting for generation sessions..."
while generation_running; do
  count=$(find "$VIDEO_DIR" -maxdepth 1 -name '*.mp4' | wc -l)
  echo "[$(date '+%F %T')] generated videos: $count / $EXPECTED_VIDEOS"
  sleep 600
done

count=$(find "$VIDEO_DIR" -maxdepth 1 -name '*.mp4' | wc -l)
echo "[$(date '+%F %T')] generation sessions finished; videos: $count / $EXPECTED_VIDEOS"
if [[ "$count" -lt "$EXPECTED_VIDEOS" ]]; then
  echo "[$(date '+%F %T')] not running VBench because the full video set is incomplete" >&2
  exit 2
fi

DIMENSIONS=(
  subject_consistency
  background_consistency
  temporal_flickering
  motion_smoothness
  dynamic_degree
  aesthetic_quality
  imaging_quality
  object_class
  multiple_objects
  human_action
  color
  spatial_relationship
  scene
  temporal_style
  appearance_style
  overall_consistency
)

echo "[$(date '+%F %T')] starting VBench full evaluation on GPUs $EVAL_GPUS"
CUDA_VISIBLE_DEVICES="$EVAL_GPUS" conda run -n infinty vbench evaluate   --ngpus "$NGPUS"   --mode vbench_standard   --videos_path "$VIDEO_DIR"   --dimension "${DIMENSIONS[@]}"   --output_path "$EVAL_DIR"

latest=$(ls -t "$EVAL_DIR"/*_eval_results.json | head -n 1)
echo "[$(date '+%F %T')] summarizing scores from $latest"
conda run -n infinty python tools/vbench_score_summary.py   --eval_results "$latest"   --output_dir "$EVAL_DIR"   --scale100

echo "[$(date '+%F %T')] done"
