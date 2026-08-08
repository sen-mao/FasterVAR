# InfinityStar

This release package contains the original InfinityStar inference code and the InfinityStar+FasterVAR code using:

- CFG=0 suffix acceleration on the last 3 scales.
- Final-scale random projection with rank ratio 0.1.

The original InfinityStar behavior is preserved by default when
`--cfg0_last_scales 0` is used and `--rp_last_scale` is not set.

## Package Contents

- `infinity/`: model code, including the original InfinityStar path and the
  FasterVAR hooks in `infinity/models/infinity.py`.
- `tools/infer_video_480p.py`, `tools/infer_video_720p.py`,
  `tools/infer_interact_480p.py`: original InfinityStar inference scripts.
- `tools/fastervar_cfg0_eval.py`: single-prompt baseline-vs-FasterVAR
  inference and timing script.
- `tools/vbench_generate_infinitystar.py`: full VBench video generation script
  for both original InfinityStar and FasterVAR.
- `tools/vbench_score_summary.py`: VBench Quality/Semantic/Total summarizer.
- `tools/run_vbench_original_full_eval.sh`: VBench evaluation helper.
- `evaluation/VBench_rewrited_prompt.json`: rewritten VBench prompt metadata.
- `assets/release_videos/`: generated comparison videos and single-prompt
  timing artifacts used in this README.

Model checkpoints are not included in this source package. Put them under:

```text
checkpoints/infinitystar_8b_480p_weights/
checkpoints/infinitystar_videovae.pth
checkpoints/text_encoder/flan-t5-xl-official/
```

## Environment

The experiments below were run from the repository root with the `infinty`
Conda environment:

```bash
conda run -n infinty python -m py_compile \
  infinity/models/infinity.py \
  tools/fastervar_cfg0_eval.py \
  tools/vbench_generate_infinitystar.py \
  tools/vbench_score_summary.py
```

## Inference

### InfinityStar

The upstream inference scripts remain available:

```bash
conda run -n infinty python tools/infer_video_480p.py
conda run -n infinty python tools/infer_video_720p.py
```

For VBench-standard generation, disable FasterVAR features explicitly:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
conda run -n infinty python tools/vbench_generate_infinitystar.py \
  --checkpoints_dir checkpoints \
  --resolution 480p \
  --samples_per_prompt 5 \
  --cfg 34 \
  --cfg0_last_scales 0 \
  --output_dir output/vbench_original_full/videos
```

### InfinityStar + FasterVAR

Enable CFG=0 suffix acceleration and final-scale random projection:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
conda run -n infinty python tools/vbench_generate_infinitystar.py \
  --checkpoints_dir checkpoints \
  --resolution 480p \
  --samples_per_prompt 5 \
  --cfg 34 \
  --cfg0_last_scales 3 \
  --rp_last_scale \
  --rp_rank_ratio 0.1 \
  --output_dir output/vbench_fastervar_full/videos
```

<!-- For a one-prompt timing comparison that generates both original and FasterVAR
videos:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
conda run -n infinty python tools/fastervar_cfg0_eval.py \
  --checkpoints_dir checkpoints \
  --resolution 480p \
  --cfg 34 \
  --cfg0_last_scales 3 \
  --rp_last_scale \
  --rp_rank_ratio 0.1 \
  --output_dir output/fastervar_cfg0_rp_eval
``` -->

<!-- The timing prompt was:

```text
A handsome smiling gardener inspecting plants, realistic cinematic lighting, detailed textures, ultra-realistic
``` -->

## 480P Runtime

All runtime numbers below use 480p, 5 seconds, CFG=34, APG guidance.

| Method | Time (s) | Speedup |
|---|---:|---:|
| InfinityStar | 60.8062 | 1.0000x |
| InfinityStar + FasterVAR | 33.6726 | 1.8058x |

## 480P Video Comparison

The two videos below are generated from the same prompt and seed.

| Original InfinityStar                                                                                       | InfinityStar + FasterVAR  |
|-------------------------------------------------------------------------------------------------------------|---|
| <video src="assets/release_videos/infinitystar480p_original_sample.gif" controls muted width="360"></video> | <video src="assets/release_videos/infinitystar480p_fastervar_sample.gif" controls muted width="360"></video> |
| [Download original sample](assets/release_videos/infinitystar480p_original_sample.mp4)                      | [Download FasterVAR sample](assets/release_videos/infinitystar480p_fastervar_sample.mp4) |

## 480P Full VBench Results

Full VBench standard evaluation used 946 prompt entries with 5 samples per
entry. The record-level video set was complete with 4730 / 4730 expected
samples.

| Method | Quality | Semantic | Total |
|---|---:|---:|---:|
| Original InfinityStar | 84.0318 | 83.2993 | 83.8853 |
| InfinityStar + FasterVAR CFG=0 + RP | 83.4253 | 82.9305 | 83.3263 |

## 480P Detailed VBench Dimension Scores

Raw dimension scores are shown below.

| Dimension | Original | FasterVAR | Delta |
|---|---:|---:|---:|
| subject consistency | 0.939354 | 0.935616 | -0.003738 |
| background consistency | 0.957590 | 0.955354 | -0.002236 |
| temporal flickering | 0.980129 | 0.980860 | +0.000731 |
| motion smoothness | 0.982615 | 0.983001 | +0.000386 |
| dynamic degree | 0.747222 | 0.727778 | -0.019444 |
| aesthetic quality | 0.658747 | 0.654360 | -0.004387 |
| imaging quality | 0.662832 | 0.641633 | -0.021199 |
| object class | 0.972627 | 0.964873 | -0.007753 |
| multiple objects | 0.880030 | 0.859604 | -0.020427 |
| human action | 0.980000 | 0.980000 | +0.000000 |
| color | 0.879746 | 0.880816 | +0.001070 |
| spatial relationship | 0.865726 | 0.867064 | +0.001338 |
| scene | 0.555523 | 0.556977 | +0.001453 |
| appearance style | 0.220971 | 0.218925 | -0.002046 |
| temporal style | 0.257838 | 0.257401 | -0.000437 |
| overall consistency | 0.277200 | 0.276911 | -0.000290 |

## 720p Runtime

All runtime numbers below use 480p, 5 seconds, CFG=34, APG guidance.

| Method | Time (s) | Speedup |
|---|---:|---:|
| InfinityStar | 66.63 | 1.0x |
| InfinityStar + FasterVAR | 24.65 | 2.70x |

<!-- Runtime is reported in seconds per 5-second video. End-to-end time includes text
encoding, Infinity inference, decoding, and video saving, but excludes model
loading, queueing, and idle time between jobs.

| Metric | Original InfinityStar | FasterVAR | Speedup |
|---|---:|---:|---:|
| End-to-end mean time (s/video) | 99.64 | 32.43 | 3.07x |
| End-to-end P50 time (s/video) | 111.78 | 32.35 | 3.46x |
| End-to-end P95 time (s/video) | 133.82 | 34.92 | 3.83x |
| Per-worker equivalent throughput (videos/hour) | 36.13 | 111.00 | 3.07x |

The core Infinity timing comes from the generation log field `infinity cost`,
which starts at autoregressive inference and excludes text encoding and video
file saving.

| Metric | Original InfinityStar | FasterVAR | Speedup |
|---|---:|---:|---:|
| Core Infinity inference mean time (s/video) | 66.63 | 24.65 | 2.70x |
| Core timing sample count | 1,585 | 4,720 | - | -->

## 720P Video Comparison

The two videos below are generated from the same prompt and seed.

| Original InfinityStar | InfinityStar + FasterVAR  |
|---|---|
| <video src="assets/release_videos/infinitystar720p_original_sample.gif" controls muted width="360"></video> | <video src="assets/release_videos/infinitystar720p_fastervar_sample.gif" controls muted width="360"></video> |
| [Download original sample](assets/release_videos/infinitystar720p_original_sample.mp4) | [Download FasterVAR sample](assets/release_videos/infinitystar720p_fastervar_sample.mp4) |

## 720P Full VBench Results

The 720p full VBench standard evaluation used the same 946 prompt entries with
5 samples per entry. Both runs covered all 16 VBench dimensions with no missing
dimensions. The accelerated run used CFG=0 on the last 3 scales and final-scale
random projection.

| Method | Quality | Semantic | Total |
|---|---:|---:|---:|
| Original InfinityStar | 84.0843 | 82.7488 | 83.8172 |
| InfinityStar + FasterVAR CFG=0 + RP | 83.1464 | 82.4298 | 83.0030 |

## 720P Detailed VBench Dimension Scores

Raw dimension scores are shown below.

| Dimension | Original | FasterVAR | Delta |
|---|---:|---:|---:|
| subject consistency | 0.951728 | 0.945275 | -0.006453 |
| background consistency | 0.955747 | 0.955269 | -0.000478 |
| temporal flickering | 0.984486 | 0.985253 | +0.000767 |
| motion smoothness | 0.987482 | 0.987364 | -0.000118 |
| dynamic degree | 0.680556 | 0.613889 | -0.066667 |
| aesthetic quality | 0.642706 | 0.635532 | -0.007174 |
| imaging quality | 0.675178 | 0.661258 | -0.013921 |
| object class | 0.974684 | 0.965032 | -0.009652 |
| multiple objects | 0.880793 | 0.857774 | -0.023018 |
| human action | 0.982000 | 0.982000 | +0.000000 |
| color | 0.912739 | 0.918361 | +0.005622 |
| spatial relationship | 0.864757 | 0.860887 | -0.003869 |
| scene | 0.530378 | 0.531831 | +0.001453 |
| appearance style | 0.207576 | 0.207479 | -0.000097 |
| temporal style | 0.255941 | 0.256524 | +0.000583 |
| overall consistency | 0.275916 | 0.275619 | -0.000297 |

