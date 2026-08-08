# Preparing and Training with Video Metadata

This guide walks you through preparing your video metadata, splitting it for efficient training, and running the training scripts.

## 1. Prepare Your Data in `.jsonl` Format

Your video metadata should be organized in JSON Lines (`.jsonl`) format, where each line is a valid JSON object representing one video.

**Example:**
```json
{
  "video_path": "data/infinitystar_toy_data/videos/e06b8ca5dbc6.mp4",
  "begin_frame_id": 0,
  "end_frame_id": 120,
  "tarsier2_caption": "The video features an animated character with long light orange hair and brown eyes.",
  "width": 1280,
  "height": 720,
  "h_div_w": 0.5625,
  "fps": 24
}
```

## 2. Split Metadata for Training

For efficient training, large `.jsonl` files can be split into smaller chunks.

```bash
python3 data/infinitystar_toy_data/split_jsonls_for_training.py --jsonl_folder_list JSONL_DIR --save_dir SAVE_DIR --chunk_size 100
```

## 3. Extract Video Features

To extract video features, modify the `extract_video_features.sh` script. Set the `video_data_path` and choose the desired resolution.

*   **480p (5s):** `pn=0.40M`
*   **480p (10s):** `pn=0.40M` with `video_frames=161`
*   **720p (5s):** `pn=0.90M`

Then, run the script:
```bash
bash scripts/extract_video_features.sh
```

## 4. Run Training Scripts

Once your metadata is prepared and features are extracted, you can start training.

**480p Training (5s or 10s):**
```bash
bash scripts/train_480p.sh
```

**720p Training (only 5s):**
```bash
bash scripts/train_720p.sh
```
The 480p configuration supports both 5-second and 10-second video training. For 10-second training, ensure that `video_frames` is set to `161` in `extract_video_features.sh` and `train_480p.sh`.