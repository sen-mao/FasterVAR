# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

import argparse
import csv
import json
import math
import os
import os.path as osp
import sys
import time
from pathlib import Path

sys.path.append(osp.dirname(osp.dirname(__file__)))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def load_runtime_deps():
    global cv2, np, torch, Image
    global gen_one_example, load_tokenizer, load_transformer, load_visual_tokenizer, save_video, transform
    global SelfCorrection, get_encode_decode_func, get_dynamic_resolution_meta, get_first_full_spatial_size_scale_index, Args

    import cv2
    import numpy as np
    import torch
    from PIL import Image

    from tools.run_infinity import (
        gen_one_example,
        load_tokenizer,
        load_transformer,
        load_visual_tokenizer,
        save_video,
        transform,
    )
    from infinity.models.self_correction import SelfCorrection
    from infinity.schedules import get_encode_decode_func
    from infinity.schedules.dynamic_resolution import (
        get_dynamic_resolution_meta,
        get_first_full_spatial_size_scale_index,
    )
    from infinity.utils.arg_util import Args


DEFAULT_PROMPT = (
    "A handsome smiling gardener inspecting plants, realistic cinematic lighting, "
    "detailed textures, ultra-realistic"
)


class InferencePipe:
    def __init__(self, args):
        self.text_tokenizer, self.text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
        self.vae = load_visual_tokenizer(args).float().to("cuda")
        self.infinity = load_transformer(self.vae, args)
        self.self_correction = SelfCorrection(self.vae, args)
        funcs = get_encode_decode_func(args.dynamic_scale_schedule)
        self.video_encode, self.video_decode, self.get_visual_rope_embeds, self.get_scale_pack_info = funcs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate InfinityStar baseline generation against FasterVAR-style CFG=0 suffix acceleration."
    )
    parser.add_argument("--resolution", choices=["480p", "720p"], default="480p")
    parser.add_argument("--checkpoints_dir", default=".")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--vae_path", default=None)
    parser.add_argument("--text_encoder_ckpt", default=None)
    parser.add_argument("--checkpoint_type", default="torch_shard", choices=["torch", "torch_shard", "omnistore"])
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompts", default=None, help="Optional .txt, .json, or .jsonl prompt file.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of prompts; 0 means no limit.")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--duration", type=int, default=5, choices=[5, 10])
    parser.add_argument("--image_path", default=None, help="Optional reference image for I2V evaluation.")
    parser.add_argument("--h_div_w", type=float, default=0.571)
    parser.add_argument("--guidance", choices=["apg", "cfg"], default="apg")
    parser.add_argument("--cfg", type=float, default=34.0)
    parser.add_argument("--cfg0_last_scales", type=int, default=3)
    parser.add_argument("--rp_last_scale", action="store_true", help="Apply FasterVAR random projection on the final scale only.")
    parser.add_argument("--rp_rank_ratio", type=float, default=0.25, help="Final-scale random projection rank as a fraction of token count.")
    parser.add_argument("--rp_rank", type=int, default=0, help="Absolute final-scale random projection rank; overrides --rp_rank_ratio when > 0.")
    parser.add_argument("--rp_seed", type=int, default=-1, help="Random projection seed; <0 derives it from sample seed.")
    parser.add_argument("--tau_image", type=float, default=1.0)
    parser.add_argument("--tau_video", type=float, default=0.4)
    parser.add_argument("--apg_norm_threshold", type=float, default=0.05)
    parser.add_argument("--max_repeat_times", type=int, default=10000)
    parser.add_argument("--output_dir", default="output/fastervar_cfg0_eval")
    parser.add_argument("--no_save_videos", action="store_true")
    parser.add_argument("--no_detail_suffix", action="store_true")
    parser.add_argument("--ssim_stride", type=int, default=8)
    return parser.parse_args()


def resolve_required_paths(cli):
    if cli.resolution == "480p":
        default_model = osp.join(cli.checkpoints_dir, "infinitystar_8b_480p_weights")
    else:
        default_model = osp.join(cli.checkpoints_dir, "infinitystar_8b_720p_weights")
    return {
        "model_path": cli.model_path or default_model,
        "vae_path": cli.vae_path or osp.join(cli.checkpoints_dir, "infinitystar_videovae.pth"),
        "text_encoder_ckpt": cli.text_encoder_ckpt or osp.join(cli.checkpoints_dir, "text_encoder/flan-t5-xl-official/"),
    }


def check_paths(paths):
    missing = [f"{label}={path}" for label, path in paths.items() if not osp.exists(path)]
    if missing:
        raise FileNotFoundError("Missing required checkpoint paths: " + "; ".join(missing))


def build_infinity_args(cli):
    args = Args()
    args.fps = 16
    args.video_frames = cli.duration * 16 + 1
    args.checkpoint_type = cli.checkpoint_type
    args.vae_path = cli.vae_path or osp.join(cli.checkpoints_dir, "infinitystar_videovae.pth")
    args.text_encoder_ckpt = cli.text_encoder_ckpt or osp.join(cli.checkpoints_dir, "text_encoder/flan-t5-xl-official/")
    args.videovae = 10
    args.model_type = "infinity_qwen8b"
    args.text_channels = 2048
    args.dynamic_scale_schedule = "infinity_elegant_clip20frames_v2"
    args.bf16 = 1
    args.use_apg = 1 if cli.guidance == "apg" else 0
    args.use_cfg = 1 if cli.guidance == "cfg" else 0
    args.tau_image = cli.tau_image
    args.tau_video = cli.tau_video
    args.apg_norm_threshold = cli.apg_norm_threshold
    args.append_duration2caption = 1
    args.use_two_stage_lfq = 1
    args.max_repeat_times = cli.max_repeat_times
    args.enable_rewriter = 0
    args.checkpoint_type = cli.checkpoint_type
    args.cfg = cli.cfg
    args.fastervar_rp_last_scale = 1 if cli.rp_last_scale else 0
    args.fastervar_rp_rank_ratio = cli.rp_rank_ratio
    args.fastervar_rp_rank = cli.rp_rank
    args.fastervar_rp_seed = cli.rp_seed

    if cli.resolution == "480p":
        args.pn = "0.40M"
        args.model_path = cli.model_path or osp.join(cli.checkpoints_dir, "infinitystar_8b_480p_weights")
        args.image_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]"
        args.video_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1]"
        args.detail_scale_min_tokens = 350
        args.semantic_scales = 11
    else:
        if cli.duration != 5:
            raise ValueError("The released 720p script supports 5 seconds only.")
        args.pn = "0.90M"
        args.model_path = cli.model_path or osp.join(cli.checkpoints_dir, "infinitystar_8b_720p_weights")
        args.image_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]"
        args.video_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 1]"
        args.detail_scale_min_tokens = 750
        args.semantic_scales = 12
    return args


def load_prompt_records(cli):
    if cli.prompts is None:
        return [{"prompt": cli.prompt, "seed": cli.seed, "image_path": cli.image_path}]

    path = Path(cli.prompts)
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    elif path.suffix == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            data = data.get("prompts", data.get("data", [data]))
        records = data
    else:
        records = [{"prompt": line.strip()} for line in path.read_text().splitlines() if line.strip()]

    normalized = []
    for i, item in enumerate(records):
        if isinstance(item, str):
            item = {"prompt": item}
        normalized.append({
            "prompt": item["prompt"],
            "seed": int(item.get("seed", cli.seed + i)),
            "image_path": item.get("image_path", cli.image_path),
        })
    if cli.limit > 0:
        normalized = normalized[:cli.limit]
    return normalized


def build_scale_info(pipe, args, cli):
    dynamic_resolution_h_w, h_div_w_templates = get_dynamic_resolution_meta(
        args.dynamic_scale_schedule, args.video_frames
    )
    h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - cli.h_div_w))]
    num_frames = cli.duration * args.fps + 1
    compressed_frames = (num_frames - 1) // 4 + 1
    scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]["pt2scale_schedule"][compressed_frames]
    args.first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
    args.tower_split_index = args.first_full_spatial_size_scale_index + 1
    context_info = pipe.get_scale_pack_info(scale_schedule, args.first_full_spatial_size_scale_index, args)
    tau = [args.tau_image] * args.tower_split_index + [args.tau_video] * (len(scale_schedule) - args.tower_split_index)
    target_h, target_w = scale_schedule[-1][1] * 16, scale_schedule[-1][2] * 16
    return {
        "dynamic_resolution_h_w": dynamic_resolution_h_w,
        "h_div_w_template": float(h_div_w_template),
        "scale_schedule": scale_schedule,
        "context_info": context_info,
        "tau": tau,
        "target_h": target_h,
        "target_w": target_w,
    }


def make_cfg_schedule(base_cfg, scale_count, cfg0_last_scales):
    if cfg0_last_scales < 0 or cfg0_last_scales > scale_count:
        raise ValueError(f"cfg0_last_scales must be in [0, {scale_count}], got {cfg0_last_scales}")
    cfg = [float(base_cfg)] * scale_count
    if cfg0_last_scales:
        cfg[-cfg0_last_scales:] = [0.0] * cfg0_last_scales
    return cfg


def prepare_condition(pipe, args, record, scale_info):
    image_path = record.get("image_path")
    if not image_path:
        return -1, None
    ref_image = cv2.imread(image_path)
    if ref_image is None:
        raise FileNotFoundError(f"Could not read image_path={image_path}")
    ref_image = [ref_image[:, :, ::-1]]
    ref_img_t3hw = [
        transform(Image.fromarray(frame).convert("RGB"), scale_info["target_h"], scale_info["target_w"])
        for frame in ref_image
    ]
    ref_img_t3hw = torch.stack(ref_img_t3hw, 0)
    ref_img_bcthw = ref_img_t3hw.permute(1, 0, 2, 3).unsqueeze(0)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
        _, _, gt_ls_bl, _, _, _ = pipe.video_encode(
            pipe.vae,
            ref_img_bcthw.cuda(),
            vae_features=None,
            self_correction=pipe.self_correction,
            args=args,
            infer_mode=True,
            dynamic_resolution_h_w=scale_info["dynamic_resolution_h_w"],
        )
    return len(scale_info["scale_schedule"]) // 2, gt_ls_bl


def normalize_prompt(prompt, args, cli):
    if not cli.no_detail_suffix:
        prompt = f"{prompt}, Close-up on big objects, emphasize scale and detail"
    if args.append_duration2caption:
        prompt = f"<<<t={cli.duration}s>>>" + prompt
    return prompt


def run_generation(pipe, args, record, scale_info, cfg_schedule, cli, gt_leak, gt_ls_bl):
    prompt = normalize_prompt(record["prompt"], args, cli)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_event = end_event = None

    start = time.perf_counter()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
        video, _ = gen_one_example(
            pipe.infinity,
            pipe.vae,
            pipe.text_tokenizer,
            pipe.text_encoder,
            prompt,
            negative_prompt="",
            g_seed=int(record["seed"]),
            gt_leak=gt_leak,
            gt_ls_Bl=gt_ls_bl,
            cfg_list=cfg_schedule,
            tau_list=scale_info["tau"],
            scale_schedule=scale_info["scale_schedule"],
            cfg_insertion_layer=[0],
            vae_type=args.vae_type,
            sampling_per_bits=1,
            enable_positive_prompt=0,
            low_vram_mode=True,
            args=args,
            get_visual_rope_embeds=pipe.get_visual_rope_embeds,
            context_info=scale_info["context_info"],
            noise_list=None,
        )
    if torch.cuda.is_available():
        end_event.record()
        torch.cuda.synchronize()
        cuda_ms = start_event.elapsed_time(end_event)
        peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    else:
        cuda_ms = None
        peak_gib = None
    elapsed = time.perf_counter() - start
    if len(video.shape) == 3:
        video = video.unsqueeze(0)
    return {
        "video": video.cpu().numpy(),
        "wall_seconds": elapsed,
        "cuda_ms": cuda_ms,
        "peak_memory_gib": peak_gib,
    }


def ssim_frame(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    scores = []
    for ch in range(a.shape[-1]):
        x = a[:, :, ch]
        y = b[:, :, ch]
        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
        sigma_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
        sigma_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
        sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
            (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        )
        scores.append(float(np.mean(score)))
    return float(np.mean(scores))


def compare_videos(reference, candidate, ssim_stride):
    ref = reference.astype(np.float32)
    cand = candidate.astype(np.float32)
    if ref.shape != cand.shape:
        raise ValueError(f"Video shape mismatch: {ref.shape} != {cand.shape}")
    diff = cand - ref
    mse = float(np.mean(diff ** 2))
    rmse = math.sqrt(mse)
    psnr = float("inf") if mse == 0 else 20.0 * math.log10(255.0 / rmse)
    frame_mse = np.mean(diff ** 2, axis=(1, 2, 3))
    if len(ref) > 1:
        temporal_diff = np.diff(cand, axis=0) - np.diff(ref, axis=0)
        temporal_mse = float(np.mean(temporal_diff ** 2))
    else:
        temporal_mse = 0.0
    stride = max(1, int(ssim_stride))
    ssim_values = [ssim_frame(ref[i], cand[i]) for i in range(0, len(ref), stride)]
    return {
        "mse": mse,
        "mae": float(np.mean(np.abs(diff))),
        "rmse": rmse,
        "psnr_db": psnr,
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_frame_mse": float(np.mean(frame_mse)),
        "max_frame_mse": float(np.max(frame_mse)),
        "temporal_mse": temporal_mse,
        "ssim": float(np.mean(ssim_values)),
        "ssim_stride": stride,
    }


def write_reports(output_dir, rows, report):
    os.makedirs(output_dir, exist_ok=True)
    report_path = osp.join(output_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    csv_path = osp.join(output_dir, "summary.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return report_path, csv_path


def main():
    cli = parse_args()
    check_paths(resolve_required_paths(cli))
    load_runtime_deps()
    args = build_infinity_args(cli)
    records = load_prompt_records(cli)
    os.makedirs(cli.output_dir, exist_ok=True)

    pipe = InferencePipe(args)
    scale_info = build_scale_info(pipe, args, cli)
    scale_count = len(scale_info["scale_schedule"])
    baseline_cfg = [float(cli.cfg)] * scale_count
    cfg0_cfg = make_cfg_schedule(cli.cfg, scale_count, cli.cfg0_last_scales)

    rows = []
    samples = []
    for i, record in enumerate(records):
        gt_leak, gt_ls_bl = prepare_condition(pipe, args, record, scale_info)
        args.fastervar_rp_last_scale = 0
        baseline = run_generation(pipe, args, record, scale_info, baseline_cfg, cli, gt_leak, gt_ls_bl)
        args.fastervar_rp_last_scale = 1 if cli.rp_last_scale else 0
        accelerated = run_generation(pipe, args, record, scale_info, cfg0_cfg, cli, gt_leak, gt_ls_bl)
        metrics = compare_videos(baseline["video"], accelerated["video"], cli.ssim_stride)
        speedup = baseline["wall_seconds"] / accelerated["wall_seconds"] if accelerated["wall_seconds"] else float("inf")
        memory_delta = None
        if baseline["peak_memory_gib"] is not None and accelerated["peak_memory_gib"] is not None:
            memory_delta = accelerated["peak_memory_gib"] - baseline["peak_memory_gib"]

        sample_id = f"sample_{i:04d}"
        if not cli.no_save_videos:
            save_video(baseline["video"], fps=args.fps, save_filepath=osp.join(cli.output_dir, f"{sample_id}_baseline.mp4"))
            save_video(accelerated["video"], fps=args.fps, save_filepath=osp.join(cli.output_dir, f"{sample_id}_cfg0.mp4"))

        row = {
            "sample_id": sample_id,
            "seed": int(record["seed"]),
            "baseline_seconds": baseline["wall_seconds"],
            "cfg0_seconds": accelerated["wall_seconds"],
            "speedup": speedup,
            "baseline_peak_gib": baseline["peak_memory_gib"],
            "cfg0_peak_gib": accelerated["peak_memory_gib"],
            "peak_gib_delta": memory_delta,
            "mse": metrics["mse"],
            "mae": metrics["mae"],
            "psnr_db": metrics["psnr_db"],
            "ssim": metrics["ssim"],
            "temporal_mse": metrics["temporal_mse"],
        }
        rows.append(row)
        samples.append({
            "sample_id": sample_id,
            "prompt": record["prompt"],
            "seed": int(record["seed"]),
            "image_path": record.get("image_path"),
            "baseline": {k: v for k, v in baseline.items() if k != "video"},
            "cfg0": {k: v for k, v in accelerated.items() if k != "video"},
            "metrics": metrics,
        })
        print(json.dumps(row, indent=2))

    mean_row = {}
    if rows:
        numeric_keys = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        mean_row = {k: float(np.mean([row[k] for row in rows if row[k] is not None])) for k in numeric_keys}

    report = {
        "config": {
            "resolution": cli.resolution,
            "duration": cli.duration,
            "guidance": cli.guidance,
            "base_cfg": cli.cfg,
            "cfg0_last_scales": cli.cfg0_last_scales,
            "rp_last_scale": cli.rp_last_scale,
            "rp_rank_ratio": cli.rp_rank_ratio,
            "rp_rank": cli.rp_rank,
            "rp_seed": cli.rp_seed,
            "baseline_cfg": baseline_cfg,
            "cfg0_cfg": cfg0_cfg,
            "scale_schedule": scale_info["scale_schedule"],
            "h_div_w_template": scale_info["h_div_w_template"],
            "model_path": args.model_path,
            "vae_path": args.vae_path,
            "text_encoder_ckpt": args.text_encoder_ckpt,
        },
        "summary_mean": mean_row,
        "samples": samples,
    }
    report_path, csv_path = write_reports(cli.output_dir, rows, report)
    print(f"Wrote {report_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
