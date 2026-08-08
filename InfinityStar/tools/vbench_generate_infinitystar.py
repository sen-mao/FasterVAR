# Copyright (c) 2026
# SPDX-License-Identifier: MIT
"""Generate original InfinityStar videos for the VBench standard prompt suite.

This script is intentionally generation-only. Run multiple processes with
--rank/--world_size and different CUDA_VISIBLE_DEVICES values to shard the work.
Videos are saved with VBench's expected file names: <prompt_en>-<sample_idx>.mp4.
"""

import argparse
import csv
import json
import os
import os.path as osp
import sys
import time
from pathlib import Path

REPO_ROOT = osp.abspath(osp.dirname(osp.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from tools.fastervar_cfg0_eval import (  # noqa: E402
    InferencePipe,
    build_infinity_args,
    build_scale_info,
    check_paths,
    load_runtime_deps,
    make_cfg_schedule,
    resolve_required_paths,
)


DEFAULT_PROMPT_JSON = "evaluation/VBench_rewrited_prompt.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate original InfinityStar videos for VBench.")
    parser.add_argument("--checkpoints_dir", default="checkpoints")
    parser.add_argument("--resolution", choices=["480p", "720p"], default="480p")
    parser.add_argument("--prompt_json", default=DEFAULT_PROMPT_JSON)
    parser.add_argument("--output_dir", default="output/vbench_original_full/videos")
    parser.add_argument("--samples_per_prompt", type=int, default=5)
    parser.add_argument("--sample_start", type=int, default=0)
    parser.add_argument("--seed_base", type=int, default=41)
    parser.add_argument("--duration", type=int, default=5, choices=[5, 10])
    parser.add_argument("--cfg", type=float, default=34.0)
    parser.add_argument("--cfg0_last_scales", type=int, default=0)
    parser.add_argument("--rp_last_scale", action="store_true", help="Apply random projection on the final scale only.")
    parser.add_argument("--rp_rank_ratio", type=float, default=0.0)
    parser.add_argument("--rp_rank", type=int, default=0)
    parser.add_argument("--rp_seed", type=int, default=-1)
    parser.add_argument("--guidance", choices=["apg", "cfg"], default="apg")
    parser.add_argument("--tau_image", type=float, default=1.0)
    parser.add_argument("--tau_video", type=float, default=0.4)
    parser.add_argument("--apg_norm_threshold", type=float, default=0.05)
    parser.add_argument("--max_repeat_times", type=int, default=10000)
    parser.add_argument("--h_div_w", type=float, default=0.571)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument(
        "--shard_mode",
        choices=["prompt", "missing"],
        default="prompt",
        help="prompt: shard prompt indices; missing: shard only currently missing output videos.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit prompt count after sharding; 0 means no limit.")
    parser.add_argument("--start_index", type=int, default=0, help="Global prompt start index before sharding.")
    parser.add_argument("--end_index", type=int, default=0, help="Exclusive global prompt end index before sharding; 0 means all.")
    parser.add_argument("--manifest_tag", default="", help="Optional suffix for the progress manifest file.")
    parser.add_argument("--prompt_key", choices=["refined_prompt", "prompt_en"], default="refined_prompt")
    parser.add_argument("--no_detail_suffix", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--keep_vae_decoder_on_gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the VAE decoder resident on CUDA and avoid low_vram_mode during generation.",
    )
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--vae_path", default=None)
    parser.add_argument("--text_encoder_ckpt", default=None)
    parser.add_argument("--checkpoint_type", default="torch_shard", choices=["torch", "torch_shard", "omnistore"])
    return parser.parse_args()


def safe_name(prompt, sample_idx):
    return f"{prompt}-{sample_idx}.mp4"


def load_prompts(path):
    records = json.loads(Path(path).read_text())
    if not isinstance(records, list):
        raise ValueError(f"Expected a list in {path}")
    return records


def normalize_prompt(prompt, duration, no_detail_suffix):
    if not no_detail_suffix:
        prompt = f"{prompt}, Close-up on big objects, emphasize scale and detail"
    return f"<<<t={duration}s>>>" + prompt


def keep_vae_decoder_cuda(vae, torch):
    if not torch.cuda.is_available():
        return
    vae.to("cuda")
    decoder = getattr(vae, "decoder", None)
    if decoder is not None:
        decoder.to("cuda")
    vae.eval()
    vae.requires_grad_(False)


def build_jobs(records, cli, out_dir):
    end = cli.end_index if cli.end_index > 0 else len(records)
    indexed = list(enumerate(records))[cli.start_index:end]
    all_jobs = [
        (prompt_idx, rec, cli.sample_start + sample_offset)
        for prompt_idx, rec in indexed
        for sample_offset in range(cli.samples_per_prompt)
    ]

    if cli.shard_mode == "missing":
        missing_jobs = [
            job
            for job in all_jobs
            if cli.overwrite or not (out_dir / safe_name(job[1]["prompt_en"], job[2])).exists()
        ]
        if cli.limit > 0:
            missing_jobs = missing_jobs[:cli.limit]
        jobs = [job for job_pos, job in enumerate(missing_jobs) if job_pos % cli.world_size == cli.rank]
        return jobs, len(missing_jobs)

    sharded_prompts = [(idx, rec) for idx, rec in indexed if idx % cli.world_size == cli.rank]
    if cli.limit > 0:
        sharded_prompts = sharded_prompts[:cli.limit]
    jobs = [
        (prompt_idx, rec, cli.sample_start + sample_offset)
        for prompt_idx, rec in sharded_prompts
        for sample_offset in range(cli.samples_per_prompt)
    ]
    return jobs, len(jobs)


def main():
    cli = parse_args()
    if cli.world_size < 1:
        raise ValueError("world_size must be >= 1")
    if not (0 <= cli.rank < cli.world_size):
        raise ValueError("rank must be in [0, world_size)")

    load_runtime_deps()
    from tools.fastervar_cfg0_eval import gen_one_example, save_video, torch

    check_paths(resolve_required_paths(cli))
    args = build_infinity_args(cli)

    out_dir = Path(cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_prompts(cli.prompt_json)
    jobs, total_available_jobs = build_jobs(records, cli, out_dir)
    manifest_name = f"manifest_rank{cli.rank}"
    if cli.manifest_tag:
        safe_tag = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in cli.manifest_tag)
        manifest_name = f"{manifest_name}_{safe_tag}"
    manifest_path = out_dir.parent / f"{manifest_name}.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    pipe = InferencePipe(args)
    if cli.keep_vae_decoder_on_gpu:
        keep_vae_decoder_cuda(pipe.vae, torch)
    scale_info = build_scale_info(pipe, args, cli)
    cfg_schedule = make_cfg_schedule(cli.cfg, len(scale_info["scale_schedule"]), cli.cfg0_last_scales)

    rows = []
    total_jobs = len(jobs)
    done_jobs = 0
    start_all = time.perf_counter()
    print(
        f"[rank {cli.rank}] shard_mode={cli.shard_mode}, assigned_jobs={total_jobs}, "
        f"available_jobs={total_available_jobs}, world_size={cli.world_size}, "
        f"keep_vae_decoder_on_gpu={cli.keep_vae_decoder_on_gpu}, "
        f"cfg0_last_scales={cli.cfg0_last_scales}, rp_last_scale={cli.rp_last_scale}, "
        f"rp_rank_ratio={cli.rp_rank_ratio}, rp_rank={cli.rp_rank}, rp_seed={cli.rp_seed}",
        flush=True,
    )

    for prompt_idx, rec, sample_idx in jobs:
        prompt_en = rec["prompt_en"]
        gen_prompt = rec.get(cli.prompt_key) or prompt_en
        video_path = out_dir / safe_name(prompt_en, sample_idx)
        if video_path.exists() and not cli.overwrite:
            rows.append({
                "prompt_index": prompt_idx,
                "sample_index": sample_idx,
                "seed": "",
                "prompt_en": prompt_en,
                "prompt_used": gen_prompt,
                "video_path": str(video_path),
                "status": "skipped_exists",
                "seconds": "",
            })
            done_jobs += 1
            continue

        seed = cli.seed_base + prompt_idx * max(1, cli.samples_per_prompt) + sample_idx
        prompt = normalize_prompt(gen_prompt, cli.duration, cli.no_detail_suffix)
        if torch.cuda.is_available():
            if cli.keep_vae_decoder_on_gpu:
                keep_vae_decoder_cuda(pipe.vae, torch)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
                video, _ = gen_one_example(
                    pipe.infinity,
                    pipe.vae,
                    pipe.text_tokenizer,
                    pipe.text_encoder,
                    prompt,
                    negative_prompt="",
                    g_seed=int(seed),
                    gt_leak=-1,
                    gt_ls_Bl=None,
                    cfg_list=cfg_schedule,
                    tau_list=scale_info["tau"],
                    scale_schedule=scale_info["scale_schedule"],
                    cfg_insertion_layer=[0],
                    vae_type=args.vae_type,
                    sampling_per_bits=1,
                    enable_positive_prompt=0,
                    low_vram_mode=not cli.keep_vae_decoder_on_gpu,
                    args=args,
                    get_visual_rope_embeds=pipe.get_visual_rope_embeds,
                    context_info=scale_info["context_info"],
                    noise_list=None,
                )
            if len(video.shape) == 3:
                video = video.unsqueeze(0)
            save_video(video.cpu().numpy(), fps=args.fps, save_filepath=str(video_path))
            status = "ok"
        except Exception as exc:
            status = f"error:{type(exc).__name__}:{exc}"
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        done_jobs += 1
        rows.append({
            "prompt_index": prompt_idx,
            "sample_index": sample_idx,
            "seed": seed,
            "prompt_en": prompt_en,
            "prompt_used": gen_prompt,
            "video_path": str(video_path),
            "status": status,
            "seconds": elapsed,
        })
        with manifest_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        avg = (time.perf_counter() - start_all) / max(done_jobs, 1)
        remain = total_jobs - done_jobs
        print(f"[rank {cli.rank}] {done_jobs}/{total_jobs} done, last={elapsed:.2f}s, eta={remain * avg / 3600:.2f}h, file={video_path}", flush=True)

    if rows:
        with manifest_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"[rank {cli.rank}] complete. manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
