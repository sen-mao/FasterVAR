# Copyright (c) 2026
# SPDX-License-Identifier: MIT
"""Compute VBench Quality/Semantic/Total scores from eval_results JSON files."""

import argparse
import csv
import json
from pathlib import Path

TASK_INFO = [
    "subject consistency", "background consistency", "temporal flickering", "motion smoothness",
    "dynamic degree", "aesthetic quality", "imaging quality", "object class", "multiple objects",
    "human action", "color", "spatial relationship", "scene", "appearance style",
    "temporal style", "overall consistency",
]
DIM_WEIGHT = {
    "subject consistency": 1,
    "background consistency": 1,
    "temporal flickering": 1,
    "motion smoothness": 1,
    "aesthetic quality": 1,
    "imaging quality": 1,
    "dynamic degree": 0.5,
    "object class": 1,
    "multiple objects": 1,
    "human action": 1,
    "color": 1,
    "spatial relationship": 1,
    "scene": 1,
    "appearance style": 1,
    "temporal style": 1,
    "overall consistency": 1,
}
NORMALIZE_DIC = {
    "subject consistency": {"Min": 0.1462, "Max": 1.0},
    "background consistency": {"Min": 0.2615, "Max": 1.0},
    "temporal flickering": {"Min": 0.6293, "Max": 1.0},
    "motion smoothness": {"Min": 0.706, "Max": 0.9975},
    "dynamic degree": {"Min": 0.0, "Max": 1.0},
    "aesthetic quality": {"Min": 0.0, "Max": 1.0},
    "imaging quality": {"Min": 0.0, "Max": 1.0},
    "object class": {"Min": 0.0, "Max": 1.0},
    "multiple objects": {"Min": 0.0, "Max": 1.0},
    "human action": {"Min": 0.0, "Max": 1.0},
    "color": {"Min": 0.0, "Max": 1.0},
    "spatial relationship": {"Min": 0.0, "Max": 1.0},
    "scene": {"Min": 0.0, "Max": 0.8222},
    "appearance style": {"Min": 0.0009, "Max": 0.2855},
    "temporal style": {"Min": 0.0, "Max": 0.364},
    "overall consistency": {"Min": 0.0, "Max": 0.364},
}
QUALITY_LIST = [
    "subject consistency", "background consistency", "temporal flickering", "motion smoothness",
    "aesthetic quality", "imaging quality", "dynamic degree",
]
SEMANTIC_LIST = [
    "object class", "multiple objects", "human action", "color", "spatial relationship",
    "scene", "appearance style", "temporal style", "overall consistency",
]
QUALITY_WEIGHT = 4
SEMANTIC_WEIGHT = 1


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize full VBench scores.")
    parser.add_argument("--eval_results", required=True, help="Path to VBench *_eval_results.json")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--scale100", action="store_true", help="Also report scores multiplied by 100.")
    return parser.parse_args()


def load_dimension_scores(path):
    raw = json.loads(Path(path).read_text())
    scores = {}
    for key, value in raw.items():
        dim = key.replace("_", " ")
        if isinstance(value, list):
            scores[dim] = float(value[0])
        else:
            scores[dim] = float(value)
    missing = [dim for dim in TASK_INFO if dim not in scores]
    return raw, scores, missing


def normalize(scores):
    normalized = {}
    for dim in TASK_INFO:
        raw = float(scores.get(dim, 0.0))
        mn = NORMALIZE_DIC[dim]["Min"]
        mx = NORMALIZE_DIC[dim]["Max"]
        normalized[dim] = ((raw - mn) / (mx - mn)) * DIM_WEIGHT[dim]
    return normalized


def weighted_mean(normalized, dims):
    return sum(normalized[dim] for dim in dims) / sum(DIM_WEIGHT[dim] for dim in dims)


def main():
    cli = parse_args()
    raw, scores, missing = load_dimension_scores(cli.eval_results)
    normalized = normalize(scores)
    quality = weighted_mean(normalized, QUALITY_LIST)
    semantic = weighted_mean(normalized, SEMANTIC_LIST)
    total = (quality * QUALITY_WEIGHT + semantic * SEMANTIC_WEIGHT) / (QUALITY_WEIGHT + SEMANTIC_WEIGHT)

    output_dir = Path(cli.output_dir) if cli.output_dir else Path(cli.eval_results).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "eval_results": cli.eval_results,
        "missing_dimensions": missing,
        "raw_dimension_scores": {dim: scores.get(dim, 0.0) for dim in TASK_INFO},
        "normalized_dimension_scores": normalized,
        "quality_score": quality,
        "semantic_score": semantic,
        "total_score": total,
    }
    if cli.scale100:
        report["quality_score_100"] = quality * 100
        report["semantic_score_100"] = semantic * 100
        report["total_score_100"] = total * 100
    (output_dir / "vbench_score_report.json").write_text(json.dumps(report, indent=2))

    with (output_dir / "vbench_score_summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "score", "score_100"])
        writer.writerow(["quality_score", quality, quality * 100])
        writer.writerow(["semantic_score", semantic, semantic * 100])
        writer.writerow(["total_score", total, total * 100])

    with (output_dir / "vbench_dimension_scores.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dimension", "raw_score", "normalized_weighted_score"])
        for dim in TASK_INFO:
            writer.writerow([dim, scores.get(dim, 0.0), normalized[dim]])

    print(json.dumps({
        "quality_score": quality,
        "semantic_score": semantic,
        "total_score": total,
        "quality_score_100": quality * 100,
        "semantic_score_100": semantic * 100,
        "total_score_100": total * 100,
        "missing_dimensions": missing,
        "output_dir": str(output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
