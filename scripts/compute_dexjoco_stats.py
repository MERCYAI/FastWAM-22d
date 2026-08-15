#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from fastwam.datasets.lerobot.dexjoco_stats import compute_dexjoco_statistics_from_root
from fastwam.datasets.lerobot.utils.normalizer import save_dataset_stats_to_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute FastWAM normalization statistics from DexJoCo training episodes."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train"], default="train")
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument(
        "--max-episodes-per-task",
        type=int,
        default=None,
        help="Limit each task for a non-production smoke run. Omit for production statistics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = compute_dexjoco_statistics_from_root(
        args.data_root,
        split=args.split,
        action_horizon=args.action_horizon,
        max_episodes_per_task=args.max_episodes_per_task,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_dataset_stats_to_json(stats, str(args.output))
    print(f"wrote={args.output.resolve()}")
    print(
        f"schema={stats['schema_name']}@{stats['schema_version']} "
        f"mode={stats['statistics_mode']} production={stats['production']} split={stats['split']}"
    )
    print(
        f"tasks={','.join(stats['tasks'])} episodes={stats['num_episodes']} "
        f"frames={stats['num_transition']} action_dim={stats['action_dim']} "
        f"proprio_dim={stats['proprio_dim']}"
    )


if __name__ == "__main__":
    main()
