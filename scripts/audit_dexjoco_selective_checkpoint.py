from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.dexjoco_checkpoint import load_dexjoco_selective_checkpoint
from fastwam.models.wan22.wan_video_dit import WanVideoDiT
from fastwam.utils.logging_config import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit an old FastWAM checkpoint against the full DexJoCo model "
            "configuration without allocating model weights or starting training."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def build_meta_audit_target():
    config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "model=dexjoco_dual_action",
                "data=dexjoco_6task_2cam",
            ],
        )
    video_config = OmegaConf.to_container(cfg.model.video_dit_config, resolve=True)
    action_config = OmegaConf.to_container(cfg.model.action_dit_config, resolve=True)
    hand_config = OmegaConf.to_container(cfg.model.hand_dit_config, resolve=True)

    with torch.device("meta"):
        video_expert = WanVideoDiT(**video_config)
        action_expert = ActionDiT(**action_config)
        hand_expert = ActionDiT(**hand_config)
        proprio_encoder = nn.Linear(int(cfg.model.proprio_dim), int(video_config["text_dim"]))

    return SimpleNamespace(
        video_expert=video_expert,
        action_expert=action_expert,
        hand_expert=hand_expert,
        proprio_encoder=proprio_encoder,
    )


def main() -> None:
    args = parse_args()
    setup_logging()
    target = build_meta_audit_target()
    report = load_dexjoco_selective_checkpoint(
        target,
        args.checkpoint,
        report_path=args.report,
        apply=False,
    )
    print(f"report={args.report.expanduser().resolve()}")
    print(f"classification counts={report.summary}")
    print("apply=false training_started=false")


if __name__ == "__main__":
    main()
