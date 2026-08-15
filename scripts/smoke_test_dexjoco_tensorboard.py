#!/usr/bin/env python
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch
from accelerate import Accelerator
from omegaconf import open_dict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.data import Dataset

from fastwam.datasets.lerobot.dexjoco_contract import DEXJOCO_TASKS
from fastwam.datasets.lerobot.dexjoco_stats import compute_dexjoco_statistics_from_root
from fastwam.datasets.lerobot.utils.normalizer import save_dataset_stats_to_json
from fastwam.trainer import Wan22Trainer
from smoke_test_dexjoco_training_loop import (
    build_six_task_batch,
    compose_config,
    prepare_model,
)


DATA_ROOT = Path("/home/shared/ai/datasets/dexlewm/dexjoco")
REQUIRED_TAGS = {
    "train/loss_total",
    "train/loss_video",
    "train/loss_action_22d",
    "train/loss_arm_6d",
    "train/loss_hand_16d",
    "val/loss_total",
    "val/loss_video",
    "val/loss_action_22d",
    "val/loss_arm_6d",
    "val/loss_hand_16d",
    "lr/action_new",
    "lr/action_backbone",
    "lr/video_backbone",
    "grad_norm/video",
    "grad_norm/arm",
    "grad_norm/hand",
    "grad_norm/action_new",
    "train/step_time",
    "train/data_time",
    "train/epoch",
    "train/samples_seen",
    "action_pred/mean",
    "action_pred/std",
    "action_target/mean",
    "action_target/std",
}


class SampleListDataset(Dataset):
    def __init__(self, samples: list[dict], repeats: int = 1):
        self.samples = samples * repeats

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


def unbatch(batch: dict) -> list[dict]:
    batch_size = batch["action"].shape[0]
    samples = []
    for index in range(batch_size):
        sample = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.shape[:1] == (batch_size,):
                sample[key] = value[index].clone()
            elif isinstance(value, (list, tuple)) and len(value) == batch_size:
                sample[key] = value[index]
            else:
                sample[key] = value
        samples.append(sample)
    return samples


def configure(cfg, *, output_dir: Path, stats_path: Path, report_path: Path, max_steps: int):
    with open_dict(cfg):
        cfg.output_dir = str(output_dir)
        cfg.max_steps = max_steps
        cfg.num_epochs = 1
        cfg.num_workers = 0
        cfg.batch_size = 2
        cfg.mixed_precision = "no"
        cfg.accelerator_cpu = True
        cfg.log_every = 1
        cfg.eval_every = 1
        cfg.save_every = 0
        cfg.gradient_accumulation_steps = 1
        cfg.tensorboard.enabled = True
        cfg.tensorboard.log_dir = str(output_dir / "tensorboard")
        cfg.tensorboard.flush_every = 1
        cfg.data.dexjoco_root = str(DATA_ROOT)
        cfg.data.train.pretrained_norm_stats = str(stats_path)
        cfg.data.train.processor.allow_non_production_stats = True
        cfg.model.selective_checkpoint_report_path = str(report_path)
    return cfg


def read_events(logdir: Path) -> EventAccumulator:
    accumulator = EventAccumulator(str(logdir), size_guidance={"scalars": 0})
    accumulator.Reload()
    return accumulator


def assert_event_contract(accumulator: EventAccumulator) -> dict[str, list]:
    available = set(accumulator.Tags().get("scalars", []))
    missing = sorted(REQUIRED_TAGS - available)
    assert not missing, f"Missing TensorBoard tags: {missing}"
    events = {tag: accumulator.Scalars(tag) for tag in REQUIRED_TAGS}
    for tag, series in events.items():
        assert series, tag
        assert all(torch.isfinite(torch.tensor(event.value)) for event in series), tag
        assert all(event.step > 0 for event in series), (tag, [event.step for event in series])

    for step in sorted({event.step for event in events["lr/action_new"]}):
        values = {
            name: next(event.value for event in events[f"lr/{name}"] if event.step == step)
            for name in ("action_new", "action_backbone", "video_backbone")
        }
        assert abs(values["action_new"] / values["action_backbone"] - 2.0) < 1e-5
        assert abs(values["action_backbone"] / values["video_backbone"] - 5.0) < 1e-5
    for tag in ("grad_norm/video", "grad_norm/arm", "grad_norm/hand", "grad_norm/action_new"):
        assert all(event.value > 0 for event in events[tag]), tag
    return events


def main() -> None:
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(DATA_ROOT)
    Accelerator(cpu=True, mixed_precision="no", step_scheduler_with_optimizer=False)
    torch.manual_seed(29)

    with tempfile.TemporaryDirectory(prefix="fastwam_dexjoco_phase7_tb_") as temp_dir:
        root = Path(temp_dir)
        output_dir = root / "run"
        output_dir.mkdir()
        stats_path = root / "smoke_dataset_stats.json"
        stats = compute_dexjoco_statistics_from_root(
            DATA_ROOT,
            action_horizon=4,
            max_episodes_per_task=1,
            val_set_proportion=0.1,
            split_seed=42,
        )
        save_dataset_stats_to_json(stats, str(stats_path))
        batch, _ = build_six_task_batch(DATA_ROOT, stats_path, output_dir)
        samples = unbatch(batch)
        assert [sample["task_name"] for sample in samples] == list(DEXJOCO_TASKS)
        train_dataset = SampleListDataset(samples, repeats=3)
        val_dataset = SampleListDataset(samples[-2:])

        report_path = output_dir / "dexjoco_selective_load_report.json"
        model = prepare_model(report_path)
        frozen = list(model.text_encoder.parameters()) + list(model.vae.parameters())
        frozen_before = [parameter.detach().clone() for parameter in frozen]
        cfg = configure(
            compose_config(),
            output_dir=output_dir,
            stats_path=stats_path,
            report_path=report_path,
            max_steps=4,
        )
        trainer = Wan22Trainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            cfg=cfg,
        )
        trainer.train()
        assert trainer.global_step == 4
        assert trainer.tensorboard_writer is None
        assert all(parameter.grad is None for parameter in frozen)
        assert all(
            torch.equal(parameter.detach(), before)
            for parameter, before in zip(frozen, frozen_before, strict=True)
        )

        state_path = output_dir / "checkpoints" / "state" / "step_000004"
        assert state_path.is_dir()
        resume_report = output_dir / "resume_selective_load_report.json"
        resumed_model = prepare_model(resume_report)
        resumed_frozen = list(resumed_model.text_encoder.parameters()) + list(
            resumed_model.vae.parameters()
        )
        resumed_cfg = configure(
            copy.deepcopy(cfg),
            output_dir=output_dir,
            stats_path=stats_path,
            report_path=resume_report,
            max_steps=5,
        )
        with open_dict(resumed_cfg):
            resumed_cfg.resume = str(state_path)
        resumed_trainer = Wan22Trainer(
            model=resumed_model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            cfg=resumed_cfg,
        )
        assert resumed_trainer.global_step == 4
        resumed_trainer.train()
        assert resumed_trainer.global_step == 5
        assert resumed_trainer.tensorboard_writer is None
        assert all(parameter.grad is None for parameter in resumed_frozen)

        event_files = sorted((output_dir / "tensorboard").glob("events.out.tfevents.*"))
        assert len(event_files) >= 2
        events = assert_event_contract(read_events(output_dir / "tensorboard"))
        train_steps = [event.step for event in events["train/loss_total"]]
        assert max(train_steps) == 5 and 0 not in train_steps

        summary_json = root / "smoke_summary.json"
        summary_markdown = root / "smoke_summary.md"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("summarize_dexjoco_training.py")),
                "--logdir",
                str(output_dir / "tensorboard"),
                "--output-json",
                str(summary_json),
                "--output-markdown",
                str(summary_markdown),
                "--window",
                "3",
                "--min-points",
                "10",
            ],
            check=True,
        )
        summary = json.loads(summary_json.read_text(encoding="utf-8"))
        assert all(
            metrics["status"] == "insufficient_data"
            for metrics in summary["losses"].values()
        )

        print(f"optimizer_steps=5 initial=4 resumed_from=4 final=5")
        print(f"event_files={len(event_files)} required_tags={len(REQUIRED_TAGS)}")
        print(f"train_loss_steps={train_steps} no_step_zero=PASS resume_continuity=PASS")
        print("lr_ratios=2:1:0.2 grad_norms=finite_nonzero Video/Arm/Hand/action_new=PASS")
        print("T5/VAE gradients=None parameters_unchanged=PASS")
        print("summary=insufficient_data engineering_diagnostic_only=true")
        print("temporary_events_checkpoints_summaries_removed=true formal_training_started=false")


if __name__ == "__main__":
    main()
