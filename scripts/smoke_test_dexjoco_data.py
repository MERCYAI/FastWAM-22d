#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from fastwam.datasets.lerobot.dexjoco_contract import DEXJOCO_TASKS
from fastwam.datasets.lerobot.dexjoco_robot_video_dataset import DexJoCoRobotVideoDataset
from fastwam.datasets.lerobot.processors.dexjoco_processor import DexJoCoFastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.transforms.action_state_merger import ConcatLeftAlign
from fastwam.datasets.lerobot.transforms.image import ToTensor
from fastwam.utils import misc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a six-task DexJoCo data-path smoke test.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def shape_meta():
    return {
        "images": [
            {"key": "primary", "raw_shape": [3, 640, 640], "shape": [3, 224, 224]},
            {"key": "wrist", "raw_shape": [3, 640, 640], "shape": [3, 224, 224]},
        ],
        "action": [{"key": "default", "raw_shape": 22, "shape": 22}],
        "state": [{"key": "default", "raw_shape": 23, "shape": 23}],
    }


def build_processor(meta):
    image_transforms = [ToTensor(), transforms.Resize([224, 224])]
    return DexJoCoFastWAMProcessor(
        shape_meta=meta,
        num_obs_steps=33,
        num_output_cameras=2,
        action_output_dim=22,
        proprio_output_dim=23,
        delta_action_dim_mask=None,
        allow_non_production_stats=True,
        expected_tasks=DEXJOCO_TASKS,
        action_state_transforms=None,
        use_stepwise_action_norm=False,
        norm_default_mode="min/max",
        norm_exception_mode=None,
        action_state_merger=ConcatLeftAlign(),
        train_transforms=image_transforms,
        val_transforms=image_transforms,
    )


def write_smoke_contexts(dataset, cache_dir: Path, context_len: int) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for source in dataset.lerobot_dataset.sources:
        instruction = source.episodes[0].instruction
        prompt = DEFAULT_PROMPT.format(task=instruction)
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        path = cache_dir / f"{digest}.t5_len{context_len}.wan22ti2v5b.pt"
        torch.save(
            {
                "context": torch.zeros(context_len, 1, dtype=torch.float32),
                "mask": torch.ones(context_len, dtype=torch.bool),
            },
            path,
        )


def first_task_indices(dataset) -> dict[str, int]:
    result = {}
    start = 0
    for _, episode in dataset.lerobot_dataset.selected_episodes:
        result.setdefault(episode.task_name, start)
        start += episode.length
    return result


def check_round_trip(dataset) -> tuple[float, float]:
    source = dataset.lerobot_dataset.sources[0]
    action, state = source.load_episode_low_dim(source.episodes[0])
    expected_action = action[:4].clone()
    expected_state = state[:4].clone()
    batch = {
        "action": {"default": expected_action.clone()},
        "state": {"default": expected_state.clone()},
    }
    normalized = dataset.lerobot_dataset.processor.normalizer.forward(batch)
    restored = dataset.lerobot_dataset.processor.normalizer.backward(normalized)
    action_error = (restored["action"]["default"] - expected_action).abs().max().item()
    state_error = (restored["state"]["default"] - expected_state).abs().max().item()
    return action_error, state_error


def main() -> None:
    args = parse_args()
    meta = shape_meta()
    dataset_dirs = [str(args.data_root / task) for task in DEXJOCO_TASKS]
    with tempfile.TemporaryDirectory(prefix="fastwam_dexjoco_smoke_") as temp_dir:
        temp = Path(temp_dir)
        cache_dir = temp / "text_cache"
        misc.register_work_dir(temp / "work")
        dataset = DexJoCoRobotVideoDataset(
            dataset_dirs=dataset_dirs,
            task_names=DEXJOCO_TASKS,
            split="train",
            shape_meta=meta,
            num_frames=33,
            video_size=[224, 448],
            global_sample_stride=1,
            action_video_freq_ratio=4,
            val_set_proportion=0.0,
            is_training_set=True,
            skip_padding_as_possible=False,
            concat_multi_camera="horizontal",
            return_camera_videos=True,
            processor=build_processor(meta),
            pretrained_norm_stats=str(args.stats),
            text_embedding_cache_dir=str(cache_dir),
            context_len=128,
        )
        write_smoke_contexts(dataset, cache_dir, 128)
        indices = first_task_indices(dataset)
        loader = DataLoader(
            Subset(dataset, [indices[task] for task in DEXJOCO_TASKS]),
            batch_size=1,
            num_workers=args.num_workers,
        )
        for task, batch in zip(DEXJOCO_TASKS, loader, strict=True):
            assert batch["task_name"][0] == task
            assert batch["action"].shape == (1, 32, 22)
            assert batch["arm_action"].shape == (1, 32, 6)
            assert batch["hand_action"].shape == (1, 32, 16)
            assert batch["proprio"].shape == (1, 32, 23)
            assert batch["camera_videos"].shape == (1, 2, 3, 9, 224, 224)
            assert batch["video"].shape == (1, 3, 9, 224, 448)
            assert batch["action_is_pad"].shape == (1, 32)
            assert batch["proprio_is_pad"].shape == (1, 32)
            assert batch["image_is_pad"].shape == (1, 9)
            print(
                f"task={task} action={tuple(batch['action'].shape)} "
                f"arm={tuple(batch['arm_action'].shape)} hand={tuple(batch['hand_action'].shape)} "
                f"state={tuple(batch['proprio'].shape)} "
                f"cameras={tuple(batch['camera_videos'].shape)} video={tuple(batch['video'].shape)}"
            )

        first_episode_length = dataset.lerobot_dataset.selected_episodes[0][1].length
        boundary = dataset[first_episode_length - 1]
        assert boundary["action_is_pad"].tolist() == [False] + [True] * 31
        assert boundary["proprio_is_pad"].tolist() == [False] + [True] * 31
        assert boundary["image_is_pad"].tolist() == [False] + [True] * 8
        print("padding_boundary action_pad=31/32 state_pad=31/32 image_pad=8/9")

        action_error, state_error = check_round_trip(dataset)
        assert action_error < 1e-5
        assert state_error < 1e-5
        print(f"round_trip_max_error action={action_error:.3e} state={state_error:.3e}")


if __name__ == "__main__":
    main()
