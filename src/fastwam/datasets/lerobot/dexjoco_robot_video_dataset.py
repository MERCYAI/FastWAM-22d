from __future__ import annotations

from typing import Sequence

from .dexjoco_contract import (
    DEXJOCO_ACTION_DIM,
    DEXJOCO_ARM_ACTION_DIM,
    DEXJOCO_HAND_ACTION_DIM,
    DEXJOCO_PROPRIO_DIM,
    DEXJOCO_TASKS,
)
from .dexjoco_v3_dataset import DexJoCoV3Dataset
from .robot_video_dataset import RobotVideoDataset


class DexJoCoRobotVideoDataset(RobotVideoDataset):
    def __init__(
        self,
        *args,
        task_names: Sequence[str] = DEXJOCO_TASKS,
        split: str = "train",
        return_camera_videos: bool = True,
        **kwargs,
    ):
        if tuple(task_names) != DEXJOCO_TASKS:
            raise ValueError(f"DexJoCo six-task contract requires tasks {list(DEXJOCO_TASKS)}.")
        if split != "train":
            raise ValueError("DexJoCo Phase 1 dataset split must be `train`.")
        if not return_camera_videos:
            raise ValueError("DexJoCo must return both camera tensors.")
        self.dexjoco_task_names = tuple(task_names)
        self.dexjoco_split = split
        super().__init__(*args, return_camera_videos=True, **kwargs)

    def _build_lerobot_dataset(self, **kwargs):
        return DexJoCoV3Dataset(
            **kwargs,
            task_names=self.dexjoco_task_names,
            split=self.dexjoco_split,
        )

    def _get(self, idx):
        data = super()._get(idx)
        action = data["action"]
        proprio = data["proprio"]
        camera_videos = data["camera_videos"]
        if action.ndim != 2 or action.shape[-1] != DEXJOCO_ACTION_DIM:
            raise ValueError(f"DexJoCo action must have shape [T, 22], got {tuple(action.shape)}.")
        if proprio.ndim != 2 or proprio.shape != (action.shape[0], DEXJOCO_PROPRIO_DIM):
            raise ValueError(
                "DexJoCo proprio must have shape [T, 23] aligned with action, "
                f"got {tuple(proprio.shape)}."
            )
        if camera_videos.ndim != 5 or camera_videos.shape[0] != 2:
            raise ValueError(
                "DexJoCo camera_videos must have shape [2, C, T_video, H, W], "
                f"got {tuple(camera_videos.shape)}."
            )
        if data["video"].shape[1] != camera_videos.shape[2]:
            raise ValueError("DexJoCo concatenated and per-camera video time axes must match.")

        proprio_is_pad = data["proprio_is_pad"]
        if proprio_is_pad.shape[0] == action.shape[0] + 1:
            data["proprio_is_pad"] = proprio_is_pad[:-1]
        if data["action_is_pad"].shape[0] != action.shape[0]:
            raise ValueError("DexJoCo action padding mask must align with the action chunk.")
        if data["proprio_is_pad"].shape[0] != proprio.shape[0]:
            raise ValueError("DexJoCo proprio padding mask must align with proprio.")
        if data["image_is_pad"].shape[0] != camera_videos.shape[2]:
            raise ValueError("DexJoCo image padding mask must align with the video time axis.")

        data["arm_action"] = action[..., :DEXJOCO_ARM_ACTION_DIM]
        data["hand_action"] = action[
            ..., DEXJOCO_ARM_ACTION_DIM : DEXJOCO_ARM_ACTION_DIM + DEXJOCO_HAND_ACTION_DIM
        ]
        return data
