from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import torch

try:
    from torchcodec.decoders import VideoDecoder as TorchCodecVideoDecoder
except (ImportError, RuntimeError):
    TorchCodecVideoDecoder = None

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .dexjoco_contract import (
    DEXJOCO_ACTION_DIM,
    DEXJOCO_PROPRIO_DIM,
    DEXJOCO_TASKS,
    validate_dexjoco_shape_meta,
)
from .processors.base_processor import BaseProcessor


@dataclass(frozen=True)
class DexJoCoEpisode:
    task_name: str
    instruction: str
    episode_index: int
    length: int
    dataset_from_index: int
    dataset_to_index: int
    data_chunk_index: int
    data_file_index: int
    camera_metadata: Mapping[str, Mapping[str, float | int]]


def _fixed_size_list_to_numpy(column: pa.ChunkedArray, dim: int) -> np.ndarray:
    array = column.combine_chunks()
    values = array.values.to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=np.float32).reshape(len(array), dim)


def _parse_episode_split(value: Any, available_indices: Sequence[int]) -> set[int]:
    if isinstance(value, str):
        start_text, stop_text = value.split(":", maxsplit=1)
        start = int(start_text) if start_text else 0
        stop = int(stop_text) if stop_text else max(available_indices) + 1
        return set(range(start, stop))
    if isinstance(value, (list, tuple)):
        return {int(index) for index in value}
    raise ValueError(f"Unsupported LeRobot v3 split declaration: {value!r}")


class DexJoCoV3TaskSource:
    def __init__(self, root: str | Path, task_name: str, split: str = "train"):
        self.root = Path(root).resolve()
        self.task_name = task_name
        self.split = split
        if not self.root.is_dir():
            raise FileNotFoundError(f"DexJoCo task directory does not exist: {self.root}")

        info_path = self.root / "meta" / "info.json"
        with info_path.open("r", encoding="utf-8") as stream:
            self.info = json.load(stream)
        self._validate_info()

        image_keys = {
            key for key, feature in self.info["features"].items() if feature.get("dtype") == "video"
        }
        if "observation.images.front" in image_keys:
            primary_key = "observation.images.front"
        elif "observation.images.ego_right" in image_keys:
            primary_key = "observation.images.ego_right"
        else:
            raise ValueError(f"{task_name} has neither `front` nor `ego_right` primary camera.")
        if "observation.images.wrist" not in image_keys:
            raise ValueError(f"{task_name} is missing `observation.images.wrist`.")
        if image_keys != {primary_key, "observation.images.wrist"}:
            raise ValueError(f"{task_name} has unexpected video keys: {sorted(image_keys)}")
        self.camera_keys = {"primary": primary_key, "wrist": "observation.images.wrist"}

        self.fps = int(self.info["fps"])
        self._data_cache: Dict[Path, Dict[str, np.ndarray]] = {}
        self._video_decoders: Dict[Path, Any] = {}
        self._all_episodes = self._load_episodes()
        split_indices = _parse_episode_split(
            self.info["splits"][split], [episode.episode_index for episode in self._all_episodes]
        )
        self.episodes = [
            episode for episode in self._all_episodes if episode.episode_index in split_indices
        ]
        if not self.episodes:
            raise ValueError(f"DexJoCo task {task_name} split {split!r} contains no episodes.")

        self._data_file_starts: Dict[tuple[int, int], int] = {}
        for episode in self._all_episodes:
            key = (episode.data_chunk_index, episode.data_file_index)
            current = self._data_file_starts.get(key)
            if current is None or episode.dataset_from_index < current:
                self._data_file_starts[key] = episode.dataset_from_index

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_video_decoders"] = {}
        return state

    def _validate_info(self) -> None:
        if self.info.get("codebase_version") != "v3.0":
            raise ValueError(
                f"DexJoCo loader requires LeRobot v3.0, got {self.info.get('codebase_version')!r}."
            )
        if self.split not in self.info.get("splits", {}):
            raise ValueError(f"DexJoCo task {self.task_name} has no split {self.split!r}.")
        expected_features = {
            "action": ("float32", [DEXJOCO_ACTION_DIM]),
            "observation.state": ("float32", [DEXJOCO_PROPRIO_DIM]),
        }
        for key, (dtype, shape) in expected_features.items():
            feature = self.info.get("features", {}).get(key, {})
            if feature.get("dtype") != dtype or feature.get("shape") != shape:
                raise ValueError(
                    f"DexJoCo {self.task_name} feature {key!r} must be {dtype}{shape}, got {feature}."
                )

    def _load_episodes(self) -> list[DexJoCoEpisode]:
        episode_files = sorted((self.root / "meta" / "episodes").glob("**/*.parquet"))
        if not episode_files:
            raise FileNotFoundError(f"No LeRobot v3 episode metadata found under {self.root}.")
        primary_key = self.camera_keys["primary"]
        wrist_key = self.camera_keys["wrist"]
        columns = [
            "episode_index",
            "tasks",
            "length",
            "data/chunk_index",
            "data/file_index",
            "dataset_from_index",
            "dataset_to_index",
        ]
        for key in (primary_key, wrist_key):
            columns.extend(
                [
                    f"videos/{key}/chunk_index",
                    f"videos/{key}/file_index",
                    f"videos/{key}/from_timestamp",
                    f"videos/{key}/to_timestamp",
                ]
            )

        episodes: list[DexJoCoEpisode] = []
        for path in episode_files:
            for row in pq.read_table(path, columns=columns).to_pylist():
                task_values = row["tasks"]
                instruction = task_values[0] if task_values else self.task_name.replace("_", " ")
                camera_metadata = {}
                for alias, key in self.camera_keys.items():
                    camera_metadata[alias] = {
                        "chunk_index": int(row[f"videos/{key}/chunk_index"]),
                        "file_index": int(row[f"videos/{key}/file_index"]),
                        "from_timestamp": float(row[f"videos/{key}/from_timestamp"]),
                        "to_timestamp": float(row[f"videos/{key}/to_timestamp"]),
                    }
                for alias, metadata in camera_metadata.items():
                    duration = metadata["to_timestamp"] - metadata["from_timestamp"]
                    if abs(duration - int(row["length"]) / self.fps) > 1e-5:
                        raise ValueError(
                            f"DexJoCo {self.task_name} episode {row['episode_index']} {alias} "
                            "video duration does not match its frame count."
                        )
                episodes.append(
                    DexJoCoEpisode(
                        task_name=self.task_name,
                        instruction=instruction,
                        episode_index=int(row["episode_index"]),
                        length=int(row["length"]),
                        dataset_from_index=int(row["dataset_from_index"]),
                        dataset_to_index=int(row["dataset_to_index"]),
                        data_chunk_index=int(row["data/chunk_index"]),
                        data_file_index=int(row["data/file_index"]),
                        camera_metadata=camera_metadata,
                    )
                )
        episodes.sort(key=lambda episode: episode.episode_index)
        if len(episodes) != int(self.info["total_episodes"]):
            raise ValueError(
                f"DexJoCo {self.task_name} episode count mismatch: "
                f"metadata has {len(episodes)}, info declares {self.info['total_episodes']}."
            )
        return episodes

    def _data_path(self, episode: DexJoCoEpisode) -> Path:
        relative = self.info["data_path"].format(
            chunk_index=episode.data_chunk_index,
            file_index=episode.data_file_index,
        )
        return self.root / relative

    def _load_data_file(self, path: Path) -> Dict[str, np.ndarray]:
        cached = self._data_cache.get(path)
        if cached is not None:
            return cached
        table = pq.read_table(path, columns=["action", "observation.state", "episode_index"])
        cached = {
            "action": _fixed_size_list_to_numpy(table["action"], DEXJOCO_ACTION_DIM),
            "state": _fixed_size_list_to_numpy(table["observation.state"], DEXJOCO_PROPRIO_DIM),
            "episode_index": np.asarray(table["episode_index"].to_numpy(), dtype=np.int64),
        }
        self._data_cache[path] = cached
        return cached

    def load_episode_low_dim(self, episode: DexJoCoEpisode) -> tuple[torch.Tensor, torch.Tensor]:
        file_key = (episode.data_chunk_index, episode.data_file_index)
        file_start = self._data_file_starts[file_key]
        row_from = episode.dataset_from_index - file_start
        row_to = episode.dataset_to_index - file_start
        data = self._load_data_file(self._data_path(episode))
        if row_to - row_from != episode.length:
            raise ValueError(f"DexJoCo {self.task_name} episode {episode.episode_index} length mismatch.")
        episode_indices = data["episode_index"][row_from:row_to]
        if not np.all(episode_indices == episode.episode_index):
            raise ValueError(
                f"DexJoCo {self.task_name} episode row mapping does not match episode_index."
            )
        action = torch.from_numpy(data["action"][row_from:row_to].copy())
        state = torch.from_numpy(data["state"][row_from:row_to].copy())
        return action, state

    def _video_path(self, episode: DexJoCoEpisode, alias: str) -> Path:
        metadata = episode.camera_metadata[alias]
        relative = self.info["video_path"].format(
            video_key=self.camera_keys[alias],
            chunk_index=metadata["chunk_index"],
            file_index=metadata["file_index"],
        )
        return self.root / relative

    def load_video_frames(
        self, episode: DexJoCoEpisode, alias: str, frame_indices: Iterable[int]
    ) -> torch.Tensor:
        path = self._video_path(episode, alias)
        if not path.is_file():
            raise FileNotFoundError(f"Missing DexJoCo video file: {path}")
        metadata = episode.camera_metadata[alias]
        local_indices = [int(index) for index in frame_indices]
        timestamps = [
            float(metadata["from_timestamp"]) + index / self.fps for index in local_indices
        ]
        try:
            if TorchCodecVideoDecoder is None:
                raise ImportError("torchcodec is unavailable")
            decoder = self._video_decoders.get(path)
            if decoder is None:
                decoder = TorchCodecVideoDecoder(str(path), device="cpu", seek_mode="approximate")
                self._video_decoders[path] = decoder
            absolute_indices = [round(timestamp * self.fps) for timestamp in timestamps]
            frames = decoder.get_frames_at(indices=absolute_indices).data
            return frames.to(torch.uint8)
        except Exception:
            from .lerobot.datasets.video_utils import decode_video_frames

            frames = decode_video_frames(path, timestamps, tolerance_s=0.51 / self.fps)
            return (frames * 255.0).round().to(torch.uint8)


class DexJoCoV3Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs: Sequence[str],
        shape_meta: Mapping[str, Any],
        action_size: int,
        obs_size: int,
        val_set_proportion: float = 0.0,
        is_training_set: bool = True,
        seed: int = 42,
        global_sample_stride: int = 1,
        task_names: Sequence[str] = DEXJOCO_TASKS,
        split: str = "train",
        **_: Any,
    ):
        validate_dexjoco_shape_meta(shape_meta)
        if action_size != obs_size - 1:
            raise ValueError("DexJoCo action_size must equal obs_size - 1.")
        if split != "train":
            raise ValueError("DexJoCo Phase 1 only permits the declared training split.")
        if len(dataset_dirs) != len(task_names):
            raise ValueError("DexJoCo requires one dataset directory per configured task.")

        resolved_dirs = [Path(path).resolve() for path in dataset_dirs]
        directory_tasks = [path.name for path in resolved_dirs]
        if directory_tasks != list(task_names):
            raise ValueError(
                "DexJoCo dataset_dirs must follow the configured task order; "
                f"expected {list(task_names)}, got {directory_tasks}."
            )

        self.shape_meta = shape_meta
        self.action_size = action_size
        self.obs_size = obs_size
        self.global_sample_stride = global_sample_stride
        self.is_training_set = is_training_set
        self.split = split
        self.task_names = tuple(task_names)
        self.processor: Optional[BaseProcessor] = None
        self.return_images = False

        self.sources = [
            DexJoCoV3TaskSource(path, task_name, split=split)
            for path, task_name in zip(resolved_dirs, self.task_names, strict=True)
        ]
        fps_values = {source.fps for source in self.sources}
        if fps_values != {30}:
            raise ValueError(f"DexJoCo tasks must all be 30 FPS, got {sorted(fps_values)}.")

        rng = np.random.default_rng(seed)
        self.selected_episodes: list[tuple[DexJoCoV3TaskSource, DexJoCoEpisode]] = []
        for source in self.sources:
            episodes = list(source.episodes)
            if val_set_proportion >= 1e-6:
                indices = np.arange(len(episodes))
                rng.shuffle(indices)
                split_index = int(len(indices) * (1 - val_set_proportion))
                selected = indices[:split_index] if is_training_set else indices[split_index:]
                episodes = [episodes[index] for index in selected]
            self.selected_episodes.extend((source, episode) for episode in episodes)
        if not self.selected_episodes:
            raise ValueError("DexJoCo dataset selection contains no episodes.")

        self._episode_ends = np.cumsum(
            [episode.length for _, episode in self.selected_episodes], dtype=np.int64
        ).tolist()

    def __len__(self) -> int:
        return self._episode_ends[-1]

    def _set_return_images(self, flag: bool) -> None:
        self.return_images = flag

    def set_processor(self, processor: BaseProcessor):
        self.processor = processor
        if self.is_training_set:
            processor.train()
        else:
            processor.eval()
        return self

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of bounds for DexJoCo dataset of size {len(self)}.")
        episode_slot = bisect.bisect_right(self._episode_ends, index)
        episode_start = 0 if episode_slot == 0 else self._episode_ends[episode_slot - 1]
        local_index = index - episode_start
        source, episode = self.selected_episodes[episode_slot]
        action, state = source.load_episode_low_dim(episode)

        obs_unclamped = local_index + np.arange(self.obs_size) * self.global_sample_stride
        action_unclamped = local_index + np.arange(self.action_size) * self.global_sample_stride
        obs_is_pad = torch.from_numpy(obs_unclamped >= episode.length)
        action_is_pad = torch.from_numpy(action_unclamped >= episode.length)
        obs_indices = np.minimum(obs_unclamped, episode.length - 1)
        action_indices = np.minimum(action_unclamped, episode.length - 1)

        sample: Dict[str, Any] = {
            "idx": index,
            "task": episode.instruction,
            "task_name": episode.task_name,
            "episode_index": episode.episode_index,
            "action": {"default": action[action_indices]},
            "state": {"default": state[obs_indices]},
            "images": {},
            "action_is_pad": action_is_pad,
            "state_is_pad": obs_is_pad.clone(),
            "image_is_pad": obs_is_pad.clone(),
        }
        if self.return_images:
            sample["images"] = {
                alias: source.load_video_frames(episode, alias, obs_indices)
                for alias in ("primary", "wrist")
            }
        if self.processor is not None:
            sample = self.processor.preprocess(sample)
            sample["task_name"] = episode.task_name
            sample["episode_index"] = episode.episode_index
        return sample

    def get_dataset_stats(self, preprocessor: BaseProcessor) -> Dict[str, Any]:
        from .dexjoco_stats import compute_dexjoco_statistics

        selected: Dict[str, list[int]] = {task: [] for task in self.task_names}
        for _, episode in self.selected_episodes:
            selected[episode.task_name].append(episode.episode_index)
        return compute_dexjoco_statistics(
            task_sources=self.sources,
            selected_episode_indices=selected,
            action_horizon=self.action_size,
            transform=preprocessor.action_state_transform,
            statistics_mode="production",
        )
