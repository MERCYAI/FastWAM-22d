from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import torch

from .dexjoco_contract import (
    DEXJOCO_ACTION_DIM,
    DEXJOCO_ACTION_ORDERING,
    DEXJOCO_ARM_ACTION_DIM,
    DEXJOCO_HAND_ACTION_DIM,
    DEXJOCO_HAND_ACTUATOR_ORDERING,
    DEXJOCO_PROPRIO_DIM,
    DEXJOCO_STATE_ORDERING,
    DEXJOCO_STATS_SCHEMA_NAME,
    DEXJOCO_STATS_SCHEMA_VERSION,
    DEXJOCO_STATS_STD_FLOOR,
    DEXJOCO_TASKS,
    validate_dexjoco_statistics,
)
from .dexjoco_v3_dataset import (
    DEXJOCO_EPISODE_SPLIT_POLICY,
    DexJoCoV3TaskSource,
    select_dexjoco_episode_indices,
)


TensorTransform = Callable[[Dict[str, Dict[str, torch.Tensor]]], Dict[str, Any]]


def _summarize(values: torch.Tensor) -> tuple[Dict[str, torch.Tensor], list[int]]:
    values = values.to(torch.float32)
    raw_std = values.std(dim=0, unbiased=False)
    near_zero = torch.nonzero(raw_std < DEXJOCO_STATS_STD_FLOOR).flatten().tolist()
    return (
        {
            "min": values.amin(dim=0),
            "max": values.amax(dim=0),
            "q01": torch.quantile(values, 0.01, dim=0),
            "q99": torch.quantile(values, 0.99, dim=0),
            "mean": values.mean(dim=0),
            "std": raw_std.clamp_min(DEXJOCO_STATS_STD_FLOOR),
        },
        near_zero,
    )


def _with_prefix(summary: Mapping[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return {f"{prefix}_{key}": value for key, value in summary.items()}


def _stepwise_action_statistics(
    episode_actions: Sequence[torch.Tensor], action_horizon: int
) -> Dict[str, torch.Tensor]:
    summaries: list[Dict[str, torch.Tensor]] = []
    for step in range(action_horizon):
        step_values = []
        for action in episode_actions:
            indices = torch.arange(action.shape[0]).add(step).clamp_max(action.shape[0] - 1)
            step_values.append(action[indices])
        summaries.append(_summarize(torch.cat(step_values, dim=0))[0])
    return {
        f"stepwise_{field}": torch.stack([summary[field] for summary in summaries], dim=0)
        for field in ("min", "max", "q01", "q99", "mean", "std")
    }


def compute_dexjoco_statistics(
    *,
    task_sources: Sequence[DexJoCoV3TaskSource],
    action_horizon: int,
    selected_episode_indices: Optional[Mapping[str, Sequence[int]]] = None,
    transform: Optional[TensorTransform] = None,
    max_episodes_per_task: Optional[int] = None,
    statistics_mode: Optional[str] = None,
    val_set_proportion: float = 0.0,
    split_seed: int = 42,
    subset: str = "train",
) -> Dict[str, Any]:
    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive.")
    if max_episodes_per_task is not None and max_episodes_per_task <= 0:
        raise ValueError("max_episodes_per_task must be positive when specified.")

    inferred_mode = "smoke" if max_episodes_per_task is not None else "production"
    mode = statistics_mode or inferred_mode
    if mode not in {"production", "smoke"}:
        raise ValueError(f"Unsupported statistics_mode: {mode!r}")
    if max_episodes_per_task is not None and mode != "smoke":
        raise ValueError("Limited DexJoCo statistics must be marked as smoke.")
    if subset != "train":
        raise ValueError("DexJoCo normalization statistics may only use the train subset.")

    episode_actions: list[torch.Tensor] = []
    episode_states: list[torch.Tensor] = []
    episode_counts: Dict[str, int] = {}
    frame_counts: Dict[str, int] = {}
    episode_indices: Dict[str, list[int]] = {}
    tasks = [source.task_name for source in task_sources]

    for source in task_sources:
        selected = None
        if selected_episode_indices is not None:
            selected = set(selected_episode_indices[source.task_name])
        episodes = [
            episode
            for episode in source.episodes
            if selected is None or episode.episode_index in selected
        ]
        if max_episodes_per_task is not None:
            episodes = episodes[:max_episodes_per_task]
        if not episodes:
            raise ValueError(f"No training episodes selected for DexJoCo task {source.task_name}.")

        task_frames = 0
        for episode in episodes:
            action, state = source.load_episode_low_dim(episode)
            batch: Dict[str, Any] = {
                "action": {"default": action},
                "state": {"default": state},
            }
            if transform is not None:
                batch = transform(batch)
            action = batch["action"]["default"].to(torch.float32)
            state = batch["state"]["default"].to(torch.float32)
            if action.ndim != 2 or action.shape[1] != DEXJOCO_ACTION_DIM:
                raise ValueError(f"Transformed DexJoCo action must have shape [N, 22], got {action.shape}.")
            if state.ndim != 2 or state.shape[1] != DEXJOCO_PROPRIO_DIM:
                raise ValueError(f"Transformed DexJoCo state must have shape [N, 23], got {state.shape}.")
            episode_actions.append(action)
            episode_states.append(state)
            task_frames += action.shape[0]
        episode_counts[source.task_name] = len(episodes)
        frame_counts[source.task_name] = task_frames
        episode_indices[source.task_name] = [episode.episode_index for episode in episodes]

    global_action, action_near_zero = _summarize(torch.cat(episode_actions, dim=0))
    global_state, state_near_zero = _summarize(torch.cat(episode_states, dim=0))
    action_stats = _with_prefix(global_action, "global")
    action_stats.update(_stepwise_action_statistics(episode_actions, action_horizon))
    state_stats = _with_prefix(global_state, "global")
    state_stats.update(
        {
            f"stepwise_{field}": value.unsqueeze(0)
            for field, value in global_state.items()
        }
    )

    production = mode == "production"
    stats: Dict[str, Any] = {
        "schema_name": DEXJOCO_STATS_SCHEMA_NAME,
        "schema_version": DEXJOCO_STATS_SCHEMA_VERSION,
        "statistics_mode": mode,
        "production": production,
        "split": "train",
        "data_distribution": "rand_obj",
        "episode_split": {
            "policy": DEXJOCO_EPISODE_SPLIT_POLICY,
            "seed": int(split_seed),
            "val_set_proportion": float(val_set_proportion),
            "subset": subset,
            "episode_indices": episode_indices,
        },
        "tasks": tasks,
        "action_ordering": list(DEXJOCO_ACTION_ORDERING),
        "state_ordering": list(DEXJOCO_STATE_ORDERING),
        "hand_actuator_ordering": list(DEXJOCO_HAND_ACTUATOR_ORDERING),
        "action_dim": DEXJOCO_ACTION_DIM,
        "arm_action_dim": DEXJOCO_ARM_ACTION_DIM,
        "hand_action_dim": DEXJOCO_HAND_ACTION_DIM,
        "proprio_dim": DEXJOCO_PROPRIO_DIM,
        "action_horizon": action_horizon,
        "std_floor": DEXJOCO_STATS_STD_FLOOR,
        "std_floor_applied_dimensions": {
            "action": action_near_zero,
            "state": state_near_zero,
        },
        "episode_counts": episode_counts,
        "frame_counts": frame_counts,
        "num_episodes": sum(episode_counts.values()),
        "num_transition": sum(frame_counts.values()),
        "max_episodes_per_task": max_episodes_per_task,
        "action": {"default": action_stats},
        "state": {"default": state_stats},
    }
    validate_dexjoco_statistics(
        stats,
        require_production=production,
        expected_tasks=tasks,
    )
    return stats


def compute_dexjoco_statistics_from_root(
    data_root: str | Path,
    *,
    output_tasks: Sequence[str] = DEXJOCO_TASKS,
    split: str = "train",
    action_horizon: int = 32,
    max_episodes_per_task: Optional[int] = None,
    val_set_proportion: float = 0.0,
    split_seed: int = 42,
) -> Dict[str, Any]:
    if split != "train":
        raise ValueError("DexJoCo statistics may only be computed from the training split.")
    root = Path(data_root).resolve()
    sources = [
        DexJoCoV3TaskSource(root / task, task, split=split) for task in output_tasks
    ]
    selected_episode_indices = {
        source.task_name: select_dexjoco_episode_indices(
            [episode.episode_index for episode in source.episodes],
            task_name=source.task_name,
            val_set_proportion=val_set_proportion,
            is_training_set=True,
            seed=split_seed,
        )
        for source in sources
    }
    return compute_dexjoco_statistics(
        task_sources=sources,
        action_horizon=action_horizon,
        selected_episode_indices=selected_episode_indices,
        max_episodes_per_task=max_episodes_per_task,
        val_set_proportion=val_set_proportion,
        split_seed=split_seed,
        subset="train",
    )
