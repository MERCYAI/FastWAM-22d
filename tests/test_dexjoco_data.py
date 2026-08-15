from types import SimpleNamespace
import unittest

import torch

from fastwam.datasets.lerobot.dexjoco_contract import (
    DEXJOCO_STATS_STD_FLOOR,
    DEXJOCO_TASKS,
    validate_dexjoco_statistics,
)
from fastwam.datasets.lerobot.dexjoco_stats import compute_dexjoco_statistics
from fastwam.datasets.lerobot.processors.dexjoco_processor import DexJoCoFastWAMProcessor
from fastwam.datasets.lerobot.transforms.action_state_merger import ConcatLeftAlign
from fastwam.datasets.lerobot.utils.normalizer import LinearNormalizer


class _FakeSource:
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.episodes = [SimpleNamespace(episode_index=0), SimpleNamespace(episode_index=1)]

    def load_episode_low_dim(self, episode):
        offset = DEXJOCO_TASKS.index(self.task_name) + episode.episode_index
        action = torch.arange(5 * 22, dtype=torch.float32).reshape(5, 22) / 100 + offset
        state = torch.arange(5 * 23, dtype=torch.float32).reshape(5, 23) / 100 + offset
        action[:, -1] = 0.25
        state[:, -1] = -0.5
        return action, state


def _shape_meta():
    return {
        "images": [
            {"key": "primary", "raw_shape": [3, 640, 640], "shape": [3, 224, 224]},
            {"key": "wrist", "raw_shape": [3, 640, 640], "shape": [3, 224, 224]},
        ],
        "action": [{"key": "default", "raw_shape": 22, "shape": 22}],
        "state": [{"key": "default", "raw_shape": 23, "shape": 23}],
    }


class DexJoCoDataTest(unittest.TestCase):
    def _processor(self, **overrides):
        kwargs = {
            "shape_meta": _shape_meta(),
            "num_obs_steps": 5,
            "num_output_cameras": 2,
            "action_output_dim": 22,
            "proprio_output_dim": 23,
            "delta_action_dim_mask": None,
            "action_state_transforms": None,
            "use_stepwise_action_norm": False,
            "norm_default_mode": "min/max",
            "norm_exception_mode": None,
            "action_state_merger": ConcatLeftAlign(),
            "train_transforms": [],
            "val_transforms": [],
        }
        kwargs.update(overrides)
        return DexJoCoFastWAMProcessor(**kwargs)

    def test_limited_statistics_are_marked_smoke_and_floor_constant_dimensions(self):
        stats = compute_dexjoco_statistics(
            task_sources=[_FakeSource(task) for task in DEXJOCO_TASKS],
            action_horizon=4,
            max_episodes_per_task=1,
        )

        self.assertEqual(stats["statistics_mode"], "smoke")
        self.assertIs(stats["production"], False)
        self.assertEqual(stats["episode_counts"], {task: 1 for task in DEXJOCO_TASKS})
        self.assertEqual(stats["action"]["default"]["stepwise_mean"].shape, (4, 22))
        self.assertEqual(
            stats["action"]["default"]["global_std"][-1], DEXJOCO_STATS_STD_FLOOR
        )
        self.assertEqual(
            stats["state"]["default"]["global_std"][-1], DEXJOCO_STATS_STD_FLOOR
        )
        with self.assertRaisesRegex(ValueError, "requires production"):
            validate_dexjoco_statistics(stats, require_production=True)

    def test_normalize_denormalize_round_trip_and_libero_stats_rejection(self):
        stats = compute_dexjoco_statistics(
            task_sources=[_FakeSource(task) for task in DEXJOCO_TASKS],
            action_horizon=4,
            statistics_mode="production",
        )
        normalizer = LinearNormalizer(
            shape_meta=_shape_meta(),
            use_stepwise_action_norm=False,
            default_mode="z-score",
            exception_mode=None,
            stats=stats,
        )
        action = torch.stack(
            [stats["action"]["default"]["global_mean"]] * 3, dim=0
        )
        state = torch.stack(
            [stats["state"]["default"]["global_mean"]] * 3, dim=0
        )
        batch = {"action": {"default": action.clone()}, "state": {"default": state.clone()}}
        normalized = normalizer.forward(batch)
        restored = normalizer.backward(normalized)
        torch.testing.assert_close(restored["action"]["default"], action, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(restored["state"]["default"], state, atol=1e-6, rtol=1e-6)

        with self.assertRaisesRegex(ValueError, "schema_name"):
            validate_dexjoco_statistics({"action": {}, "state": {}})

    def test_processor_rejects_delta_mask_and_smoke_stats_by_default(self):
        with self.assertRaisesRegex(ValueError, "delta_action_dim_mask"):
            self._processor(delta_action_dim_mask={"default": [False] * 22})

        smoke_stats = compute_dexjoco_statistics(
            task_sources=[_FakeSource(task) for task in DEXJOCO_TASKS],
            action_horizon=4,
            max_episodes_per_task=1,
        )
        with self.assertRaisesRegex(ValueError, "requires production"):
            self._processor().set_normalizer_from_stats(smoke_stats)


if __name__ == "__main__":
    unittest.main()
