from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from ..dexjoco_contract import (
    DEXJOCO_ACTION_DIM,
    DEXJOCO_PROPRIO_DIM,
    DEXJOCO_TASKS,
    validate_dexjoco_shape_meta,
    validate_dexjoco_statistics,
)
from .fastwam_processor import FastWAMProcessor


class DexJoCoFastWAMProcessor(FastWAMProcessor):
    def __init__(
        self,
        *,
        shape_meta,
        num_output_cameras: int,
        action_output_dim: int,
        proprio_output_dim: int,
        delta_action_dim_mask: Optional[Dict[str, Any]] = None,
        allow_non_production_stats: bool = False,
        expected_tasks: Sequence[str] = DEXJOCO_TASKS,
        **kwargs,
    ):
        validate_dexjoco_shape_meta(shape_meta)
        if action_output_dim != DEXJOCO_ACTION_DIM:
            raise ValueError("DexJoCo action_output_dim must be 22.")
        if proprio_output_dim != DEXJOCO_PROPRIO_DIM:
            raise ValueError("DexJoCo proprio_output_dim must be 23.")
        if num_output_cameras != 2:
            raise ValueError("DexJoCo num_output_cameras must be 2.")
        if delta_action_dim_mask is not None:
            raise ValueError(
                "DexJoCo actions are absolute TCP targets; delta_action_dim_mask must be null/None."
            )
        self.allow_non_production_stats = allow_non_production_stats
        self.expected_tasks = tuple(expected_tasks)
        super().__init__(
            shape_meta=shape_meta,
            num_output_cameras=num_output_cameras,
            action_output_dim=action_output_dim,
            proprio_output_dim=proprio_output_dim,
            delta_action_dim_mask=None,
            **kwargs,
        )

    def set_normalizer_from_stats(self, dataset_stats: Dict[str, Any] = None):
        validate_dexjoco_statistics(
            dataset_stats,
            require_production=not self.allow_non_production_stats,
            expected_tasks=self.expected_tasks,
        )
        return super().set_normalizer_from_stats(dataset_stats)
