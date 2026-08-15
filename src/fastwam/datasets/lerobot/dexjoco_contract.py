from __future__ import annotations

from typing import Any, Mapping, Sequence


DEXJOCO_TASKS = (
    "water_plant",
    "hammer_nail",
    "click_mouse",
    "pick_bucket",
    "pinch_tongs",
    "fold_glasses",
)

DEXJOCO_HAND_ACTUATOR_ORDERING = (
    "ffj0",
    "ffj1",
    "ffj2",
    "ffj3",
    "mfj0",
    "mfj1",
    "mfj2",
    "mfj3",
    "rfj0",
    "rfj1",
    "rfj2",
    "rfj3",
    "thj0",
    "thj1",
    "thj2",
    "thj3",
)

DEXJOCO_ACTION_ORDERING = (
    "tcp_x_m",
    "tcp_y_m",
    "tcp_z_m",
    "tcp_rotvec_x_rad",
    "tcp_rotvec_y_rad",
    "tcp_rotvec_z_rad",
    *DEXJOCO_HAND_ACTUATOR_ORDERING,
)

DEXJOCO_STATE_ORDERING = (
    "tcp_x_m",
    "tcp_y_m",
    "tcp_z_m",
    "tcp_quat_w",
    "tcp_quat_x",
    "tcp_quat_y",
    "tcp_quat_z",
    *DEXJOCO_HAND_ACTUATOR_ORDERING,
)

DEXJOCO_ACTION_DIM = 22
DEXJOCO_ARM_ACTION_DIM = 6
DEXJOCO_HAND_ACTION_DIM = 16
DEXJOCO_PROPRIO_DIM = 23

DEXJOCO_STATS_SCHEMA_NAME = "fastwam.dexjoco.dataset_stats"
DEXJOCO_STATS_SCHEMA_VERSION = 1
DEXJOCO_STATS_STD_FLOOR = 1e-6


def _value_shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(shape)
    if isinstance(value, (list, tuple)):
        if not value:
            return (0,)
        return (len(value), *_value_shape(value[0]))
    return ()


def validate_dexjoco_shape_meta(shape_meta: Mapping[str, Any]) -> None:
    action = list(shape_meta["action"])
    state = list(shape_meta["state"])
    images = list(shape_meta["images"])
    if len(action) != 1 or action[0]["key"] != "default":
        raise ValueError("DexJoCo requires one `action.default` field.")
    if action[0]["raw_shape"] != DEXJOCO_ACTION_DIM or action[0]["shape"] != DEXJOCO_ACTION_DIM:
        raise ValueError("DexJoCo action raw/output dimensions must both be 22.")
    if len(state) != 1 or state[0]["key"] != "default":
        raise ValueError("DexJoCo requires one `state.default` field.")
    if state[0]["raw_shape"] != DEXJOCO_PROPRIO_DIM or state[0]["shape"] != DEXJOCO_PROPRIO_DIM:
        raise ValueError("DexJoCo state raw/output dimensions must both be 23.")
    if [item["key"] for item in images] != ["primary", "wrist"]:
        raise ValueError("DexJoCo image fields must be ordered as `primary`, then `wrist`.")


def validate_dexjoco_statistics(
    stats: Mapping[str, Any],
    *,
    require_production: bool = True,
    expected_tasks: Sequence[str] = DEXJOCO_TASKS,
) -> None:
    expected_scalars = {
        "schema_name": DEXJOCO_STATS_SCHEMA_NAME,
        "schema_version": DEXJOCO_STATS_SCHEMA_VERSION,
        "split": "train",
        "action_dim": DEXJOCO_ACTION_DIM,
        "arm_action_dim": DEXJOCO_ARM_ACTION_DIM,
        "hand_action_dim": DEXJOCO_HAND_ACTION_DIM,
        "proprio_dim": DEXJOCO_PROPRIO_DIM,
    }
    for key, expected in expected_scalars.items():
        actual = stats.get(key)
        if actual != expected:
            raise ValueError(
                f"Invalid DexJoCo statistics `{key}`: expected {expected!r}, got {actual!r}."
            )

    expected_sequences = {
        "tasks": tuple(expected_tasks),
        "action_ordering": DEXJOCO_ACTION_ORDERING,
        "state_ordering": DEXJOCO_STATE_ORDERING,
        "hand_actuator_ordering": DEXJOCO_HAND_ACTUATOR_ORDERING,
    }
    for key, expected in expected_sequences.items():
        actual = stats.get(key)
        if actual is None or tuple(actual) != tuple(expected):
            raise ValueError(f"Invalid DexJoCo statistics `{key}`.")

    production = stats.get("production")
    mode = stats.get("statistics_mode")
    if production not in (True, False):
        raise ValueError("DexJoCo statistics `production` must be boolean.")
    if production is True and mode != "production":
        raise ValueError("Production DexJoCo statistics must use `statistics_mode=production`.")
    if require_production and (production is not True or mode != "production"):
        raise ValueError(
            "DexJoCo training/inference requires production training-split statistics; "
            f"got production={production!r}, statistics_mode={mode!r}."
        )
    if production is False and mode != "smoke":
        raise ValueError("Non-production DexJoCo statistics must use `statistics_mode=smoke`.")

    action_horizon = stats.get("action_horizon")
    if not isinstance(action_horizon, int) or action_horizon <= 0:
        raise ValueError("DexJoCo statistics `action_horizon` must be a positive integer.")
    if stats.get("std_floor") != DEXJOCO_STATS_STD_FLOOR:
        raise ValueError(f"DexJoCo statistics `std_floor` must be {DEXJOCO_STATS_STD_FLOOR}.")

    required_fields = {
        "global_min",
        "global_max",
        "global_q01",
        "global_q99",
        "global_mean",
        "global_std",
        "stepwise_min",
        "stepwise_max",
        "stepwise_q01",
        "stepwise_q99",
        "stepwise_mean",
        "stepwise_std",
    }
    for group in ("action", "state"):
        default_stats = stats.get(group, {}).get("default", {})
        missing = sorted(required_fields - set(default_stats))
        if missing:
            raise ValueError(f"DexJoCo statistics `{group}.default` misses fields: {missing}.")
        dim = DEXJOCO_ACTION_DIM if group == "action" else DEXJOCO_PROPRIO_DIM
        step_count = action_horizon if group == "action" else 1
        for field in required_fields:
            expected_shape = (step_count, dim) if field.startswith("stepwise_") else (dim,)
            actual_shape = _value_shape(default_stats[field])
            if actual_shape != expected_shape:
                raise ValueError(
                    f"DexJoCo statistics `{group}.default.{field}` must have shape "
                    f"{expected_shape}, got {actual_shape}."
                )
