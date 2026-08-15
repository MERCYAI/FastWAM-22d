"""Strict, versioned FastWAM-DexJoCo websocket protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from fastwam.datasets.lerobot.dexjoco_contract import (
    DEXJOCO_ACTION_DIM,
    DEXJOCO_ACTION_ORDERING,
    DEXJOCO_PROPRIO_DIM,
)


FASTWAM_DEXJOCO_PROTOCOL_SCHEMA = "fastwam.dexjoco.websocket"
FASTWAM_DEXJOCO_PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class DexJoCoInferenceRequest:
    primary: np.ndarray
    wrist: np.ndarray
    state: np.ndarray
    prompt: str
    horizon: int


def protocol_metadata(*, horizon: int) -> dict[str, Any]:
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
        raise ValueError("Server horizon must be a positive integer.")
    return {
        "schema_name": FASTWAM_DEXJOCO_PROTOCOL_SCHEMA,
        "schema_version": FASTWAM_DEXJOCO_PROTOCOL_VERSION,
        "action_dim": DEXJOCO_ACTION_DIM,
        "proprio_dim": DEXJOCO_PROPRIO_DIM,
        "action_ordering": list(DEXJOCO_ACTION_ORDERING),
        "horizon": horizon,
        "action_mode": "absolute_tcp_xyz_rotvec",
    }


def _validate_schema(payload: Mapping[str, Any], source: str) -> None:
    if payload.get("schema_name") != FASTWAM_DEXJOCO_PROTOCOL_SCHEMA:
        raise ValueError(f"{source} uses an incompatible schema_name.")
    if payload.get("schema_version") != FASTWAM_DEXJOCO_PROTOCOL_VERSION:
        raise ValueError(f"{source} uses an incompatible schema_version.")
    if payload.get("action_dim") != DEXJOCO_ACTION_DIM:
        raise ValueError(f"{source} must declare action_dim={DEXJOCO_ACTION_DIM}.")
    if payload.get("proprio_dim") != DEXJOCO_PROPRIO_DIM:
        raise ValueError(f"{source} must declare proprio_dim={DEXJOCO_PROPRIO_DIM}.")


def validate_inference_request(
    payload: Mapping[str, Any],
    *,
    expected_horizon: int,
) -> DexJoCoInferenceRequest:
    if not isinstance(payload, Mapping):
        raise TypeError("DexJoCo inference request must be a mapping.")
    _validate_schema(payload, "DexJoCo request")
    required = {"primary", "wrist", "state", "prompt", "horizon"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"DexJoCo request misses fields: {missing}.")
    horizon = payload["horizon"]
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
        raise ValueError("DexJoCo request horizon must be a positive integer.")
    if horizon != expected_horizon:
        raise ValueError(
            f"DexJoCo request horizon must equal server horizon {expected_horizon}, got {horizon}."
        )

    primary = np.asarray(payload["primary"])
    wrist = np.asarray(payload["wrist"])
    for name, image in (("primary", primary), ("wrist", wrist)):
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"DexJoCo {name} image must use unbatched HWC RGB layout.")
        if image.dtype != np.uint8:
            raise TypeError(f"DexJoCo {name} image must use uint8 dtype, got {image.dtype}.")
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ValueError(f"DexJoCo {name} image has an empty spatial dimension.")
    if primary.shape != wrist.shape:
        raise ValueError(
            "DexJoCo primary and wrist images must have identical HWC shapes, "
            f"got {primary.shape} and {wrist.shape}."
        )

    state = np.asarray(payload["state"])
    if state.shape != (DEXJOCO_PROPRIO_DIM,):
        raise ValueError(
            f"DexJoCo state must have shape ({DEXJOCO_PROPRIO_DIM},), got {state.shape}."
        )
    if state.dtype.kind != "f":
        raise TypeError(f"DexJoCo state must use a floating dtype, got {state.dtype}.")
    state = state.astype(np.float32, copy=False)
    if not np.isfinite(state).all():
        raise ValueError("DexJoCo state contains NaN or Inf.")

    prompt = payload["prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("DexJoCo prompt must be a non-empty string.")
    return DexJoCoInferenceRequest(
        primary=np.ascontiguousarray(primary),
        wrist=np.ascontiguousarray(wrist),
        state=np.ascontiguousarray(state),
        prompt=prompt.strip(),
        horizon=horizon,
    )


def build_action_response(actions: Any, *, horizon: int) -> dict[str, Any]:
    array = np.asarray(actions)
    expected_shape = (horizon, DEXJOCO_ACTION_DIM)
    if array.shape != expected_shape:
        raise ValueError(
            f"FastWAM action response must have shape {expected_shape}, got {array.shape}."
        )
    if array.dtype.kind != "f":
        raise TypeError(f"FastWAM action response must use a floating dtype, got {array.dtype}.")
    array = np.ascontiguousarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("FastWAM action response contains NaN or Inf.")
    response = protocol_metadata(horizon=horizon)
    response["actions"] = array
    return response
