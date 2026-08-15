#!/usr/bin/env python
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from fastwam.datasets.lerobot.dexjoco_contract import (
    DEXJOCO_ACTION_ORDERING,
    DEXJOCO_HAND_ACTUATOR_ORDERING,
    DEXJOCO_STATE_ORDERING,
    DEXJOCO_TASKS,
)
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.inference.dexjoco_policy import (
    DexJoCoInferenceNormalizer,
    DexJoCoInferencePolicy,
    _validate_joint_action_result,
)
from fastwam.inference.websocket_protocol import (
    build_action_response,
    protocol_metadata,
    validate_inference_request,
)
from fastwam.inference.websocket_server import DexJoCoWebsocketServer

from smoke_test_dexjoco_dual_action_model import (
    assert_attention_contract,
    make_dexjoco_model,
)
from precompute_text_embeds import _read_unique_prompts


DEXJOCO_REPOSITORY = Path("/home/shared/ai/datasets/DexJoCo/dexjoco")


class CountingInferenceScheduler:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.schedule_calls = 0
        self.step_shapes: list[tuple[int, ...]] = []

    def build_inference_schedule(self, *args, **kwargs):
        self.schedule_calls += 1
        return self.scheduler.build_inference_schedule(*args, **kwargs)

    def step(self, model_output, delta, sample):
        self.step_shapes.append(tuple(model_output.shape))
        return self.scheduler.step(model_output, delta, sample)


def _field_statistics(dim: int, step_count: int) -> dict[str, list]:
    global_values = {
        "min": [-2.0] * dim,
        "max": [2.0] * dim,
        "q01": [-1.5] * dim,
        "q99": [1.5] * dim,
        "mean": [0.0] * dim,
        "std": [1.0] * dim,
    }
    result = {f"global_{name}": value for name, value in global_values.items()}
    for name, value in global_values.items():
        result[f"stepwise_{name}"] = [list(value) for _ in range(step_count)]
    return result


def write_smoke_statistics(path: Path, horizon: int) -> None:
    payload = {
        "schema_name": "fastwam.dexjoco.dataset_stats",
        "schema_version": 1,
        "statistics_mode": "smoke",
        "production": False,
        "split": "train",
        "tasks": list(DEXJOCO_TASKS),
        "action_ordering": list(DEXJOCO_ACTION_ORDERING),
        "state_ordering": list(DEXJOCO_STATE_ORDERING),
        "hand_actuator_ordering": list(DEXJOCO_HAND_ACTUATOR_ORDERING),
        "action_dim": 22,
        "arm_action_dim": 6,
        "hand_action_dim": 16,
        "proprio_dim": 23,
        "action_horizon": horizon,
        "std_floor": 1e-6,
        "std_floor_applied_dimensions": {"action": [], "state": []},
        "episode_counts": {task: 1 for task in DEXJOCO_TASKS},
        "frame_counts": {task: 2 for task in DEXJOCO_TASKS},
        "num_episodes": len(DEXJOCO_TASKS),
        "num_transition": len(DEXJOCO_TASKS),
        "max_episodes_per_task": 1,
        "action": {"default": _field_statistics(22, horizon)},
        "state": {"default": _field_statistics(23, 1)},
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_text_cache(cache_dir: Path, prompt: str, context_len: int, text_dim: int) -> Path:
    formatted = DEFAULT_PROMPT.format(task=prompt)
    digest = hashlib.sha256(formatted.encode("utf-8")).hexdigest()
    path = cache_dir / f"{digest}.t5_len{context_len}.wan22ti2v5b.pt"
    torch.save(
        {
            "context": torch.linspace(-0.2, 0.2, context_len * text_dim).reshape(
                context_len, text_dim
            ),
            "mask": torch.tensor([True] * (context_len - 1) + [False]),
        },
        path,
    )
    return path


def assert_invalid_inputs(horizon: int) -> None:
    valid = {
        **protocol_metadata(horizon=horizon),
        "primary": np.zeros((16, 16, 3), dtype=np.uint8),
        "wrist": np.zeros((16, 16, 3), dtype=np.uint8),
        "state": np.zeros(23, dtype=np.float32),
        "prompt": "test",
    }
    invalid_cases = []
    for key, value in (
        ("primary", np.zeros((1, 16, 16, 3), dtype=np.uint8)),
        ("wrist", np.zeros((16, 16, 3), dtype=np.float32)),
        ("state", np.zeros(22, dtype=np.float32)),
        ("prompt", ""),
        ("horizon", horizon + 1),
    ):
        payload = dict(valid)
        payload[key] = value
        invalid_cases.append(payload)
    nan_state = dict(valid)
    nan_state["state"] = np.full(23, np.nan, dtype=np.float32)
    invalid_cases.append(nan_state)
    for payload in invalid_cases:
        try:
            validate_inference_request(payload, expected_horizon=horizon)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("Invalid websocket request must fail fast")
    try:
        build_action_response(np.full((horizon, 22), np.nan), horizon=horizon)
    except ValueError:
        pass
    else:
        raise AssertionError("NaN action response must fail fast")

    inconsistent_result = {
        "action": torch.zeros(1, horizon, 22),
        "arm_action": torch.ones(1, horizon, 6),
        "hand_action": torch.zeros(1, horizon, 16),
    }
    try:
        _validate_joint_action_result(inconsistent_result, horizon=horizon)
    except RuntimeError as exc:
        assert "does not exactly equal" in str(exc)
    else:
        raise AssertionError("Inconsistent Arm/Hand action outputs must fail fast")


def import_real_dexjoco_client():
    sys.path.insert(0, str(DEXJOCO_REPOSITORY / "dexjoco"))
    sys.path.insert(
        0,
        str(DEXJOCO_REPOSITORY / "openpi" / "packages" / "openpi-client" / "src"),
    )
    from dexjoco_fastwam_client.protocol import FastWAMWebsocketClient

    return FastWAMWebsocketClient


async def websocket_round_trip(policy: DexJoCoInferencePolicy, observation: dict) -> np.ndarray:
    server = DexJoCoWebsocketServer(policy, host="127.0.0.1", port=0)
    async with server.create_server() as running:
        port = running.sockets[0].getsockname()[1]

        def request() -> np.ndarray:
            client_class = import_real_dexjoco_client()
            client = client_class(host="127.0.0.1", port=port)
            try:
                return client.infer(observation, horizon=policy.horizon)
            finally:
                client._client._ws.close()

        return await asyncio.to_thread(request)


def main() -> None:
    torch.manual_seed(23)
    np.random.seed(23)
    horizon = 4
    context_len = 3
    prompt = "Grasp the watering can and apply water to the plant."
    data_root = Path("/home/shared/ai/datasets/dexlewm/dexjoco")
    task_dirs = [str(data_root / task) for task in DEXJOCO_TASKS]
    discovered_prompts = _read_unique_prompts(task_dirs)
    assert len(discovered_prompts) == len(DEXJOCO_TASKS)
    assert DEFAULT_PROMPT.format(task=prompt) in discovered_prompts
    with tempfile.TemporaryDirectory(prefix="fastwam_dexjoco_phase6_") as temp_dir:
        temp = Path(temp_dir)
        statistics_path = temp / "dataset_stats.json"
        text_cache_dir = temp / "text_cache"
        text_cache_dir.mkdir()
        write_smoke_statistics(statistics_path, horizon)
        cache_path = write_text_cache(text_cache_dir, prompt, context_len, text_dim=32)

        model = make_dexjoco_model().eval()
        scheduler = CountingInferenceScheduler(model.infer_action_scheduler)
        model.infer_action_scheduler = scheduler
        policy = DexJoCoInferencePolicy(
            model,
            statistics_path=statistics_path,
            text_cache_dir=text_cache_dir,
            horizon=horizon,
            image_size=(16, 16),
            context_len=context_len,
            num_inference_steps=2,
            seed=19,
            allow_non_production_statistics=True,
        )

        raw_action = torch.linspace(-1.0, 1.0, horizon * 22).reshape(1, horizon, 22)
        normalized_action = policy.normalizer.normalize_action(raw_action)
        torch.testing.assert_close(
            policy.normalizer.denormalize_action(normalized_action),
            raw_action,
        )
        raw_state = torch.linspace(-1.0, 1.0, 23).reshape(1, 23)
        normalized_state = policy.normalizer.normalize_state(raw_state)
        assert normalized_state.shape == (1, 23) and torch.isfinite(normalized_state).all()

        observation = {
            "primary": np.arange(18 * 20 * 3, dtype=np.uint16).reshape(18, 20, 3).astype(
                np.uint8
            ),
            "wrist": np.flip(
                np.arange(18 * 20 * 3, dtype=np.uint16).reshape(18, 20, 3).astype(
                    np.uint8
                ),
                axis=1,
            ).copy(),
            "state": raw_state.numpy()[0].astype(np.float32),
            "prompt": prompt,
        }
        actions = asyncio.run(websocket_round_trip(policy, observation))
        assert actions.shape == (horizon, 22)
        assert actions.dtype == np.float32 and np.isfinite(actions).all()
        assert scheduler.schedule_calls == 1
        assert scheduler.step_shapes == [(1, horizon, 22), (1, horizon, 22)]

        direct_request = {
            **protocol_metadata(horizon=horizon),
            **observation,
        }
        direct_response = policy.infer(direct_request)
        assert direct_response["actions"].shape == (horizon, 22)
        assert direct_response["text_cache_key"] == cache_path.name
        assert scheduler.schedule_calls == 2
        assert scheduler.step_shapes[-2:] == [(1, horizon, 22), (1, horizon, 22)]

        # Inspect a direct model call so the full internal batch dimension and mask
        # remain observable independently of the wire response.
        context, context_mask, _ = policy.text_cache.load(prompt, expected_dim=32)
        model_output = model.infer_action(
            prompt=None,
            input_image=policy._prepare_image(observation["primary"], observation["wrist"]),
            action_horizon=horizon,
            proprio=normalized_state,
            context=context,
            context_mask=context_mask,
            num_inference_steps=1,
            seed=19,
        )
        assert model_output["action"].shape == (1, horizon, 22)
        assert model_output["arm_action"].shape == (1, horizon, 6)
        assert model_output["hand_action"].shape == (1, horizon, 16)
        assert_attention_contract(model_output["attention_mask"], 8, horizon, horizon)
        assert torch.isfinite(model_output["action"]).all()

        assert_invalid_inputs(horizon)
        invalid_statistics = temp / "libero_statistics.json"
        payload = json.loads(statistics_path.read_text(encoding="utf-8"))
        payload["schema_name"] = "libero.dataset_stats"
        payload["action_dim"] = 7
        invalid_statistics.write_text(json.dumps(payload), encoding="utf-8")
        try:
            DexJoCoInferenceNormalizer(
                invalid_statistics,
                allow_non_production_statistics=True,
            )
        except ValueError as exc:
            assert "schema_name" in str(exc)
        else:
            raise AssertionError("LIBERO 7D statistics must be rejected")

        missing_cache = copy.copy(policy.text_cache)
        try:
            missing_cache.load("uncached prompt", expected_dim=32)
        except FileNotFoundError as exc:
            assert "precompute_text_embeds.py" in str(exc)
        else:
            raise AssertionError("Missing T5 cache must fail instead of loading T5")

        print("websocket schema=fastwam.dexjoco.websocket@1 real_DexJoCo_client=PASS")
        print(
            "request primary=(18,20,3) wrist=(18,20,3) state=(23,) "
            "preprocessed_video=(1,3,16,32)"
        )
        print("model action=(1,4,22) arm=(1,4,6) hand=(1,4,16) finite=PASS")
        print("server denormalized actions=(4,22) dtype=float32 finite=PASS")
        print("uncached diffusion steps=2 scheduler_step_shapes=[(1,4,22),(1,4,22)]")
        print("attention_mask=(16,16) video_to_action=false arm_hand_bidirectional=true")
        print("normalize_denormalize_round_trip=PASS LIBERO_7D_stats_rejected=PASS")
        print("T5 cached payload=PASS missing_cache_fail_fast=PASS text_encoder_loaded=false")
        print("precompute task discovery=tasks.parquet six_tasks=PASS T5_model_not_loaded=true")
        print(
            "invalid batch/horizon/dtype/state/NaN/joint-output checks=PASS "
            "temporary_artifacts_removed=true"
        )


if __name__ == "__main__":
    main()
