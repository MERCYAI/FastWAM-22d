"""DexJoCo preprocessing, cached text conditioning, and 22D policy inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

from fastwam.datasets.lerobot.dexjoco_contract import (
    DEXJOCO_ACTION_DIM,
    DEXJOCO_PROPRIO_DIM,
    validate_dexjoco_statistics,
)
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import (
    LinearNormalizer,
    load_dataset_stats_from_json,
)
from fastwam.models.wan22.dexjoco_dual_action import DexJoCoDualActionFastWAM
from fastwam.trainer import (
    DEXJOCO_DATASET_STATS,
    DEXJOCO_TRAINING_CONFIG,
    validate_dexjoco_training_checkpoint,
)
from fastwam.utils.logging_config import get_logger

from .websocket_protocol import (
    build_action_response,
    protocol_metadata,
    validate_inference_request,
)


logger = get_logger(__name__)
DEFAULT_TEXT_ENCODER_ID = "wan22ti2v5b"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing DexJoCo dataset statistics: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"DexJoCo dataset statistics must be a JSON object: {path}")
    return payload


def _validate_joint_action_result(
    result: Mapping[str, Any],
    *,
    horizon: int,
) -> torch.Tensor:
    if not isinstance(result, Mapping):
        raise TypeError("DexJoCo model inference result must be a mapping.")
    action = result.get("action")
    arm_action = result.get("arm_action")
    hand_action = result.get("hand_action")
    expected_shapes = {
        "action": (1, horizon, 22),
        "arm_action": (1, horizon, 6),
        "hand_action": (1, horizon, 16),
    }
    for name, tensor in (
        ("action", action),
        ("arm_action", arm_action),
        ("hand_action", hand_action),
    ):
        if not isinstance(tensor, torch.Tensor) or tensor.shape != expected_shapes[name]:
            shape = None if not isinstance(tensor, torch.Tensor) else tuple(tensor.shape)
            raise RuntimeError(
                f"DexJoCo internal {name} must have shape {expected_shapes[name]}, got {shape}."
            )
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"DexJoCo internal {name} contains NaN or Inf.")

    combined = torch.cat([arm_action, hand_action], dim=-1)
    if not torch.equal(action, combined):
        max_abs_error = float((action - combined).abs().max().detach().cpu())
        raise RuntimeError(
            "DexJoCo internal 22D action does not exactly equal [arm_6d, hand_16d]; "
            f"max_abs_error={max_abs_error:.8g}."
        )
    return action


def _normalizer_shape_meta() -> dict[str, list[dict[str, Any]]]:
    return {
        "images": [
            {"key": "primary", "raw_shape": [3, 1, 1], "shape": [3, 1, 1]},
            {"key": "wrist", "raw_shape": [3, 1, 1], "shape": [3, 1, 1]},
        ],
        "action": [
            {"key": "default", "raw_shape": DEXJOCO_ACTION_DIM, "shape": DEXJOCO_ACTION_DIM}
        ],
        "state": [
            {"key": "default", "raw_shape": DEXJOCO_PROPRIO_DIM, "shape": DEXJOCO_PROPRIO_DIM}
        ],
    }


class DexJoCoInferenceNormalizer:
    """Use the training normalizer while exposing only the 22D/23D inference views."""

    def __init__(
        self,
        statistics_path: str | Path,
        *,
        allow_non_production_statistics: bool = False,
    ) -> None:
        self.statistics_path = Path(statistics_path).expanduser().resolve()
        raw_statistics = _read_json(self.statistics_path)
        validate_dexjoco_statistics(
            raw_statistics,
            require_production=not allow_non_production_statistics,
        )
        statistics = load_dataset_stats_from_json(str(self.statistics_path))
        self.statistics = raw_statistics
        self.action_horizon = int(raw_statistics["action_horizon"])
        self.normalizer = LinearNormalizer(
            shape_meta=_normalizer_shape_meta(),
            use_stepwise_action_norm=False,
            default_mode="min/max",
            exception_mode=None,
            stats=statistics,
        )

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        if state.shape != (1, DEXJOCO_PROPRIO_DIM):
            raise ValueError(
                f"DexJoCo state tensor must be [1,{DEXJOCO_PROPRIO_DIM}], got {tuple(state.shape)}."
            )
        output = self.normalizer.normalizers["state"]["default"].forward(state)
        if not torch.isfinite(output).all():
            raise ValueError("Normalized DexJoCo state contains NaN or Inf.")
        return output

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        if action.ndim != 3 or action.shape[0] != 1 or action.shape[-1] != DEXJOCO_ACTION_DIM:
            raise ValueError(
                "DexJoCo action tensor must be [1,T,22], "
                f"got {tuple(action.shape)}."
            )
        output = self.normalizer.normalizers["action"]["default"].forward(action)
        if not torch.isfinite(output).all():
            raise ValueError("Normalized DexJoCo action contains NaN or Inf.")
        return output

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        if action.ndim != 3 or action.shape[0] != 1 or action.shape[-1] != DEXJOCO_ACTION_DIM:
            raise ValueError(
                "DexJoCo normalized action tensor must be [1,T,22], "
                f"got {tuple(action.shape)}."
            )
        output = self.normalizer.normalizers["action"]["default"].backward(action)
        if not torch.isfinite(output).all():
            raise ValueError("Denormalized DexJoCo action contains NaN or Inf.")
        return output


class DexJoCoT5Cache:
    """Read the exact cache format emitted by ``scripts/precompute_text_embeds.py``."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        context_len: int = 128,
        encoder_id: str = DEFAULT_TEXT_ENCODER_ID,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        if not self.cache_dir.is_dir():
            raise FileNotFoundError(f"DexJoCo T5 cache directory does not exist: {self.cache_dir}")
        if not isinstance(context_len, int) or context_len <= 0:
            raise ValueError("T5 context_len must be a positive integer.")
        if not isinstance(encoder_id, str) or not encoder_id.strip():
            raise ValueError("T5 encoder_id must be non-empty.")
        self.context_len = context_len
        self.encoder_id = encoder_id.strip()

    @staticmethod
    def format_prompt(prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("DexJoCo prompt must be a non-empty string.")
        prompt = prompt.strip()
        prefix = DEFAULT_PROMPT.split("{task}", maxsplit=1)[0]
        return prompt if prompt.startswith(prefix) else DEFAULT_PROMPT.format(task=prompt)

    def path_for_prompt(self, prompt: str) -> Path:
        formatted = self.format_prompt(prompt)
        digest = hashlib.sha256(formatted.encode("utf-8")).hexdigest()
        return self.cache_dir / (
            f"{digest}.t5_len{self.context_len}.{self.encoder_id}.pt"
        )

    def load(self, prompt: str, *, expected_dim: int) -> tuple[torch.Tensor, torch.Tensor, Path]:
        path = self.path_for_prompt(prompt)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing cached T5 embedding: {path}. "
                "Run scripts/precompute_text_embeds.py for this exact prompt first."
            )
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping) or set(payload) < {"context", "mask"}:
            raise ValueError(f"Invalid cached T5 payload: {path}")
        context = payload["context"]
        mask = payload["mask"]
        if not isinstance(context, torch.Tensor) or not isinstance(mask, torch.Tensor):
            raise TypeError(f"Cached T5 context and mask must be tensors: {path}")
        if context.shape != (self.context_len, expected_dim):
            raise ValueError(
                "Cached T5 context shape mismatch: "
                f"expected {(self.context_len, expected_dim)}, got {tuple(context.shape)} in {path}."
            )
        if mask.shape != (self.context_len,):
            raise ValueError(
                f"Cached T5 mask must have shape ({self.context_len},), "
                f"got {tuple(mask.shape)} in {path}."
            )
        if not context.is_floating_point() or not torch.isfinite(context).all():
            raise ValueError(f"Cached T5 context must be finite floating point: {path}")

        context = context.clone()
        mask = mask.to(dtype=torch.bool)
        context[~mask] = 0
        # Match the training processor and Wan prompt encoder behavior.
        model_mask = torch.ones_like(mask, dtype=torch.bool)
        return context.unsqueeze(0), model_mask.unsqueeze(0), path


def _torch_dtype(name: str | torch.dtype) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    values = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return values[str(name).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported model dtype: {name!r}.") from exc


def _load_accelerate_model_state(checkpoint_dir: Path, manifest: Mapping[str, Any]) -> dict:
    state_files = manifest["artifacts"]["accelerate_state_files"]
    candidates = [
        checkpoint_dir / path
        for path in state_files
        if Path(path).name.startswith("pytorch_model") and Path(path).suffix == ".bin"
    ]
    if len(candidates) != 1:
        raise ValueError(
            "DexJoCo inference requires exactly one Accelerate PyTorch model state file; "
            f"found {[str(path) for path in candidates]}."
        )
    state = torch.load(candidates[0], map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Invalid Accelerate model state: {candidates[0]}")
    if all(isinstance(key, str) and key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def load_dexjoco_inference_model(
    checkpoint_dir: str | Path,
    *,
    device: str = "cuda",
    dtype: str | torch.dtype = "bfloat16",
    allow_non_production_statistics: bool = False,
) -> DexJoCoDualActionFastWAM:
    """Instantiate and strictly load a versioned Phase 5 training state directory."""
    root = Path(checkpoint_dir).expanduser().resolve()
    config_path = root / DEXJOCO_TRAINING_CONFIG
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Expected a Phase 5 DexJoCo state directory containing {DEXJOCO_TRAINING_CONFIG}: {root}"
        )
    config = OmegaConf.load(config_path)
    if OmegaConf.select(config, "model._target_") != "fastwam.runtime.create_dexjoco_dual_action_fastwam":
        raise ValueError("Checkpoint training config is not a DexJoCo dual-action model.")
    model_config = OmegaConf.create(OmegaConf.to_container(config.model, resolve=True))
    with open_dict(model_config):
        model_config.load_text_encoder = False
        model_config.skip_dit_load_from_pretrain = True
        model_config.selective_checkpoint_path = None
        model_config.selective_checkpoint_report_path = None
        model_config.action_dit_pretrained_path = None
    model = instantiate(
        model_config,
        model_dtype=_torch_dtype(dtype),
        device=device,
    )
    if not isinstance(model, DexJoCoDualActionFastWAM):
        raise TypeError(f"Expected DexJoCoDualActionFastWAM, got {type(model).__name__}.")
    manifest = validate_dexjoco_training_checkpoint(
        root,
        model,
        require_production_statistics=not allow_non_production_statistics,
    )
    state = _load_accelerate_model_state(root, manifest)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "DexJoCo inference checkpoint state mismatch: "
            f"missing={incompatible.missing_keys[:20]}, "
            f"unexpected={incompatible.unexpected_keys[:20]}."
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    logger.info(
        "Loaded DexJoCo inference checkpoint %s on %s (%s); tensors=%d",
        root,
        model.device,
        model.torch_dtype,
        len(state),
    )
    return model


class DexJoCoInferencePolicy:
    """Strict one-observation policy used by the websocket server."""

    def __init__(
        self,
        model: DexJoCoDualActionFastWAM,
        *,
        statistics_path: str | Path,
        text_cache_dir: str | Path,
        horizon: int,
        image_size: tuple[int, int] = (224, 224),
        context_len: int = 128,
        text_encoder_id: str = DEFAULT_TEXT_ENCODER_ID,
        num_inference_steps: int = 20,
        sigma_shift: float | None = None,
        seed: int | None = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        allow_non_production_statistics: bool = False,
    ) -> None:
        if not isinstance(model, DexJoCoDualActionFastWAM):
            raise TypeError("DexJoCoInferencePolicy requires DexJoCoDualActionFastWAM.")
        dimensions = (
            model.action_dim,
            model.arm_action_dim,
            model.hand_action_dim,
            model.proprio_dim,
        )
        if dimensions != (22, 6, 16, 23):
            raise ValueError(f"DexJoCo model dimensions must be (22,6,16,23), got {dimensions}.")
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
            raise ValueError("DexJoCo server horizon must be a positive integer.")
        if len(image_size) != 2 or any(int(size) <= 0 or int(size) % 16 for size in image_size):
            raise ValueError("DexJoCo server image height/width must be positive multiples of 16.")
        if not isinstance(num_inference_steps, int) or num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be a positive integer.")

        self.model = model.eval()
        self.horizon = horizon
        self.image_size = tuple(int(size) for size in image_size)
        self.num_inference_steps = num_inference_steps
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.rand_device = rand_device
        self.tiled = tiled
        self.normalizer = DexJoCoInferenceNormalizer(
            statistics_path,
            allow_non_production_statistics=allow_non_production_statistics,
        )
        if self.normalizer.action_horizon != self.horizon:
            raise ValueError(
                "Server horizon must match checkpoint statistics action_horizon: "
                f"server={self.horizon}, statistics={self.normalizer.action_horizon}."
            )
        self.text_cache = DexJoCoT5Cache(
            text_cache_dir,
            context_len=context_len,
            encoder_id=text_encoder_id,
        )
        logger.info(
            "DexJoCo policy ready: device=%s dtype=%s horizon=%d image=%s stats=%s "
            "production=%s text_cache=%s cache_mode=disabled",
            model.device,
            model.torch_dtype,
            horizon,
            self.image_size,
            self.normalizer.statistics_path,
            self.normalizer.statistics["production"],
            self.text_cache.cache_dir,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = protocol_metadata(horizon=self.horizon)
        metadata.update(
            {
                "model_dtype": str(self.model.torch_dtype),
                "model_device": str(self.model.device),
                "statistics_schema": self.normalizer.statistics["schema_name"],
                "statistics_version": self.normalizer.statistics["schema_version"],
                "statistics_production": self.normalizer.statistics["production"],
                "video_kv_cache": False,
            }
        )
        return metadata

    def _prepare_image(self, primary: np.ndarray, wrist: np.ndarray) -> torch.Tensor:
        cameras = np.stack([primary, wrist], axis=0)
        tensor = torch.from_numpy(cameras.copy()).permute(0, 3, 1, 2).float()
        tensor = F.interpolate(
            tensor,
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        tensor = tensor.mul(2.0 / 255.0).sub(1.0)
        concatenated = torch.cat([tensor[0], tensor[1]], dim=2).unsqueeze(0)
        expected = (1, 3, self.image_size[0], self.image_size[1] * 2)
        if concatenated.shape != expected or not torch.isfinite(concatenated).all():
            raise RuntimeError(
                f"DexJoCo camera preprocessing must produce finite {expected}, "
                f"got {tuple(concatenated.shape)}."
            )
        return concatenated

    @torch.inference_mode()
    def infer(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = validate_inference_request(payload, expected_horizon=self.horizon)
        input_image = self._prepare_image(request.primary, request.wrist)
        state = torch.from_numpy(request.state.copy()).reshape(1, DEXJOCO_PROPRIO_DIM)
        normalized_state = self.normalizer.normalize_state(state)
        context, context_mask, cache_path = self.text_cache.load(
            request.prompt,
            expected_dim=self.model.text_dim,
        )
        result = self.model.infer_action(
            prompt=None,
            input_image=input_image,
            action_horizon=self.horizon,
            proprio=normalized_state,
            context=context,
            context_mask=context_mask,
            num_inference_steps=self.num_inference_steps,
            sigma_shift=self.sigma_shift,
            seed=self.seed,
            rand_device=self.rand_device,
            tiled=self.tiled,
        )
        action = _validate_joint_action_result(result, horizon=self.horizon)
        denormalized = self.normalizer.denormalize_action(action.float())
        response = build_action_response(denormalized[0].cpu().numpy(), horizon=self.horizon)
        response["text_cache_key"] = cache_path.name
        return response


def checkpoint_statistics_path(checkpoint_dir: str | Path) -> Path:
    return Path(checkpoint_dir).expanduser().resolve() / DEXJOCO_DATASET_STATS
