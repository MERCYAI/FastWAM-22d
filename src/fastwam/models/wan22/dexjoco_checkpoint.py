from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from fastwam.utils.logging_config import get_logger

from .helpers.io import load_state_dict

logger = get_logger(__name__)


REPORT_SCHEMA = "fastwam.dexjoco_selective_checkpoint"
REPORT_VERSION = 1
REPORT_CATEGORIES = (
    "loaded",
    "copied_to_hand",
    "skipped_shape",
    "skipped_policy",
    "missing_in_checkpoint",
    "unexpected_in_checkpoint",
    "newly_initialized",
)

_COMMON_CONTAINER_PREFIXES = (
    "module.",
    "_orig_mod.",
    "model.",
    "policy.",
    "network.",
)
_AUTO_SOURCE_ROOTS = (
    ("mot.mixtures.video.", "video"),
    ("mot.mixtures.action.", "action"),
    ("dit.mixtures.video.", "video"),
    ("dit.mixtures.action.", "action"),
    ("mixtures.video.", "video"),
    ("mixtures.action.", "action"),
    ("experts.video.", "video"),
    ("experts.action.", "action"),
    ("video_expert.", "video"),
    ("action_expert.", "action"),
    ("video.", "video"),
    ("action.", "action"),
    ("dit.", "video"),
    ("proprio_encoder.", "proprio_encoder"),
)
_POLICY_LOCAL_KEYS = (
    "action_encoder.weight",
    "action_encoder.bias",
    "head.weight",
    "head.bias",
)


class SelectiveCheckpointError(RuntimeError):
    def __init__(self, message: str, report: "SelectiveCheckpointReport"):
        super().__init__(message)
        self.report = report


class SelectiveCheckpointReport:
    def __init__(
        self,
        *,
        checkpoint: str,
        checkpoint_format: str,
        requested_apply: bool,
    ):
        self.checkpoint = checkpoint
        self.checkpoint_format = checkpoint_format
        self.requested_apply = bool(requested_apply)
        self.applied = False
        self.categories: dict[str, list[dict[str, Any]]] = {
            category: [] for category in REPORT_CATEGORIES
        }

    def add(
        self,
        category: str,
        *,
        target: str | None,
        target_module: str | None,
        source: str | None,
        checkpoint_shape: tuple[int, ...] | None,
        model_shape: tuple[int, ...] | None,
        reason: str,
        initialization: str | None = None,
        initialization_applied: bool | None = None,
    ) -> None:
        if category not in self.categories:
            raise ValueError(f"Unknown checkpoint report category: {category}")
        entry = {
            "target": target,
            "target_module": target_module,
            "source": source,
            "checkpoint_shape": (
                None if checkpoint_shape is None else list(checkpoint_shape)
            ),
            "model_shape": None if model_shape is None else list(model_shape),
            "reason": reason,
        }
        if initialization is not None:
            entry["initialization"] = initialization
        if initialization_applied is not None:
            entry["initialization_applied"] = bool(initialization_applied)
        self.categories[category].append(entry)

    @property
    def summary(self) -> dict[str, int]:
        return {
            category: len(self.categories[category])
            for category in REPORT_CATEGORIES
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REPORT_SCHEMA,
            "version": REPORT_VERSION,
            "checkpoint": self.checkpoint,
            "checkpoint_format": self.checkpoint_format,
            "requested_apply": self.requested_apply,
            "applied": self.applied,
            "prefix_policy": {
                "common_container_prefixes": list(_COMMON_CONTAINER_PREFIXES),
                "matching": "exact_root_and_exact_local_key_only",
                "fuzzy_suffix_matching": False,
            },
            "initialization_policy": {
                "action.action_encoder": "torch.nn.Linear.reset_parameters",
                "action.head": "torch.nn.Linear.reset_parameters",
                "hand.action_encoder": "torch.nn.Linear.reset_parameters",
                "hand.head": "torch.nn.Linear.reset_parameters",
                "proprio_encoder": "torch.nn.Linear.reset_parameters",
                "partial_projection_copy": False,
                "projection_interpolation": False,
            },
            "summary": self.summary,
            "categories": self.categories,
        }

    def write_json(self, path: str | Path) -> Path:
        output_path = Path(path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, ensure_ascii=True, indent=2)
            file.write("\n")
        return output_path

    def log(self) -> None:
        summary = ", ".join(
            f"{category}={self.summary[category]}" for category in REPORT_CATEGORIES
        )
        logger.info(
            "DexJoCo selective checkpoint summary: checkpoint=%s format=%s applied=%s %s",
            self.checkpoint,
            self.checkpoint_format,
            self.applied,
            summary,
        )
        for category in REPORT_CATEGORIES:
            entries = self.categories[category]
            logger.info("DexJoCo checkpoint category=%s count=%d", category, len(entries))
            for entry in entries:
                logger.info(
                    "[%s] target=%s module=%s source=%s checkpoint_shape=%s "
                    "model_shape=%s reason=%s initialization=%s initialization_applied=%s",
                    category,
                    entry.get("target"),
                    entry.get("target_module"),
                    entry.get("source"),
                    entry.get("checkpoint_shape"),
                    entry.get("model_shape"),
                    entry.get("reason"),
                    entry.get("initialization"),
                    entry.get("initialization_applied"),
                )


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _tensor_entries(
    state: Mapping[str, Any],
    *,
    scope: str,
) -> list[tuple[str, torch.Tensor, str]]:
    return [
        (str(key), value, scope)
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    ]


def _extract_checkpoint_entries(
    checkpoint: Mapping[str, Any],
) -> tuple[list[tuple[str, torch.Tensor, str]], str]:
    payload: Mapping[str, Any] = checkpoint
    wrappers: list[str] = []
    while True:
        wrapper = next(
            (
                name
                for name in (
                    "state_dict",
                    "model_state_dict",
                    "model_state",
                    "model",
                    "module",
                    "policy",
                )
                if name in payload and _is_mapping(payload[name])
            ),
            None,
        )
        if wrapper is None:
            break
        wrappers.append(wrapper)
        payload = payload[wrapper]

    entries: list[tuple[str, torch.Tensor, str]] = []
    checkpoint_format = "flat_state_dict"
    if "mot" in payload and _is_mapping(payload["mot"]):
        checkpoint_format = "fastwam_mot_payload"
        entries.extend(_tensor_entries(payload["mot"], scope="auto"))
        proprio = payload.get("proprio_encoder")
        if _is_mapping(proprio):
            entries.extend(
                (
                    f"proprio_encoder.{key}",
                    value,
                    "auto",
                )
                for key, value in proprio.items()
                if isinstance(value, torch.Tensor)
            )
    elif "dit" in payload and _is_mapping(payload["dit"]):
        checkpoint_format = "legacy_video_dit_payload"
        entries.extend(_tensor_entries(payload["dit"], scope="video"))
    elif "backbone_state_dict" in payload and _is_mapping(payload["backbone_state_dict"]):
        checkpoint_format = "action_backbone_payload"
        entries.extend(_tensor_entries(payload["backbone_state_dict"], scope="action"))
    else:
        entries.extend(_tensor_entries(payload, scope="auto"))

    if wrappers:
        checkpoint_format += ":" + ".".join(wrappers)
    if not entries:
        raise ValueError("Checkpoint contains no tensor state_dict entries.")
    return entries, checkpoint_format


def _load_checkpoint_payload(path: str | Path) -> Mapping[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Selective checkpoint not found: {checkpoint_path}")
    if checkpoint_path.suffix == ".safetensors":
        return load_state_dict(str(checkpoint_path), device="cpu")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not _is_mapping(payload):
        raise ValueError(
            f"Selective checkpoint must contain a mapping, got {type(payload).__name__}: "
            f"{checkpoint_path}"
        )
    return payload


def _strip_common_prefixes(key: str) -> str:
    normalized = key
    while True:
        matched = next(
            (prefix for prefix in _COMMON_CONTAINER_PREFIXES if normalized.startswith(prefix)),
            None,
        )
        if matched is None:
            return normalized
        normalized = normalized[len(matched) :]


def _canonical_source_key(key: str, scope: str) -> tuple[str, str] | None:
    normalized = _strip_common_prefixes(key)
    if scope in ("video", "action"):
        scoped_prefixes = {
            "video": ("dit.", "video_expert.", "video."),
            "action": ("action_expert.", "action."),
        }
        for prefix in scoped_prefixes[scope]:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                break
        return scope, normalized

    for prefix, module_name in _AUTO_SOURCE_ROOTS:
        if normalized.startswith(prefix):
            return module_name, normalized[len(prefix) :]
    return None


def _shape(tensor: torch.Tensor | None) -> tuple[int, ...] | None:
    return None if tensor is None else tuple(int(dim) for dim in tensor.shape)


def _copy_tensor(target: torch.Tensor, source: torch.Tensor) -> None:
    if target.is_meta:
        raise ValueError("Cannot apply selective checkpoint tensors to a meta-device model.")
    with torch.no_grad():
        target.copy_(source.to(device=target.device, dtype=target.dtype))


def load_dexjoco_selective_checkpoint(
    model,
    checkpoint: str | Path | Mapping[str, Any],
    *,
    report_path: str | Path | None = None,
    apply: bool = True,
    source_name: str | None = None,
) -> SelectiveCheckpointReport:
    """Load an old two-expert FastWAM checkpoint into a DexJoCo three-expert model.

    Matching is limited to documented exact roots plus exact local module keys. The
    old Action Expert backbone is copied to both Arm and Hand, while all action
    projections and the 23D proprio projection are explicitly reinitialized.
    """

    target_dimensions = (
        int(model.action_expert.action_dim),
        int(model.hand_expert.action_dim),
        int(model.proprio_encoder.in_features),
    )
    if target_dimensions != (6, 16, 23):
        raise ValueError(
            "DexJoCo selective checkpoint target must use Arm/Hand/proprio dimensions "
            f"(6, 16, 23), got {target_dimensions}."
        )

    if isinstance(checkpoint, (str, Path)):
        checkpoint_name = str(Path(checkpoint).expanduser().resolve())
        payload = _load_checkpoint_payload(checkpoint)
    elif _is_mapping(checkpoint):
        checkpoint_name = source_name or "<in-memory-state-dict>"
        payload = checkpoint
    else:
        raise TypeError(
            "`checkpoint` must be a path or mapping, "
            f"got {type(checkpoint).__name__}."
        )

    source_entries, checkpoint_format = _extract_checkpoint_entries(payload)
    report = SelectiveCheckpointReport(
        checkpoint=checkpoint_name,
        checkpoint_format=checkpoint_format,
        requested_apply=apply,
    )

    source_index: dict[tuple[str, str], tuple[str, torch.Tensor]] = {}
    unrecognized_sources: list[tuple[str, torch.Tensor]] = []
    for source_key, source_tensor, scope in source_entries:
        canonical = _canonical_source_key(source_key, scope)
        if canonical is None:
            unrecognized_sources.append((source_key, source_tensor))
            continue
        if canonical in source_index:
            previous_key = source_index[canonical][0]
            raise ValueError(
                "Checkpoint prefix normalization produced a duplicate exact key: "
                f"{previous_key!r} and {source_key!r} -> {canonical}."
            )
        source_index[canonical] = (source_key, source_tensor)

    target_states = {
        "video": model.video_expert.state_dict(),
        "action": model.action_expert.state_dict(),
        "hand": model.hand_expert.state_dict(),
        "proprio_encoder": model.proprio_encoder.state_dict(),
    }
    action_backbone_keys = model.action_expert.backbone_key_set(
        target_states["action"].keys()
    )
    hand_backbone_keys = model.hand_expert.backbone_key_set(
        target_states["hand"].keys()
    )
    if action_backbone_keys != hand_backbone_keys:
        raise ValueError("Arm and Hand backbone keys diverged before checkpoint loading.")

    consumed_sources: set[tuple[str, str]] = set()
    copy_operations: list[tuple[torch.Tensor, torch.Tensor]] = []
    critical_errors: list[str] = []

    def classify_backbone(
        *,
        target_module: str,
        source_module: str,
        local_keys,
        success_category: str,
    ) -> None:
        for local_key in sorted(local_keys):
            target_key = f"{target_module}.{local_key}"
            target_tensor = target_states[target_module][local_key]
            source_id = (source_module, local_key)
            source_entry = source_index.get(source_id)
            if source_entry is None:
                report.add(
                    "missing_in_checkpoint",
                    target=target_key,
                    target_module=target_module,
                    source=None,
                    checkpoint_shape=None,
                    model_shape=_shape(target_tensor),
                    reason=f"required_{target_module}_backbone_key_missing",
                )
                critical_errors.append(target_key)
                continue

            source_key, source_tensor = source_entry
            consumed_sources.add(source_id)
            if _shape(source_tensor) != _shape(target_tensor):
                report.add(
                    "skipped_shape",
                    target=target_key,
                    target_module=target_module,
                    source=source_key,
                    checkpoint_shape=_shape(source_tensor),
                    model_shape=_shape(target_tensor),
                    reason=f"required_{target_module}_backbone_shape_mismatch",
                )
                critical_errors.append(target_key)
                continue

            report.add(
                success_category,
                target=target_key,
                target_module=target_module,
                source=source_key,
                checkpoint_shape=_shape(source_tensor),
                model_shape=_shape(target_tensor),
                reason=(
                    "exact_video_backbone_match"
                    if target_module == "video"
                    else (
                        "exact_old_action_backbone_match"
                        if target_module == "action"
                        else "explicit_old_action_to_hand_backbone_remap"
                    )
                ),
            )
            copy_operations.append((target_tensor, source_tensor))

    classify_backbone(
        target_module="video",
        source_module="video",
        local_keys=target_states["video"].keys(),
        success_category="loaded",
    )
    classify_backbone(
        target_module="action",
        source_module="action",
        local_keys=action_backbone_keys,
        success_category="loaded",
    )
    classify_backbone(
        target_module="hand",
        source_module="action",
        local_keys=hand_backbone_keys,
        success_category="copied_to_hand",
    )

    policy_targets: list[tuple[str, str, str, str]] = []
    for target_module in ("action", "hand"):
        for local_key in _POLICY_LOCAL_KEYS:
            policy_targets.append((target_module, local_key, "action", local_key))
    for local_key in sorted(target_states["proprio_encoder"]):
        policy_targets.append(
            ("proprio_encoder", local_key, "proprio_encoder", local_key)
        )

    for target_module, local_key, source_module, source_local_key in policy_targets:
        target_key = f"{target_module}.{local_key}"
        target_tensor = target_states[target_module][local_key]
        source_id = (source_module, source_local_key)
        source_entry = source_index.get(source_id)
        source_key = None if source_entry is None else source_entry[0]
        source_tensor = None if source_entry is None else source_entry[1]
        if source_entry is not None:
            consumed_sources.add(source_id)
        report.add(
            "skipped_policy",
            target=target_key,
            target_module=target_module,
            source=source_key,
            checkpoint_shape=_shape(source_tensor),
            model_shape=_shape(target_tensor),
            reason="new_projection_must_not_reuse_old_action_or_proprio_weights",
        )
        report.add(
            "newly_initialized",
            target=target_key,
            target_module=target_module,
            source=None,
            checkpoint_shape=None,
            model_shape=_shape(target_tensor),
            reason="explicit_new_parameter_initialization",
            initialization="torch.nn.Linear.reset_parameters",
            initialization_applied=False,
        )

    for source_key, source_tensor in unrecognized_sources:
        report.add(
            "unexpected_in_checkpoint",
            target=None,
            target_module=None,
            source=source_key,
            checkpoint_shape=_shape(source_tensor),
            model_shape=None,
            reason="source_key_has_no_supported_exact_root",
        )
    for source_id, (source_key, source_tensor) in sorted(source_index.items()):
        if source_id in consumed_sources:
            continue
        report.add(
            "unexpected_in_checkpoint",
            target=None,
            target_module=source_id[0],
            source=source_key,
            checkpoint_shape=_shape(source_tensor),
            model_shape=None,
            reason="source_key_not_used_by_selective_policy",
        )

    if critical_errors:
        if report_path is not None:
            report.write_json(report_path)
        report.log()
        unique_errors = sorted(set(critical_errors))
        raise SelectiveCheckpointError(
            "Selective checkpoint is missing required backbone tensors or has incompatible "
            f"backbone shapes ({len(unique_errors)} targets): {unique_errors[:10]}",
            report,
        )

    if apply:
        model.action_expert.action_encoder.reset_parameters()
        model.action_expert.head.reset_parameters()
        model.hand_expert.action_encoder.reset_parameters()
        model.hand_expert.head.reset_parameters()
        model.proprio_encoder.reset_parameters()
        for target_tensor, source_tensor in copy_operations:
            _copy_tensor(target_tensor, source_tensor)
        for entry in report.categories["newly_initialized"]:
            entry["initialization_applied"] = True
        report.applied = True

    if report_path is not None:
        report.write_json(report_path)
    report.log()
    return report
