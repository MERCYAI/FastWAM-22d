import logging
import json
import inspect
import os
import re
import shutil
from collections.abc import Mapping
from math import ceil, cos, pi
from pathlib import Path
import time

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LambdaLR,
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader

from .datasets.lerobot.dexjoco_contract import (
    DEXJOCO_ACTION_DIM,
    DEXJOCO_ARM_ACTION_DIM,
    DEXJOCO_HAND_ACTION_DIM,
    DEXJOCO_PROPRIO_DIM,
    validate_dexjoco_statistics,
)
from .models.wan22.dexjoco_checkpoint import REPORT_SCHEMA, REPORT_VERSION
from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)


DEXJOCO_TRAINING_CHECKPOINT_SCHEMA = "fastwam.dexjoco.training_checkpoint"
DEXJOCO_TRAINING_CHECKPOINT_VERSION = 1
DEXJOCO_TRAINING_MANIFEST = "dexjoco_training_manifest.json"
DEXJOCO_TRAINING_CONFIG = "training_config.yaml"
DEXJOCO_DATASET_STATS = "dataset_stats.json"
DEXJOCO_SELECTIVE_REPORT = "selective_loading_report.json"


def _read_json_mapping(path: Path, description: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(dict(payload), file, ensure_ascii=True, indent=2)
        file.write("\n")


def _dexjoco_dimensions(model) -> dict[str, int] | None:
    if not hasattr(model, "hand_expert"):
        return None
    dimensions = {
        "action_dim": int(getattr(model, "action_dim", -1)),
        "arm_action_dim": int(getattr(model, "arm_action_dim", -1)),
        "hand_action_dim": int(getattr(model, "hand_action_dim", -1)),
        "proprio_dim": int(getattr(model, "proprio_dim", -1)),
    }
    expected = {
        "action_dim": DEXJOCO_ACTION_DIM,
        "arm_action_dim": DEXJOCO_ARM_ACTION_DIM,
        "hand_action_dim": DEXJOCO_HAND_ACTION_DIM,
        "proprio_dim": DEXJOCO_PROPRIO_DIM,
    }
    if dimensions != expected:
        raise ValueError(
            "DexJoCo model dimensional contract mismatch: "
            f"expected {expected}, got {dimensions}."
        )
    return dimensions


def validate_dexjoco_training_checkpoint(
    state_dir: str | Path,
    model,
    *,
    require_production_statistics: bool = True,
) -> dict:
    """Validate the DexJoCo resume contract before loading any tensor state."""

    dimensions = _dexjoco_dimensions(model)
    if dimensions is None:
        raise TypeError("DexJoCo checkpoint validation requires a dual-action model.")

    root = Path(state_dir).expanduser().resolve()
    manifest = _read_json_mapping(
        root / DEXJOCO_TRAINING_MANIFEST,
        "DexJoCo training checkpoint manifest",
    )
    if manifest.get("schema_name") != DEXJOCO_TRAINING_CHECKPOINT_SCHEMA:
        raise ValueError(
            "Invalid DexJoCo training checkpoint schema: "
            f"expected {DEXJOCO_TRAINING_CHECKPOINT_SCHEMA!r}, "
            f"got {manifest.get('schema_name')!r}."
        )
    if manifest.get("schema_version") != DEXJOCO_TRAINING_CHECKPOINT_VERSION:
        raise ValueError(
            "Invalid DexJoCo training checkpoint schema version: "
            f"expected {DEXJOCO_TRAINING_CHECKPOINT_VERSION}, "
            f"got {manifest.get('schema_version')!r}."
        )
    if manifest.get("dimensions") != dimensions:
        raise ValueError(
            "DexJoCo training checkpoint dimensions do not match the current model: "
            f"expected {dimensions}, got {manifest.get('dimensions')}."
        )

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("DexJoCo training checkpoint manifest is missing `artifacts`.")
    expected_artifacts = {
        "training_config": DEXJOCO_TRAINING_CONFIG,
        "dataset_statistics": DEXJOCO_DATASET_STATS,
        "selective_loading_report": DEXJOCO_SELECTIVE_REPORT,
    }
    for key, expected_name in expected_artifacts.items():
        if artifacts.get(key) != expected_name:
            raise ValueError(
                f"DexJoCo training checkpoint artifact `{key}` must be "
                f"{expected_name!r}, got {artifacts.get(key)!r}."
            )

    config_path = root / DEXJOCO_TRAINING_CONFIG
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing DexJoCo resolved training config: {config_path}")
    saved_config = OmegaConf.load(config_path)
    saved_contract = OmegaConf.select(saved_config, "data.contract", default=None)
    if saved_contract is not None:
        saved_dimensions = {
            key: int(saved_contract[key])
            for key in dimensions
            if key in saved_contract
        }
        if saved_dimensions != dimensions:
            raise ValueError(
                "DexJoCo saved training config dimensions do not match the checkpoint: "
                f"expected {dimensions}, got {saved_dimensions}."
            )

    statistics = _read_json_mapping(
        root / DEXJOCO_DATASET_STATS,
        "DexJoCo dataset statistics",
    )
    validate_dexjoco_statistics(
        statistics,
        require_production=require_production_statistics,
    )
    selective_report = _read_json_mapping(
        root / DEXJOCO_SELECTIVE_REPORT,
        "DexJoCo selective loading report",
    )
    if selective_report.get("schema") != REPORT_SCHEMA:
        raise ValueError(
            "Invalid DexJoCo selective loading report schema: "
            f"expected {REPORT_SCHEMA!r}, got {selective_report.get('schema')!r}."
        )
    if selective_report.get("version") != REPORT_VERSION:
        raise ValueError(
            "Invalid DexJoCo selective loading report version: "
            f"expected {REPORT_VERSION}, got {selective_report.get('version')!r}."
        )

    components = manifest.get("training_state_components")
    expected_components = {"model": True, "optimizer": True, "scheduler": True}
    if components != expected_components:
        raise ValueError(
            "DexJoCo checkpoint must declare model/optimizer/scheduler training state: "
            f"expected {expected_components}, got {components}."
        )
    state_files = artifacts.get("accelerate_state_files")
    if not isinstance(state_files, list) or not state_files:
        raise ValueError(
            "DexJoCo checkpoint manifest must list its Accelerate state files."
        )
    for relative_path in state_files:
        if not isinstance(relative_path, str):
            raise ValueError("DexJoCo Accelerate state file entries must be strings.")
        artifact_path = (root / relative_path).resolve()
        if root not in artifact_path.parents or not artifact_path.is_file():
            raise FileNotFoundError(
                f"Missing or invalid DexJoCo Accelerate state artifact: {relative_path!r}."
            )
    return manifest


def build_lr_scheduler(
    optimizer,
    scheduler_type,
    total_train_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.01,
):
    """Build a scheduler that preserves relative LRs across parameter groups."""

    scheduler_type = str(scheduler_type).strip().lower()
    total_train_steps = max(int(total_train_steps), 1)
    warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)
    remaining_steps = max(total_train_steps - warmup_steps, 1)

    if scheduler_type == "cosine":
        if len(optimizer.param_groups) == 1:
            base_lr = float(optimizer.param_groups[0]["lr"])
            main_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=remaining_steps,
                eta_min=base_lr * float(min_lr_ratio),
            )
        else:

            def lr_factor(step: int) -> float:
                progress = min(max(float(step) / remaining_steps, 0.0), 1.0)
                return float(min_lr_ratio) + (1.0 - float(min_lr_ratio)) * 0.5 * (
                    1.0 + cos(pi * progress)
                )

            main_scheduler = LambdaLR(optimizer, lr_lambda=lr_factor)
    elif scheduler_type == "constant":
        main_scheduler = ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=remaining_steps,
        )
    else:
        raise ValueError(
            f"Unsupported lr_scheduler_type: {scheduler_type}. "
            "Expected one of: ['cosine', 'constant']."
        )

    if warmup_steps <= 0:
        return main_scheduler

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1.0 / warmup_steps,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps],
    )


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        
        self.resume = cfg.resume
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)
        tensorboard_cfg = cfg.get("tensorboard") or {}
        self.tensorboard_enabled = bool(tensorboard_cfg.get("enabled", False))
        self.tensorboard_log_dir = str(
            tensorboard_cfg.get("log_dir", os.path.join(self.output_dir, "tensorboard"))
        )
        self.tensorboard_flush_every = max(
            int(tensorboard_cfg.get("flush_every", 10)), 1
        )

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
            cpu=bool(cfg.get("accelerator_cpu", False)),
        )
        
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage = (
            deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get(
                "stage", "unknown"
            )
            if deepspeed_plugin is not None
            else "disabled"
        )
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            zero_stage,
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional proprio encoder) as trainable when ZeRO builds optimizer state.
        self._apply_dit_only_train_mode(self.model)
        self.optimizer = self._build_optimizer()
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps = int(total_train_steps * 0.05)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self._restore_optimizer_group_names()
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self.tensorboard_writer = None
        self._init_wandb()
        self._resume_or_load_checkpoint()
        self._init_tensorboard()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _init_tensorboard(self):
        if not self.tensorboard_enabled or not self.accelerator.is_main_process:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise ImportError(
                "TensorBoard logging is enabled but the declared `tensorboard` dependency "
                "is unavailable. Install the project dependencies from pyproject.toml."
            ) from exc

        ensure_dir(self.tensorboard_log_dir)
        purge_step = self.global_step if self.global_step > 0 else None
        self.tensorboard_writer = SummaryWriter(
            log_dir=self.tensorboard_log_dir,
            purge_step=purge_step,
        )
        logger.info(
            "Initialized TensorBoard writer: log_dir=%s global_step=%d purge_step=%s",
            self.tensorboard_log_dir,
            self.global_step,
            purge_step,
        )

    def _tensorboard_log(self, payload: Mapping[str, float]) -> None:
        if self.tensorboard_writer is None:
            return
        for tag, value in payload.items():
            scalar = float(value)
            if not np.isfinite(scalar):
                raise FloatingPointError(
                    f"Refusing to write non-finite TensorBoard scalar {tag}={scalar}."
                )
            self.tensorboard_writer.add_scalar(tag, scalar, self.global_step)
        if self.global_step % self.tensorboard_flush_every == 0:
            self.tensorboard_writer.flush()

    def _finish_tensorboard(self) -> None:
        if self.tensorboard_writer is None:
            return
        try:
            self.tensorboard_writer.flush()
        finally:
            self.tensorboard_writer.close()
            self.tensorboard_writer = None

    def _finish_tracking(self) -> None:
        self._finish_tensorboard()
        self._finish_wandb()

    def _is_dexjoco_model(self) -> bool:
        return _dexjoco_dimensions(self.accelerator.unwrap_model(self.model)) is not None

    @staticmethod
    def _parameter_grad_squared_norm(
        parameters, *, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        unique_parameters = {id(parameter): parameter for parameter in parameters}.values()
        squared_norm = torch.zeros((), device=device, dtype=torch.float32)
        gradient_count = torch.zeros((), device=device, dtype=torch.float32)
        safe_get_local_grad = None
        for parameter in unique_parameters:
            gradient = parameter.grad
            if gradient is None and hasattr(parameter, "ds_id"):
                if safe_get_local_grad is None:
                    from deepspeed.utils import safe_get_local_grad as zero3_local_grad

                    safe_get_local_grad = zero3_local_grad
                gradient = safe_get_local_grad(parameter)
            if gradient is None:
                continue
            gradient_count += 1
            squared_norm = squared_norm + gradient.detach().float().pow(2).sum()
        return squared_norm, gradient_count

    def _distributed_parameter_grad_norm(
        self, parameters, *, device: torch.device
    ) -> torch.Tensor:
        squared_norm, gradient_count = self._parameter_grad_squared_norm(
            parameters, device=device
        )
        squared_norm = self.accelerator.reduce(squared_norm, reduction="sum")
        gradient_count = self.accelerator.reduce(gradient_count, reduction="sum")
        if float(gradient_count.item()) <= 0:
            return torch.full((), float("nan"), device=device, dtype=torch.float32)

        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage = (
            int(
                deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get(
                    "stage", 0
                )
            )
            if deepspeed_plugin is not None
            else 0
        )
        if zero_stage < 2:
            squared_norm = squared_norm / max(self.accelerator.num_processes, 1)
        return squared_norm.sqrt()

    def _dexjoco_gradient_norms(self, *, device: torch.device) -> dict[str, torch.Tensor]:
        model = self.accelerator.unwrap_model(self.model)
        if _dexjoco_dimensions(model) is None:
            return {}
        action_new_parameters = self._optimizer_group_parameters.get("action_new", ())
        if not action_new_parameters:
            raise RuntimeError("DexJoCo optimizer is missing the `action_new` parameter group.")
        return {
            "video": self._distributed_parameter_grad_norm(
                model.video_expert.parameters(), device=device
            ),
            "arm": self._distributed_parameter_grad_norm(
                model.action_expert.parameters(), device=device
            ),
            "hand": self._distributed_parameter_grad_norm(
                model.hand_expert.parameters(), device=device
            ),
            "action_new": self._distributed_parameter_grad_norm(
                action_new_parameters, device=device
            ),
        }

    @staticmethod
    def _weighted_action_mse(
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        action_is_pad: torch.Tensor | None,
        action_weight: torch.Tensor,
    ) -> torch.Tensor:
        token_loss = torch.nn.functional.mse_loss(
            prediction.float(), target.float(), reduction="none"
        ).mean(dim=2)
        if action_is_pad is None:
            per_sample = token_loss.mean(dim=1)
        else:
            valid = (~action_is_pad).to(device=token_loss.device, dtype=token_loss.dtype)
            per_sample = (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return (per_sample * action_weight.to(per_sample)).mean()

    def _dexjoco_action_diagnostics(
        self,
        sample: Mapping,
        outputs: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        model = self.accelerator.unwrap_model(self.model)
        prediction = outputs["action"].detach()
        target = outputs["target_action"].detach()
        action_weight = model.train_action_scheduler.training_weight(
            outputs["timestep_action"]
        )
        action_is_pad = sample.get("action_is_pad")
        scale = float(model.loss_lambda_action)
        return {
            "loss_arm_6d": scale
            * self._weighted_action_mse(
                prediction[..., :DEXJOCO_ARM_ACTION_DIM],
                target[..., :DEXJOCO_ARM_ACTION_DIM],
                action_is_pad=action_is_pad,
                action_weight=action_weight,
            ),
            "loss_hand_16d": scale
            * self._weighted_action_mse(
                prediction[..., DEXJOCO_ARM_ACTION_DIM:],
                target[..., DEXJOCO_ARM_ACTION_DIM:],
                action_is_pad=action_is_pad,
                action_weight=action_weight,
            ),
            "action_pred_mean": prediction.detach().float().mean(),
            "action_pred_std": prediction.detach().float().std(unbiased=False),
            "action_target_mean": target.detach().float().mean(),
            "action_target_std": target.detach().float().std(unbiased=False),
        }

    def _gather_scalar(self, value: torch.Tensor | float, *, device: torch.device) -> float:
        tensor = torch.as_tensor(value, device=device, dtype=torch.float32).reshape(1)
        return float(self.accelerator.gather(tensor).mean().item())

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
        )

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_optimizer(self):
        parameter_group_builder = getattr(
            self.model,
            "build_joint_post_training_parameter_groups",
            None,
        )
        optimizer_groups_cfg = self.cfg.get("optimizer_groups")
        if callable(parameter_group_builder):
            if optimizer_groups_cfg is None:
                raise ValueError(
                    "DexJoCo joint post-training requires the three named `optimizer_groups` "
                    "in the training config."
                )
            if isinstance(optimizer_groups_cfg, DictConfig):
                optimizer_groups_cfg = OmegaConf.to_container(
                    optimizer_groups_cfg,
                    resolve=True,
                )
            if not isinstance(optimizer_groups_cfg, Mapping):
                raise TypeError(
                    "`optimizer_groups` must resolve to a mapping, got "
                    f"{type(optimizer_groups_cfg).__name__}."
                )
            group_hyperparameters = {}
            for group_name, group_cfg in optimizer_groups_cfg.items():
                if not isinstance(group_cfg, Mapping):
                    raise TypeError(
                        f"Optimizer group `{group_name}` must be a mapping, "
                        f"got {type(group_cfg).__name__}."
                    )
                if "lr" not in group_cfg:
                    raise ValueError(f"Optimizer group `{group_name}` is missing `lr`.")
                group_hyperparameters[str(group_name)] = {
                    "lr": float(group_cfg["lr"]),
                    "weight_decay": float(group_cfg.get("weight_decay", self.weight_decay)),
                }
            parameter_groups = parameter_group_builder(group_hyperparameters)
        else:
            if optimizer_groups_cfg is not None:
                raise ValueError(
                    "Named `optimizer_groups` were configured for a model that does not "
                    "provide a parameter-group builder."
                )
            trainable_params = list(self.model.dit.parameters())
            proprio_encoder = getattr(self.model, "proprio_encoder", None)
            if proprio_encoder is not None:
                trainable_params.extend(list(proprio_encoder.parameters()))
            parameter_groups = [
                {
                    "name": "default",
                    "params": trainable_params,
                    "lr": self.learning_rate,
                    "weight_decay": self.weight_decay,
                }
            ]

        optimizer = torch.optim.AdamW(
            parameter_groups,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        self._optimizer_group_names = tuple(
            str(group.get("name", f"group_{index}"))
            for index, group in enumerate(optimizer.param_groups)
        )
        self._optimizer_group_parameters = {
            name: tuple(group["params"])
            for name, group in zip(
                self._optimizer_group_names,
                optimizer.param_groups,
                strict=True,
            )
        }
        for index, group in enumerate(optimizer.param_groups):
            parameters = list(group["params"])
            logger.info(
                "Optimizer group: name=%s lr=%.6g weight_decay=%.6g tensors=%d parameters=%d",
                group.get("name", f"group_{index}"),
                float(group["lr"]),
                float(group["weight_decay"]),
                len(parameters),
                sum(parameter.numel() for parameter in parameters),
            )
        return optimizer

    def _restore_optimizer_group_names(self) -> None:
        prepared_groups = self.optimizer.param_groups
        if len(prepared_groups) != len(self._optimizer_group_names):
            raise RuntimeError(
                "Accelerator changed the number of optimizer parameter groups: "
                f"before={len(self._optimizer_group_names)}, after={len(prepared_groups)}."
            )
        for expected_name, group in zip(
            self._optimizer_group_names,
            prepared_groups,
            strict=True,
        ):
            existing_name = group.get("name")
            if existing_name is not None and str(existing_name) != expected_name:
                raise RuntimeError(
                    "Accelerator reordered optimizer parameter groups: "
                    f"expected={expected_name!r}, got={existing_name!r}."
                )
            group["name"] = expected_name

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        return build_lr_scheduler(
            self.optimizer,
            scheduler_type=scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
        logger.warning("Loaded .pt weights only; optimizer/scheduler/step were not restored under ZeRO2.")

    def _set_dit_only_train_mode(self):
        # Match DiffSynth's freeze_except("dit"): only DiT stays trainable/in-train-mode.
        logger.info("Setting DiT to train mode and freezing other model components.")
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model)

    @staticmethod
    def _apply_dit_only_train_mode(model):
        configure_joint_post_training = getattr(
            model,
            "configure_joint_post_training",
            None,
        )
        if callable(configure_joint_post_training):
            configure_joint_post_training()
            return
        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)
        padding_masks = {}
        for key in ("action_is_pad", "image_is_pad", "proprio_is_pad"):
            value = sample.get(key)
            if value is not None:
                if value.ndim == 1:
                    value = value.unsqueeze(0)
                if value.ndim != 2:
                    raise ValueError(
                        f"`sample[{key!r}]` must have shape [T] or [B,T], "
                        f"got {tuple(value.shape)}."
                    )
            padding_masks[key] = value

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        result = {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
        }
        result.update(padding_masks)
        return result

    @torch.no_grad()
    def evaluate_loss(self):
        """Evaluate one deterministic per-rank sample without rollout or video decoding."""

        if self.val_dataset is None:
            return None
        model = self.accelerator.unwrap_model(self.model)
        if _dexjoco_dimensions(model) is None:
            raise TypeError("`evaluate_loss` is reserved for the DexJoCo dual-action model.")

        self.model.eval()
        try:
            rng = torch.Generator(device="cpu").manual_seed(
                self.global_step + self.accelerator.process_index
            )
            eval_index = torch.randint(
                0, len(self.val_dataset), (1,), generator=rng
            ).item()
            sample = self._to_batched_eval_sample(self.val_dataset[eval_index])
            with self.accelerator.autocast():
                loss, loss_dict, outputs = self.model(sample, return_outputs=True)
            diagnostics = self._dexjoco_action_diagnostics(sample, outputs)
            device = loss.device
            return {
                "loss_total": self._gather_scalar(loss.detach(), device=device),
                "loss_video": self._gather_scalar(loss_dict["loss_video"], device=device),
                "loss_action_22d": self._gather_scalar(
                    loss_dict["loss_action"], device=device
                ),
                "loss_arm_6d": self._gather_scalar(
                    diagnostics["loss_arm_6d"], device=device
                ),
                "loss_hand_16d": self._gather_scalar(
                    diagnostics["loss_hand_16d"], device=device
                ),
            }
        finally:
            self._set_dit_only_train_mode()

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        # eval_index = (self.global_step + self.accelerator.process_index) % len(self.val_dataset)
        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        # 1. training loss
        with self.accelerator.autocast():
            val_loss, _ = model.training_loss(sample)
            val_loss = val_loss.float().item()
        
        prompt = sample["prompt"][0]
        video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None # from [1, T, d] to [d]
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        # 2. inference and video saving
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample['action_horizon'],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        pred = model.infer(
            **infer_kwargs,
        )
        
        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        # 3. inference metrics against GT video
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_l1 = None
        action_l2 = None
        if action is not None and pred_action is not None:
            if sample["proprio"] is None:
                raise ValueError("Eval sample must contain `proprio` for action denormalization.")
            proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)
            
            processor = self.val_dataset.lerobot_dataset.processor

            denorm_actions = {}
            action_meta = processor.shape_meta["action"]
            state_meta = processor.shape_meta["state"]
            for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                if not isinstance(raw_action, torch.Tensor):
                    raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                if raw_action.ndim == 2:
                    action_btd = raw_action.unsqueeze(0)
                elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                    action_btd = raw_action
                else:
                    raise ValueError(
                        f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                    )
                action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                batch = {
                    "action": action_btd,
                    "state": proprio,
                }
                batch = processor.action_state_merger.backward(batch)
                batch = processor.normalizer.backward(batch)
                merged_batch = {
                    "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                    "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                }
                merged_batch = processor.action_state_merger.forward(merged_batch)
                denorm_action = merged_batch["action"].unsqueeze(0)
                if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                    raise ValueError(
                        f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                    )
                denorm_actions[action_name] = denorm_action

            pred_action_denorm = denorm_actions["pred"]
            gt_action_denorm = denorm_actions["gt"]

            if pred_action_denorm.shape != gt_action_denorm.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch after denormalization: "
                    f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                )
            action_diff = pred_action_denorm - gt_action_denorm
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

        # 4. VAE reconstruction metrics against GT video
        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        stitched_video_tensor = torch.cat(
            [pred_video_tensor, vae_video_tensor, gt_video_tensor],
            dim=2,
        ).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)

        local_metrics = torch.tensor(
            [
                float(val_loss),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_l2) if action_l2 is not None else -1.0,
                float(action_l1) if action_l1 is not None else -1.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics[:, :7].mean(dim=0)
        action_l2_mean = gathered_metrics[:, 7].mean().item() if action_l2 is not None else None
        action_l1_mean = gathered_metrics[:, 8].mean().item() if action_l1 is not None else None

        if was_dit_training:
            self._set_dit_only_train_mode()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[1].item()),
            "ssim_rg": float(mean_metrics[2].item()),
            "psnr_rd": float(mean_metrics[3].item()),
            "ssim_rd": float(mean_metrics[4].item()),
            "psnr_dg": float(mean_metrics[5].item()),
            "ssim_dg": float(mean_metrics[6].item()),
            "video_path": video_path,
        }
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        model.save_checkpoint(ckpt_path, optimizer=None, step=self.global_step)
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def _dexjoco_requires_production_statistics(self) -> bool:
        allow_non_production = OmegaConf.select(
            self.cfg,
            "data.train.processor.allow_non_production_stats",
            default=False,
        )
        return not bool(allow_non_production)

    def _resolve_dexjoco_dataset_stats(self) -> Path:
        candidates = [Path(self.output_dir) / DEXJOCO_DATASET_STATS]
        configured = OmegaConf.select(
            self.cfg,
            "data.train.pretrained_norm_stats",
            default=None,
        )
        if configured:
            candidates.append(Path(str(configured)).expanduser())
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(
            "DexJoCo checkpoint saving requires dataset_stats.json from the training dataset; "
            f"checked {[str(path) for path in candidates]}."
        )

    def _resolve_dexjoco_selective_report(self, model) -> tuple[dict, str]:
        report = getattr(model, "selective_checkpoint_report", None)
        if isinstance(report, Mapping):
            return dict(report), "model.selective_checkpoint_report"
        configured = OmegaConf.select(
            self.cfg,
            "model.selective_checkpoint_report_path",
            default=None,
        )
        if configured:
            path = Path(str(configured)).expanduser().resolve()
            return _read_json_mapping(path, "DexJoCo selective loading report"), str(path)
        raise FileNotFoundError(
            "DexJoCo checkpoint saving requires the selective checkpoint loading report."
        )

    def _save_dexjoco_checkpoint_artifacts(
        self,
        state_path: str,
        *,
        weights_path: str | None,
    ) -> Path | None:
        model = self.accelerator.unwrap_model(self.model)
        dimensions = _dexjoco_dimensions(model)
        if dimensions is None:
            return None

        root = Path(state_path).resolve()
        stats_source = self._resolve_dexjoco_dataset_stats()
        statistics = _read_json_mapping(stats_source, "DexJoCo dataset statistics")
        validate_dexjoco_statistics(
            statistics,
            require_production=self._dexjoco_requires_production_statistics(),
        )
        report, report_source = self._resolve_dexjoco_selective_report(model)
        if report.get("schema") != REPORT_SCHEMA or report.get("version") != REPORT_VERSION:
            raise ValueError(
                "DexJoCo selective loading report has an incompatible schema/version: "
                f"schema={report.get('schema')!r}, version={report.get('version')!r}."
            )

        stats_target = root / DEXJOCO_DATASET_STATS
        if stats_source != stats_target:
            shutil.copyfile(stats_source, stats_target)
        _write_json(root / DEXJOCO_SELECTIVE_REPORT, report)
        OmegaConf.save(
            config=OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True)),
            f=root / DEXJOCO_TRAINING_CONFIG,
        )

        state_files = sorted(
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.is_file()
            and path.name
            not in {
                DEXJOCO_TRAINING_MANIFEST,
                DEXJOCO_TRAINING_CONFIG,
                DEXJOCO_DATASET_STATS,
                DEXJOCO_SELECTIVE_REPORT,
                "trainer_state.json",
            }
        )
        manifest = {
            "schema_name": DEXJOCO_TRAINING_CHECKPOINT_SCHEMA,
            "schema_version": DEXJOCO_TRAINING_CHECKPOINT_VERSION,
            "dimensions": dimensions,
            "training_state_components": {
                "model": True,
                "optimizer": True,
                "scheduler": True,
            },
            "artifacts": {
                "training_config": DEXJOCO_TRAINING_CONFIG,
                "dataset_statistics": DEXJOCO_DATASET_STATS,
                "selective_loading_report": DEXJOCO_SELECTIVE_REPORT,
                "accelerate_state_files": state_files,
                "weights_checkpoint": (
                    None
                    if weights_path is None
                    else os.path.relpath(Path(weights_path).resolve(), start=root)
                ),
            },
            "dataset_statistics_source": str(stats_source),
            "dataset_statistics_schema": {
                "schema_name": statistics.get("schema_name"),
                "schema_version": statistics.get("schema_version"),
                "statistics_mode": statistics.get("statistics_mode"),
                "production": statistics.get("production"),
            },
            "selective_loading_report_source": report_source,
            "selective_loading_report_schema": {
                "schema": report.get("schema"),
                "version": report.get("version"),
            },
        }
        manifest_path = root / DEXJOCO_TRAINING_MANIFEST
        _write_json(manifest_path, manifest)
        validate_dexjoco_training_checkpoint(
            root,
            model,
            require_production_statistics=self._dexjoco_requires_production_statistics(),
        )
        logger.info(
            "Saved DexJoCo training checkpoint contract: manifest=%s stats=%s "
            "selective_report=%s accelerate_files=%d",
            manifest_path,
            stats_target,
            root / DEXJOCO_SELECTIVE_REPORT,
            len(state_files),
        )
        return manifest_path

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage = (
            int(
                deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get(
                    "stage", 0
                )
            )
            if deepspeed_plugin is not None
            else 0
        )
        if self.accelerator.is_main_process and zero_stage < 3:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        elif self.accelerator.is_main_process:
            logger.info(
                "Skipping standalone weights checkpoint for ZeRO stage %d; "
                "the sharded Accelerate/DeepSpeed state is the authoritative checkpoint.",
                zero_stage,
            )
        self.accelerator.wait_for_everyone()

        state_path = os.path.join(self.state_dir, step_tag)
        ensure_dir(state_path)
        model = self.accelerator.unwrap_model(self.model)
        save_state_kwargs = {}
        if _dexjoco_dimensions(model) is not None:
            # The experts are also registered under MoT, so preserve their shared
            # parameter aliases instead of letting safetensors discard duplicate keys.
            save_state_kwargs["safe_serialization"] = False
        self.accelerator.save_state(output_dir=state_path, **save_state_kwargs)
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self._save_dexjoco_checkpoint_artifacts(
                state_path,
                weights_path=ckpt_path,
            )
            self._save_trainer_state(state_path)
        self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        model = self.accelerator.unwrap_model(self.model)
        if _dexjoco_dimensions(model) is not None:
            self.dexjoco_resume_manifest = validate_dexjoco_training_checkpoint(
                state_dir,
                model,
                require_production_statistics=self._dexjoco_requires_production_statistics(),
            )
            active_stats_path = self._resolve_dexjoco_dataset_stats()
            active_statistics = _read_json_mapping(
                active_stats_path,
                "active DexJoCo dataset statistics",
            )
            checkpoint_statistics = _read_json_mapping(
                Path(state_dir) / DEXJOCO_DATASET_STATS,
                "checkpoint DexJoCo dataset statistics",
            )
            if active_statistics != checkpoint_statistics:
                raise ValueError(
                    "Active DexJoCo dataset statistics do not exactly match the resume "
                    f"checkpoint copy: active={active_stats_path}, "
                    f"checkpoint={Path(state_dir) / DEXJOCO_DATASET_STATS}."
                )
            logger.info(
                "Validated DexJoCo training checkpoint contract before tensor-state load: %s",
                Path(state_dir) / DEXJOCO_TRAINING_MANIFEST,
            )
        self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch_offset(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def train(self):
        try:
            return self._train_loop()
        finally:
            self._finish_tracking()

    def _train_loop(self):
        self._set_dit_only_train_mode()

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        is_dexjoco = _dexjoco_dimensions(unwrapped_model) is not None

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()
        data_wait_started = time.perf_counter()
        optimizer_step_started = time.perf_counter()

        while self.global_step < self.max_steps:
            try:
                sample = next(data_iter)
                data_time = time.perf_counter() - data_wait_started
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                data_wait_started = time.perf_counter()
                continue

            with self.accelerator.accumulate(self.model):
                should_capture_diagnostics = (
                    is_dexjoco
                    and self.tensorboard_enabled
                    and self.log_every > 0
                    and (self.global_step + 1) % self.log_every == 0
                )

                with self.accelerator.autocast():
                    if should_capture_diagnostics:
                        loss, loss_dict, outputs = self.model(
                            sample, return_outputs=True
                        )
                    else:
                        loss, loss_dict = self.model(sample)
                        outputs = None
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    should_log = self.log_every > 0 and (self.global_step + 1) % self.log_every == 0
                    self.accelerator.unscale_gradients()
                    group_grad_norms = (
                        self._dexjoco_gradient_norms(device=loss.device)
                        if should_log and is_dexjoco and self.tensorboard_enabled
                        else {}
                    )
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    optimizer_step_was_skipped = self.accelerator.optimizer_step_was_skipped
                    if not optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if optimizer_step_was_skipped:
                        logger.warning(
                            "Optimizer step was skipped; global_step and TensorBoard step remain %d.",
                            self.global_step,
                        )
                        data_wait_started = time.perf_counter()
                        optimizer_step_started = data_wait_started
                        continue
                    self.global_step += 1
                    global_loss = self._gather_scalar(loss.detach(), device=loss.device)
                    global_loss_metrics = {}
                    for key, value in loss_dict.items():
                        global_loss_metrics[key] = self._gather_scalar(
                            value, device=loss.device
                        )
                    global_grad_norm = self._gather_scalar(grad_norm, device=loss.device)

                    global_diagnostics = {}
                    if should_log and outputs is not None:
                        diagnostics = self._dexjoco_action_diagnostics(sample, outputs)
                        global_diagnostics = {
                            key: self._gather_scalar(value, device=loss.device)
                            for key, value in diagnostics.items()
                        }
                    global_group_grad_norms = {
                        key: self._gather_scalar(value, device=loss.device)
                        for key, value in group_grad_norms.items()
                    }

                    current_lrs = {
                        str(group.get("name", f"group_{index}")): float(group["lr"])
                        for index, group in enumerate(self.optimizer.param_groups)
                    }
                    current_lr = next(iter(current_lrs.values()))
                    step_time = time.perf_counter() - optimizer_step_started

                    if should_log and self.accelerator.is_main_process:
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                            description += detail_str + " "
                        lr_description = ",".join(
                            f"{name}:{lr:.2e}" for name, lr in current_lrs.items()
                        )
                        description += "lr=[%s] speed=%.2f step/s, %.2f samples/s eta=%s" % (
                            lr_description,
                            steps_per_sec,
                            steps_per_sec * self.batch_size * self.accelerator.num_processes,
                            eta_str,
                        )
                        logger.info(description)

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": steps_per_sec * self.batch_size * self.accelerator.num_processes,
                        }
                        for group_name, group_lr in current_lrs.items():
                            wandb_payload[f"train/lr/{group_name}"] = group_lr
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        self._wandb_log(wandb_payload)

                        if is_dexjoco and self.tensorboard_enabled:
                            tensorboard_payload = {
                                "train/loss_total": global_loss,
                                "train/loss_video": global_loss_metrics["loss_video"],
                                "train/loss_action_22d": global_loss_metrics["loss_action"],
                                "train/loss_arm_6d": global_diagnostics["loss_arm_6d"],
                                "train/loss_hand_16d": global_diagnostics["loss_hand_16d"],
                                "train/step_time": step_time,
                                "train/data_time": data_time,
                                "train/epoch": float(self.epoch),
                                "train/samples_seen": float(
                                    self.global_step
                                    * self.batch_size
                                    * self.accelerator.num_processes
                                    * self.gradient_accumulation_steps
                                ),
                                "action_pred/mean": global_diagnostics["action_pred_mean"],
                                "action_pred/std": global_diagnostics["action_pred_std"],
                                "action_target/mean": global_diagnostics["action_target_mean"],
                                "action_target/std": global_diagnostics["action_target_std"],
                            }
                            tensorboard_payload.update(
                                {
                                    f"lr/{group_name}": group_lr
                                    for group_name, group_lr in current_lrs.items()
                                }
                            )
                            tensorboard_payload.update(
                                {
                                    f"grad_norm/{group_name}": norm
                                    for group_name, norm in global_group_grad_norms.items()
                                }
                            )
                            self._tensorboard_log(tensorboard_payload)

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate_loss() if is_dexjoco else self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            if is_dexjoco:
                                logger.info(
                                    "[eval] step=%d loss=%.4f video=%.4f action=%.4f "
                                    "arm_diag=%.4f hand_diag=%.4f",
                                    self.global_step,
                                    metrics["loss_total"],
                                    metrics["loss_video"],
                                    metrics["loss_action_22d"],
                                    metrics["loss_arm_6d"],
                                    metrics["loss_hand_16d"],
                                )
                                val_payload = {
                                    "val/loss_total": metrics["loss_total"],
                                    "val/loss_video": metrics["loss_video"],
                                    "val/loss_action_22d": metrics["loss_action_22d"],
                                    "val/loss_arm_6d": metrics["loss_arm_6d"],
                                    "val/loss_hand_16d": metrics["loss_hand_16d"],
                                }
                                self._tensorboard_log(val_payload)
                                self._wandb_log(val_payload)
                            else:
                                description = "[eval] step=%d val_loss=%.4f infer_psnr=%.4f infer_ssim=%.4f" % (
                                    self.global_step,
                                    metrics["val_loss"],
                                    metrics["psnr_rd"],
                                    metrics["ssim_rd"],
                                )
                                if "action_l2" in metrics:
                                    description += " action_l2=%.4f" % metrics["action_l2"]
                                if "action_l1" in metrics:
                                    description += " action_l1=%.4f" % metrics["action_l1"]
                                logger.info(description)
                                eval_payload = {
                                    "eval/val_loss": float(metrics["val_loss"]),
                                    "eval/psnr_rg": float(metrics["psnr_rg"]),
                                    "eval/ssim_rg": float(metrics["ssim_rg"]),
                                    "eval/psnr_rd": float(metrics["psnr_rd"]),
                                    "eval/ssim_rd": float(metrics["ssim_rd"]),
                                    "eval/psnr_dg": float(metrics["psnr_dg"]),
                                    "eval/ssim_dg": float(metrics["ssim_dg"]),
                                }
                                if "action_l2" in metrics:
                                    eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                                if "action_l1" in metrics:
                                    eval_payload["eval/action_l1"] = float(metrics["action_l1"])
                                self._wandb_log(eval_payload)

                    if self.save_every > 0 and self.global_step % self.save_every == 0:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    if self.global_step >= self.max_steps:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[done] max_steps reached step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                        return

                    optimizer_step_started = time.perf_counter()

            data_wait_started = time.perf_counter()

        ckpt_info = self.save_checkpoint()
        if self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
        
