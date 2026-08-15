from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Optional

import torch
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .dexjoco_checkpoint import load_dexjoco_selective_checkpoint
from .fastwam import FastWAM
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT

logger = get_logger(__name__)


class DexJoCoDualActionFastWAM(FastWAM):
    """FastWAM variant with separate arm and hand ActionDiT experts."""

    ACTION_DIM = 22
    ARM_ACTION_DIM = 6
    HAND_ACTION_DIM = 16
    PROPRIO_DIM = 23
    EXPERT_ORDER = ("video", "action", "hand")
    JOINT_OPTIMIZER_GROUP_NAMES = (
        "action_new",
        "action_backbone",
        "video_backbone",
    )

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        hand_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: int = PROPRIO_DIM,
        **kwargs,
    ):
        if not isinstance(action_expert, ActionDiT):
            raise TypeError(
                f"DexJoCo Arm Expert must be ActionDiT, got {type(action_expert).__name__}."
            )
        if not isinstance(hand_expert, ActionDiT):
            raise TypeError(
                f"DexJoCo Hand Expert must be ActionDiT, got {type(hand_expert).__name__}."
            )
        if int(action_expert.action_dim) != self.ARM_ACTION_DIM:
            raise ValueError(
                f"DexJoCo Arm Expert (`action`) must use {self.ARM_ACTION_DIM}D input/output, "
                f"got {action_expert.action_dim}."
            )
        if int(hand_expert.action_dim) != self.HAND_ACTION_DIM:
            raise ValueError(
                f"DexJoCo Hand Expert must use {self.HAND_ACTION_DIM}D input/output, "
                f"got {hand_expert.action_dim}."
            )
        action_state = action_expert.state_dict()
        hand_state = hand_expert.state_dict()
        action_backbone_shapes = {
            key: tuple(action_state[key].shape)
            for key in ActionDiT.backbone_key_set(action_state)
        }
        hand_backbone_shapes = {
            key: tuple(hand_state[key].shape)
            for key in ActionDiT.backbone_key_set(hand_state)
        }
        if action_backbone_shapes != hand_backbone_shapes:
            missing = sorted(action_backbone_shapes.keys() - hand_backbone_shapes.keys())
            unexpected = sorted(hand_backbone_shapes.keys() - action_backbone_shapes.keys())
            mismatched = sorted(
                key
                for key in action_backbone_shapes.keys() & hand_backbone_shapes.keys()
                if action_backbone_shapes[key] != hand_backbone_shapes[key]
            )
            raise ValueError(
                "DexJoCo Arm/Hand ActionDiT backbones must have identical keys and shapes; "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}, mismatched={mismatched[:5]}."
            )
        if int(proprio_dim) != self.PROPRIO_DIM:
            raise ValueError(
                f"DexJoCo proprio encoder must accept {self.PROPRIO_DIM}D state, got {proprio_dim}."
            )
        if tuple(mot.expert_order) != self.EXPERT_ORDER:
            raise ValueError(
                f"DexJoCo MoT expert order must be {self.EXPERT_ORDER}, got {tuple(mot.expert_order)}."
            )
        expected_modules = {
            "video": video_expert,
            "action": action_expert,
            "hand": hand_expert,
        }
        for name, expert in expected_modules.items():
            if mot.mixtures[name] is not expert:
                raise ValueError(f"DexJoCo MoT expert `{name}` does not match the provided module.")

        super().__init__(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_dim=text_dim,
            proprio_dim=proprio_dim,
            **kwargs,
        )
        self.hand_expert = hand_expert
        self.action_dim = self.ACTION_DIM
        self.arm_action_dim = self.ARM_ACTION_DIM
        self.hand_action_dim = self.HAND_ACTION_DIM
        self.supports_video_kv_cache = False
        logger.warning(
            "DexJoCo Hand Expert is not supported by the current single-action video KV cache; "
            "cached DexJoCo action inference is disabled in Phase 2."
        )

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: int = PROPRIO_DIM,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        hand_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        selective_checkpoint_path: str | None = None,
        selective_checkpoint_report_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
    ) -> "DexJoCoDualActionFastWAM":
        if video_dit_config is None or "text_dim" not in video_dit_config:
            raise ValueError("DexJoCo requires `video_dit_config` with `text_dim`.")
        if action_dit_config is None:
            raise ValueError("DexJoCo requires `action_dit_config` for the Arm Expert.")
        if hand_dit_config is None:
            raise ValueError("DexJoCo requires `hand_dit_config` for the Hand Expert.")
        if int(action_dit_config.get("action_dim", -1)) != cls.ARM_ACTION_DIM:
            raise ValueError("DexJoCo `action_dit_config.action_dim` must be 6.")
        if int(hand_dit_config.get("action_dim", -1)) != cls.HAND_ACTION_DIM:
            raise ValueError("DexJoCo `hand_dit_config.action_dim` must be 16.")

        use_selective_checkpoint = selective_checkpoint_path is not None
        skip_base_dit_load = bool(skip_dit_load_from_pretrain or use_selective_checkpoint)
        if use_selective_checkpoint:
            logger.info(
                "DexJoCo selective checkpoint requested; initializing Video/Arm/Hand DiTs "
                "before audited loading from %s.",
                selective_checkpoint_path,
            )

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_base_dit_load,
            load_text_encoder=load_text_encoder,
        )
        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_base_dit_load,
            device=device,
            torch_dtype=torch_dtype,
        )
        # Phase 2 only defines structure. The old Action Expert -> Hand Expert
        # backbone remap is deliberately deferred to the checkpoint migration phase.
        hand_expert = ActionDiT(**hand_dit_config).to(device=device, dtype=torch_dtype)

        for name, expert in (("action", action_expert), ("hand", hand_expert)):
            if int(expert.num_heads) != int(video_expert.num_heads):
                raise ValueError(f"DexJoCo {name} expert `num_heads` must match the video expert.")
            if int(expert.attn_head_dim) != int(video_expert.attn_head_dim):
                raise ValueError(
                    f"DexJoCo {name} expert `attn_head_dim` must match the video expert."
                )
            if len(expert.blocks) != len(video_expert.blocks):
                raise ValueError(f"DexJoCo {name} expert `num_layers` must match the video expert.")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert, "hand": hand_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )
        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            hand_expert=hand_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
        )
        model.model_paths = {
            "video_dit": (
                f"SELECTIVE_CHECKPOINT:{selective_checkpoint_path}"
                if use_selective_checkpoint
                else components.dit_path
            ),
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                f"SELECTIVE_CHECKPOINT:{selective_checkpoint_path}"
                if use_selective_checkpoint
                else (
                    "SKIPPED_PRETRAIN"
                    if skip_dit_load_from_pretrain
                    else action_dit_pretrained_path
                )
            ),
            "hand_dit_backbone": (
                f"SELECTIVE_CHECKPOINT_REMAP:{selective_checkpoint_path}"
                if use_selective_checkpoint
                else "RANDOM_INIT_PHASE2_PENDING_CHECKPOINT_REMAP"
            ),
        }
        if use_selective_checkpoint:
            model.selective_checkpoint_report = model.load_selective_pretrained_checkpoint(
                selective_checkpoint_path,
                report_path=selective_checkpoint_report_path,
            ).to_dict()
        return model

    def load_selective_pretrained_checkpoint(
        self,
        checkpoint,
        *,
        report_path: str | None = None,
        apply: bool = True,
        source_name: str | None = None,
    ):
        return load_dexjoco_selective_checkpoint(
            self,
            checkpoint,
            report_path=report_path,
            apply=apply,
            source_name=source_name,
        )

    def configure_joint_post_training(self) -> None:
        """Apply the DexJoCo joint post-training train/eval and freeze policy."""

        if self.proprio_encoder is None:
            raise ValueError("DexJoCo joint post-training requires a 23D proprio encoder.")

        self.train()
        self.requires_grad_(False)
        trainable_modules = (
            self.video_expert,
            self.action_expert,
            self.hand_expert,
            self.proprio_encoder,
        )
        for module in trainable_modules:
            module.train()
            module.requires_grad_(True)

        for module in (self.text_encoder, self.vae):
            if module is not None:
                module.eval()
                module.requires_grad_(False)

    def build_joint_post_training_parameter_groups(
        self,
        group_hyperparameters: Mapping[str, Mapping[str, float]],
    ) -> list[dict[str, Any]]:
        """Build complete, disjoint DexJoCo optimizer groups by module ownership."""

        expected_names = set(self.JOINT_OPTIMIZER_GROUP_NAMES)
        actual_names = set(group_hyperparameters)
        if actual_names != expected_names:
            raise ValueError(
                "DexJoCo optimizer groups must be exactly "
                f"{self.JOINT_OPTIMIZER_GROUP_NAMES}, got {sorted(actual_names)}."
            )

        options: dict[str, dict[str, float]] = {}
        for group_name in self.JOINT_OPTIMIZER_GROUP_NAMES:
            group_options = group_hyperparameters[group_name]
            if "lr" not in group_options or "weight_decay" not in group_options:
                raise ValueError(
                    f"DexJoCo optimizer group `{group_name}` requires `lr` and `weight_decay`."
                )
            lr = float(group_options["lr"])
            weight_decay = float(group_options["weight_decay"])
            if not math.isfinite(lr) or lr <= 0:
                raise ValueError(f"Optimizer group `{group_name}` LR must be finite and > 0, got {lr}.")
            if not math.isfinite(weight_decay) or weight_decay < 0:
                raise ValueError(
                    f"Optimizer group `{group_name}` weight decay must be finite and >= 0, "
                    f"got {weight_decay}."
                )
            options[group_name] = {"lr": lr, "weight_decay": weight_decay}

        action_new_lr = options["action_new"]["lr"]
        action_backbone_lr = options["action_backbone"]["lr"]
        video_backbone_lr = options["video_backbone"]["lr"]
        if not action_new_lr > action_backbone_lr >= video_backbone_lr:
            raise ValueError(
                "DexJoCo optimizer LR ordering must satisfy "
                "action_new > action_backbone >= video_backbone, got "
                f"{action_new_lr}, {action_backbone_lr}, {video_backbone_lr}."
            )

        named_groups: dict[str, list[tuple[str, torch.nn.Parameter]]] = {
            group_name: [] for group_name in self.JOINT_OPTIMIZER_GROUP_NAMES
        }

        def add_module(group_name: str, prefix: str, module) -> None:
            for local_name, parameter in module.named_parameters():
                named_groups[group_name].append((f"{prefix}.{local_name}", parameter))

        add_module("action_new", "action.action_encoder", self.action_expert.action_encoder)
        add_module("action_new", "action.head", self.action_expert.head)
        add_module("action_new", "hand.action_encoder", self.hand_expert.action_encoder)
        add_module("action_new", "hand.head", self.hand_expert.head)
        add_module("action_new", "proprio_encoder", self.proprio_encoder)

        for expert_name, expert in (
            ("action", self.action_expert),
            ("hand", self.hand_expert),
        ):
            for local_name, parameter in expert.named_parameters():
                if any(
                    local_name.startswith(prefix)
                    for prefix in ActionDiT.ACTION_BACKBONE_SKIP_PREFIXES
                ):
                    continue
                named_groups["action_backbone"].append(
                    (f"{expert_name}.{local_name}", parameter)
                )
        add_module("video_backbone", "video", self.video_expert)

        model_parameter_names = {
            id(parameter): name for name, parameter in self.named_parameters()
        }
        trainable_ids = {
            id(parameter)
            for parameter in self.parameters()
            if parameter.requires_grad
        }
        assigned_ids: set[int] = set()
        duplicate_names: list[str] = []
        frozen_names: list[str] = []
        for entries in named_groups.values():
            for stable_name, parameter in entries:
                parameter_id = id(parameter)
                if parameter_id in assigned_ids:
                    duplicate_names.append(stable_name)
                assigned_ids.add(parameter_id)
                if not parameter.requires_grad:
                    frozen_names.append(stable_name)

        if duplicate_names:
            raise ValueError(
                "DexJoCo optimizer parameter groups overlap: "
                f"{sorted(duplicate_names)[:10]}."
            )
        if frozen_names:
            raise ValueError(
                "DexJoCo optimizer groups contain frozen parameters: "
                f"{sorted(frozen_names)[:10]}."
            )
        if assigned_ids != trainable_ids:
            missing_ids = trainable_ids - assigned_ids
            extra_ids = assigned_ids - trainable_ids
            missing_names = sorted(
                model_parameter_names.get(parameter_id, f"<unknown:{parameter_id}>")
                for parameter_id in missing_ids
            )
            extra_names = sorted(
                model_parameter_names.get(parameter_id, f"<unknown:{parameter_id}>")
                for parameter_id in extra_ids
            )
            raise ValueError(
                "DexJoCo optimizer groups must cover every trainable parameter exactly once; "
                f"missing={missing_names[:10]}, extra={extra_names[:10]}."
            )

        self.joint_optimizer_group_parameter_names = {
            group_name: tuple(name for name, _ in named_groups[group_name])
            for group_name in self.JOINT_OPTIMIZER_GROUP_NAMES
        }
        return [
            {
                "name": group_name,
                "params": [parameter for _, parameter in named_groups[group_name]],
                "lr": options[group_name]["lr"],
                "weight_decay": options[group_name]["weight_decay"],
            }
            for group_name in self.JOINT_OPTIMIZER_GROUP_NAMES
        ]

    def split_action(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if action.ndim != 3:
            raise ValueError(
                f"DexJoCo action must be [B, T, {self.ACTION_DIM}], got {tuple(action.shape)}."
            )
        if action.shape[-1] != self.ACTION_DIM:
            raise ValueError(
                f"DexJoCo action last dimension must be {self.ACTION_DIM}, got {action.shape[-1]}."
            )
        return action[..., : self.ARM_ACTION_DIM], action[..., self.ARM_ACTION_DIM :]

    @torch.no_grad()
    def _build_dual_action_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        hand_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        lengths = (video_seq_len, action_seq_len, hand_seq_len)
        if any(length <= 0 for length in lengths):
            raise ValueError(f"DexJoCo expert sequence lengths must be positive, got {lengths}.")
        if action_seq_len != hand_seq_len:
            raise ValueError(
                "DexJoCo Arm/Hand token sequences must share one action horizon, "
                f"got {action_seq_len} and {hand_seq_len}."
            )

        total_seq_len = sum(lengths)
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        # Both action experts read all video tokens and interact bidirectionally.
        # Video query rows keep the action/hand columns false.
        mask[video_seq_len:, :] = True
        return mask

    def _forward_dual_experts(
        self,
        *,
        latents_video: torch.Tensor,
        noisy_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action_condition: Optional[torch.Tensor],
        fuse_vae_embedding_in_latents: bool,
    ) -> dict[str, torch.Tensor]:
        arm_action, hand_action = self.split_action(noisy_action)
        if action_condition is not None:
            self.split_action(action_condition)

        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action_condition,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=arm_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        hand_pre = self.hand_expert.pre_dit(
            action_tokens=hand_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_dual_action_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            hand_seq_len=hand_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
                "hand": hand_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
                "hand": hand_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
                "hand": {
                    "context": hand_pre["context"],
                    "mask": hand_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
                "hand": hand_pre["t_mod"],
            },
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_arm_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        pred_hand_action = self.hand_expert.post_dit(tokens_out["hand"], hand_pre)
        pred_action = torch.cat([pred_arm_action, pred_hand_action], dim=-1)
        if pred_action.shape[-1] != self.ACTION_DIM:
            raise RuntimeError(f"DexJoCo concatenated prediction must be 22D, got {pred_action.shape[-1]}.")

        return {
            "video": pred_video,
            "arm_action": pred_arm_action,
            "hand_action": pred_hand_action,
            "action": pred_action,
            "attention_mask": attention_mask,
        }

    def training_loss(
        self,
        sample,
        tiled: bool = False,
        return_outputs: bool = False,
    ):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        self.split_action(action)
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(
            input_latents, noise_video, timestep_video
        )
        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        # Generate one full 22D noise tensor and one timestep, then split only
        # after scheduler processing so Arm and Hand share the diffusion state.
        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(
            action, noise_action, timestep_action
        )
        target_action = self.train_action_scheduler.training_target(
            action, noise_action, timestep_action
        )

        outputs = self._forward_dual_experts(
            latents_video=latents,
            noisy_action=noisy_action,
            timestep_video=timestep_video,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            action_condition=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        pred_video = outputs["video"]
        pred_action = outputs["action"]

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        # Preserve the original FastWAM reduction: one MSE over the complete
        # 22D prediction/target, followed by padding and scheduler weighting.
        action_loss_token = F.mse_loss(
            pred_action.float(), target_action.float(), reduction="none"
        ).mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(
                device=action_loss_token.device, dtype=action_loss_token.dtype
            )
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()
        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        if return_outputs:
            outputs["target_action"] = target_action
            outputs["timestep_action"] = timestep_action
            return loss_total, loss_dict, outputs
        return loss_total, loss_dict

    def _raise_cache_disabled(self) -> None:
        message = (
            "DexJoCo cached action inference is disabled in Phase 2: "
            "MoT.forward_action_with_video_cache only updates the `action` expert and cannot "
            "represent bidirectional Arm/Hand token interaction."
        )
        logger.warning(message)
        raise NotImplementedError(message)

    @torch.no_grad()
    def _predict_action_noise_with_cache(self, *args, **kwargs):
        del args, kwargs
        self._raise_cache_disabled()

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Sample one complete 22D action chunk without the single-expert KV cache."""
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "DexJoCo `infer_action` requires "
                "`video_attention_mask_mode='first_frame_causal'`."
            )
        if not isinstance(action_horizon, int) or isinstance(action_horizon, bool) or action_horizon <= 0:
            raise ValueError("DexJoCo `action_horizon` must be a positive integer.")
        if not isinstance(num_inference_steps, int) or num_inference_steps <= 0:
            raise ValueError("DexJoCo `num_inference_steps` must be a positive integer.")
        if negative_prompt not in (None, ""):
            raise ValueError("DexJoCo uncached action inference does not support negative_prompt.")
        if float(text_cfg_scale) != 1.0:
            raise ValueError("DexJoCo uncached action inference requires text_cfg_scale=1.0.")

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                "DexJoCo `input_image` must have shape [1,3,H,W] or [3,H,W], "
                f"got {tuple(input_image.shape)}."
            )
        if not input_image.is_floating_point():
            raise TypeError(
                f"DexJoCo `input_image` must use a floating dtype, got {input_image.dtype}."
            )
        if not torch.isfinite(input_image).all():
            raise ValueError("DexJoCo `input_image` contains NaN or Inf.")
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                "DexJoCo `input_image` spatial dimensions must be multiples of 16, "
                f"got HxW=({height},{width})."
            )

        if proprio is None:
            raise ValueError("DexJoCo `proprio` is required for action inference.")
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        if proprio.ndim != 2 or proprio.shape != (1, self.PROPRIO_DIM):
            raise ValueError(
                f"DexJoCo `proprio` must have shape [1,{self.PROPRIO_DIM}] or "
                f"[{self.PROPRIO_DIM}], got {tuple(proprio.shape)}."
            )
        if not proprio.is_floating_point():
            raise TypeError(f"DexJoCo `proprio` must use a floating dtype, got {proprio.dtype}.")
        if not torch.isfinite(proprio).all():
            raise ValueError("DexJoCo `proprio` contains NaN or Inf.")

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("DexJoCo `prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Provide either `prompt` or both `context/context_mask`.")
        if use_prompt:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("DexJoCo `prompt` must be a non-empty string.")
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("DexJoCo `context` and `context_mask` must be provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context.shape[0] != 1:
                raise ValueError(
                    "DexJoCo `context` must have shape [1,L,D] or [L,D], "
                    f"got {tuple(context.shape)}."
                )
            if context_mask.ndim != 2 or context_mask.shape != context.shape[:2]:
                raise ValueError(
                    "DexJoCo `context_mask` must match context [1,L], "
                    f"got {tuple(context_mask.shape)} for {tuple(context.shape)}."
                )
            if not context.is_floating_point() or not torch.isfinite(context).all():
                raise ValueError("DexJoCo `context` must be finite floating point.")
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )

        proprio = proprio.to(device=self.device, dtype=self.torch_dtype)
        context, context_mask = self._append_proprio_to_context(
            context=context,
            context_mask=context_mask,
            proprio=proprio,
        )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(
            input_image=input_image,
            tiled=tiled,
        )
        if not torch.isfinite(first_frame_latents).all():
            raise RuntimeError("DexJoCo VAE produced NaN or Inf first-frame latents.")

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.ACTION_DIM),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        timestep_video = torch.zeros(
            (1,),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
        infer_timesteps, infer_deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )

        last_outputs = None
        for step_t, step_delta in zip(infer_timesteps, infer_deltas):
            timestep_action = step_t.reshape(1).to(
                dtype=latents_action.dtype,
                device=self.device,
            )
            last_outputs = self._forward_dual_experts(
                latents_video=first_frame_latents,
                noisy_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                action_condition=None,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            prediction = last_outputs["action"]
            expected_shape = (1, action_horizon, self.ACTION_DIM)
            if prediction.shape != expected_shape:
                raise RuntimeError(
                    f"DexJoCo model output must have shape {expected_shape}, "
                    f"got {tuple(prediction.shape)}."
                )
            if not torch.isfinite(prediction).all():
                raise RuntimeError("DexJoCo model produced NaN or Inf action noise.")
            latents_action = self.infer_action_scheduler.step(
                prediction,
                step_delta,
                latents_action,
            )

        if last_outputs is None:
            raise RuntimeError("DexJoCo inference schedule produced no denoising steps.")
        if not torch.isfinite(latents_action).all():
            raise RuntimeError("DexJoCo sampler produced NaN or Inf actions.")
        arm_action, hand_action = self.split_action(latents_action)
        return {
            "action": latents_action.detach().to(device="cpu", dtype=torch.float32),
            "arm_action": arm_action.detach().to(device="cpu", dtype=torch.float32),
            "hand_action": hand_action.detach().to(device="cpu", dtype=torch.float32),
            "attention_mask": last_outputs["attention_mask"].detach().to(device="cpu"),
        }

    def _raise_phase2_joint_inference_disabled(self) -> None:
        message = (
            "DexJoCo joint inference is disabled in Phase 2 because the inherited FastWAM "
            "sampler allocates action latents from the 6D `action` expert and has no Hand Expert path."
        )
        logger.warning(message)
        raise NotImplementedError(message)

    @torch.no_grad()
    def infer_joint(self, *args, **kwargs):
        del args, kwargs
        self._raise_phase2_joint_inference_disabled()
