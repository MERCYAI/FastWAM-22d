from __future__ import annotations

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
    def infer_action(self, *args, **kwargs):
        del args, kwargs
        self._raise_cache_disabled()

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
