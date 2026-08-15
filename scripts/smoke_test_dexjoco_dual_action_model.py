from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.dexjoco_dual_action import DexJoCoDualActionFastWAM
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import WanVideoDiT


class TinyVAE(nn.Module):
    temporal_downsample_factor = 4
    upsampling_factor = 8

    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(z_dim=4)

    def encode(self, video, **_):
        if isinstance(video, list):
            video = torch.stack(video)
        spatial = F.avg_pool3d(video, kernel_size=(1, 8, 8), stride=(1, 8, 8))
        first = spatial[:, :, :1]
        if spatial.shape[2] > 1:
            tail = spatial[:, :, 1:].mean(dim=2, keepdim=True)
            latent = torch.cat([first, tail], dim=2)
        else:
            latent = first
        return torch.cat([latent, latent.mean(dim=1, keepdim=True)], dim=1)


class CountingActionScheduler:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.sample_calls = 0
        self.add_noise_shapes = []

    def sample_training_t(self, *args, **kwargs):
        self.sample_calls += 1
        return self.scheduler.sample_training_t(*args, **kwargs)

    def add_noise(self, original_samples, noise, timestep):
        self.add_noise_shapes.append(tuple(original_samples.shape))
        return self.scheduler.add_noise(original_samples, noise, timestep)

    def training_target(self, *args, **kwargs):
        return self.scheduler.training_target(*args, **kwargs)

    def training_weight(self, *args, **kwargs):
        return self.scheduler.training_weight(*args, **kwargs)


def make_video_expert(action_dim: int = 22) -> WanVideoDiT:
    return WanVideoDiT(
        hidden_dim=24,
        in_dim=4,
        ffn_dim=48,
        out_dim=4,
        text_dim=32,
        freq_dim=16,
        eps=1e-6,
        patch_size=(1, 1, 1),
        num_heads=3,
        attn_head_dim=8,
        num_layers=1,
        has_image_input=False,
        seperated_timestep=True,
        require_vae_embedding=False,
        require_clip_embedding=False,
        fuse_vae_embedding_in_latents=True,
        action_conditioned=False,
        action_dim=action_dim,
        video_attention_mask_mode="first_frame_causal",
    )


def make_action_expert(action_dim: int) -> ActionDiT:
    return ActionDiT(
        hidden_dim=24,
        action_dim=action_dim,
        ffn_dim=48,
        text_dim=32,
        freq_dim=16,
        eps=1e-6,
        num_heads=3,
        attn_head_dim=8,
        num_layers=1,
    )


def make_dexjoco_model() -> DexJoCoDualActionFastWAM:
    video = make_video_expert()
    arm = make_action_expert(6)
    hand = make_action_expert(16)
    mot = MoT(
        mixtures={"video": video, "action": arm, "hand": hand},
        mot_checkpoint_mixed_attn=False,
    )
    return DexJoCoDualActionFastWAM(
        video_expert=video,
        action_expert=arm,
        hand_expert=hand,
        mot=mot,
        vae=TinyVAE(),
        text_dim=32,
        proprio_dim=23,
        device="cpu",
        torch_dtype=torch.float32,
        video_num_train_timesteps=10,
        action_num_train_timesteps=10,
    )


def assert_attention_contract(mask: torch.Tensor, video_len: int, arm_len: int, hand_len: int):
    assert mask.shape == (video_len + arm_len + hand_len,) * 2
    action_start = video_len
    assert not mask[:video_len, action_start:].any(), "video must not read Arm/Hand tokens"
    assert mask[action_start:, :video_len].all(), "Arm/Hand must read all video tokens"
    assert mask[action_start:, action_start:].all(), "Arm/Hand interaction must be bidirectional"


def main() -> None:
    torch.manual_seed(7)
    model = make_dexjoco_model()
    counting_scheduler = CountingActionScheduler(model.train_action_scheduler)
    model.train_action_scheduler = counting_scheduler

    sample = {
        "video": torch.randn(1, 3, 5, 16, 16),
        "action": torch.randn(1, 4, 22),
        "proprio": torch.randn(1, 4, 23),
        "context": torch.randn(1, 3, 32),
        "context_mask": torch.ones(1, 3, dtype=torch.bool),
        "action_is_pad": torch.zeros(1, 4, dtype=torch.bool),
        "image_is_pad": torch.zeros(1, 5, dtype=torch.bool),
    }
    assert sample["action"].shape[-1] == 22
    try:
        model.split_action(torch.randn(1, 4, 21))
    except ValueError as exc:
        assert "22" in str(exc)
    else:
        raise AssertionError("DexJoCo model must reject non-22D action input")
    loss, loss_dict, outputs = model.training_loss(sample, return_outputs=True)

    assert loss.ndim == 0 and torch.isfinite(loss)
    assert set(loss_dict) == {"loss_video", "loss_action"}
    assert outputs["arm_action"].shape == (1, 4, 6)
    assert outputs["hand_action"].shape == (1, 4, 16)
    assert outputs["action"].shape == (1, 4, 22)
    torch.testing.assert_close(
        outputs["action"],
        torch.cat([outputs["arm_action"], outputs["hand_action"]], dim=-1),
    )
    assert_attention_contract(outputs["attention_mask"], video_len=8, arm_len=4, hand_len=4)
    assert counting_scheduler.sample_calls == 1
    assert counting_scheduler.add_noise_shapes == [(1, 4, 22)]
    assert model.proprio_encoder.in_features == 23
    assert all(parameter.requires_grad for parameter in model.proprio_encoder.parameters())
    assert model.supports_video_kv_cache is False

    try:
        model._predict_action_noise_with_cache()
    except NotImplementedError as exc:
        assert "cache" in str(exc).lower()
    else:
        raise AssertionError("DexJoCo single-action cache must remain disabled")

    libero_video = make_video_expert(action_dim=7)
    libero_action = make_action_expert(7)
    libero_mot = MoT(
        mixtures={"video": libero_video, "action": libero_action},
        mot_checkpoint_mixed_attn=False,
    )
    libero_model = FastWAM(
        video_expert=libero_video,
        action_expert=libero_action,
        mot=libero_mot,
        vae=TinyVAE(),
        text_dim=32,
        proprio_dim=8,
        device="cpu",
        torch_dtype=torch.float32,
        video_num_train_timesteps=10,
        action_num_train_timesteps=10,
    )
    assert tuple(libero_model.mot.expert_order) == ("video", "action")
    assert libero_model.action_expert.action_dim == 7

    print(f"loss={float(loss.detach()):.6f} loss_dict={loss_dict}")
    print("experts=('video', 'action', 'hand') action_name=arm")
    print(
        "forward shapes: "
        f"arm={tuple(outputs['arm_action'].shape)} "
        f"hand={tuple(outputs['hand_action'].shape)} "
        f"action={tuple(outputs['action'].shape)}"
    )
    print(f"attention_mask={tuple(outputs['attention_mask'].shape)}")
    print("action scheduler: timestep_calls=1 add_noise_shape=(1, 4, 22)")
    print("proprio_encoder.in_features=23 single-action video_kv_cache=disabled-explicit")
    print("legacy FastWAM: experts=('video', 'action') action_dim=7 instantiate=PASS")


if __name__ == "__main__":
    main()
