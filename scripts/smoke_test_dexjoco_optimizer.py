from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.dexjoco_dual_action import DexJoCoDualActionFastWAM
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import WanVideoDiT
from fastwam.trainer import Wan22Trainer, build_lr_scheduler
from fastwam.utils.logging_config import setup_logging
from smoke_test_dexjoco_dual_action_model import (
    TinyVAE,
    make_action_expert,
    make_dexjoco_model,
    make_video_expert,
)


EXPECTED_GROUPS = ("action_new", "action_backbone", "video_backbone")


def compose_config() -> DictConfig:
    config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(
            config_name="train",
            overrides=["task=dexjoco_joint_2cam224_1e-4"],
        )


def prepare_optimizer(model, cfg: DictConfig):
    Wan22Trainer._apply_dit_only_train_mode(model)
    trainer = object.__new__(Wan22Trainer)
    trainer.model = model
    trainer.cfg = cfg
    trainer.learning_rate = float(cfg.learning_rate)
    trainer.weight_decay = float(cfg.weight_decay)
    return Wan22Trainer._build_optimizer(trainer)


def assert_optimizer_contract(model, optimizer) -> dict[str, dict[str, int | float]]:
    assert tuple(group["name"] for group in optimizer.param_groups) == EXPECTED_GROUPS
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    grouped_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    grouped_ids = [id(parameter) for parameter in grouped_parameters]
    assert len(grouped_ids) == len(set(grouped_ids))
    assert set(grouped_ids) == {id(parameter) for parameter in trainable_parameters}

    frozen_parameters = list(model.vae.parameters()) + list(model.text_encoder.parameters())
    assert frozen_parameters
    assert all(not parameter.requires_grad for parameter in frozen_parameters)
    assert {id(parameter) for parameter in frozen_parameters}.isdisjoint(grouped_ids)
    assert model.vae.training is False
    assert model.text_encoder.training is False
    assert model.video_expert.training is True
    assert model.action_expert.training is True
    assert model.hand_expert.training is True
    assert model.proprio_encoder.training is True

    summary = {}
    for group in optimizer.param_groups:
        summary[group["name"]] = {
            "lr": float(group["lr"]),
            "weight_decay": float(group["weight_decay"]),
            "tensors": len(group["params"]),
            "parameters": sum(parameter.numel() for parameter in group["params"]),
        }
    return summary


def parameter_group_name(optimizer, parameter: torch.nn.Parameter) -> str:
    matches = [
        group["name"]
        for group in optimizer.param_groups
        if any(candidate is parameter for candidate in group["params"])
    ]
    assert len(matches) == 1
    return str(matches[0])


def make_formal_meta_model(cfg: DictConfig) -> DexJoCoDualActionFastWAM:
    video_config = OmegaConf.to_container(cfg.model.video_dit_config, resolve=True)
    action_config = OmegaConf.to_container(cfg.model.action_dit_config, resolve=True)
    hand_config = OmegaConf.to_container(cfg.model.hand_dit_config, resolve=True)
    with torch.device("meta"):
        video = WanVideoDiT(**video_config)
        arm = ActionDiT(**action_config)
        hand = ActionDiT(**hand_config)
        mot = MoT(
            mixtures={"video": video, "action": arm, "hand": hand},
            mot_checkpoint_mixed_attn=bool(cfg.model.mot_checkpoint_mixed_attn),
        )
        vae = nn.Linear(1, 1)
        text_encoder = nn.Linear(1, 1)
    return DexJoCoDualActionFastWAM(
        video_expert=video,
        action_expert=arm,
        hand_expert=hand,
        mot=mot,
        vae=vae,
        text_encoder=text_encoder,
        text_dim=int(video_config["text_dim"]),
        proprio_dim=23,
        device="meta",
        torch_dtype=torch.float32,
    )


def make_legacy_model() -> FastWAM:
    video = make_video_expert(action_dim=7)
    action = make_action_expert(7)
    mot = MoT(
        mixtures={"video": video, "action": action},
        mot_checkpoint_mixed_attn=False,
    )
    return FastWAM(
        video_expert=video,
        action_expert=action,
        mot=mot,
        vae=TinyVAE(),
        text_dim=32,
        proprio_dim=8,
        device="cpu",
        torch_dtype=torch.float32,
    )


def main() -> None:
    setup_logging()
    cfg = compose_config()
    checkpoint_path = Path(str(cfg.model.selective_checkpoint_path)).resolve()
    assert checkpoint_path.is_file()
    assert int(cfg.model.action_dit_config.action_dim) == 6
    assert int(cfg.model.hand_dit_config.action_dim) == 16
    assert int(cfg.model.proprio_dim) == 23
    assert cfg.data.train.processor.delta_action_dim_mask is None

    tiny_model = make_dexjoco_model()
    tiny_model.text_encoder = nn.Sequential(nn.Linear(3, 4), nn.Dropout(0.1))
    tiny_model.vae.freeze_probe = nn.Linear(2, 2)
    tiny_optimizer = prepare_optimizer(tiny_model, cfg)
    tiny_summary = assert_optimizer_contract(tiny_model, tiny_optimizer)

    tiny_model.train()
    assert tiny_model.vae.training is True
    assert tiny_model.text_encoder.training is True
    Wan22Trainer._apply_dit_only_train_mode(tiny_model)
    assert_optimizer_contract(tiny_model, tiny_optimizer)

    representative_groups = {
        "action.action_encoder.weight": parameter_group_name(
            tiny_optimizer, tiny_model.action_expert.action_encoder.weight
        ),
        "hand.head.weight": parameter_group_name(
            tiny_optimizer, tiny_model.hand_expert.head.weight
        ),
        "proprio_encoder.weight": parameter_group_name(
            tiny_optimizer, tiny_model.proprio_encoder.weight
        ),
        "action.blocks.0.self_attn.q.weight": parameter_group_name(
            tiny_optimizer, tiny_model.action_expert.blocks[0].self_attn.q.weight
        ),
        "hand.blocks.0.ffn.0.weight": parameter_group_name(
            tiny_optimizer, tiny_model.hand_expert.blocks[0].ffn[0].weight
        ),
        "video.blocks.0.self_attn.q.weight": parameter_group_name(
            tiny_optimizer, tiny_model.video_expert.blocks[0].self_attn.q.weight
        ),
    }
    assert representative_groups == {
        "action.action_encoder.weight": "action_new",
        "hand.head.weight": "action_new",
        "proprio_encoder.weight": "action_new",
        "action.blocks.0.self_attn.q.weight": "action_backbone",
        "hand.blocks.0.ffn.0.weight": "action_backbone",
        "video.blocks.0.self_attn.q.weight": "video_backbone",
    }

    base_lrs = [float(group["lr"]) for group in tiny_optimizer.param_groups]
    scheduler = build_lr_scheduler(
        tiny_optimizer,
        scheduler_type="cosine",
        total_train_steps=20,
        warmup_steps=2,
    )
    tiny_optimizer.step()
    scheduler.step()
    stepped_lrs = [float(group["lr"]) for group in tiny_optimizer.param_groups]
    factors = [current / base for current, base in zip(stepped_lrs, base_lrs)]
    torch.testing.assert_close(
        torch.tensor(factors),
        torch.full((3,), factors[0]),
        rtol=1e-12,
        atol=1e-12,
    )
    assert stepped_lrs[0] > stepped_lrs[1] >= stepped_lrs[2]

    invalid_groups = OmegaConf.to_container(cfg.optimizer_groups, resolve=True)
    invalid_groups["action_new"]["lr"] = invalid_groups["action_backbone"]["lr"]
    invalid_groups = {
        name: {
            "lr": float(options["lr"]),
            "weight_decay": float(cfg.weight_decay),
        }
        for name, options in invalid_groups.items()
    }
    try:
        tiny_model.build_joint_post_training_parameter_groups(invalid_groups)
    except ValueError as exc:
        assert "action_new > action_backbone" in str(exc)
    else:
        raise AssertionError("Invalid DexJoCo LR ordering must fail fast")

    formal_model = make_formal_meta_model(cfg)
    formal_optimizer = prepare_optimizer(formal_model, cfg)
    formal_summary = assert_optimizer_contract(formal_model, formal_optimizer)

    legacy_model = make_legacy_model()
    legacy_cfg = OmegaConf.create(
        {
            "learning_rate": 1.0e-4,
            "weight_decay": 1.0e-2,
            "optimizer_groups": None,
        }
    )
    legacy_optimizer = prepare_optimizer(legacy_model, legacy_cfg)
    assert len(legacy_optimizer.param_groups) == 1
    assert legacy_optimizer.param_groups[0]["name"] == "default"

    print(f"checkpoint path exists={checkpoint_path}")
    print(f"tiny group summary={tiny_summary}")
    print(f"formal group summary={formal_summary}")
    print(f"representative assignments={representative_groups}")
    print(f"scheduler base_lrs={base_lrs}")
    print(f"scheduler stepped_lrs={stepped_lrs} common_factor={factors[0]:.6f}")
    print("trainable coverage=PASS overlap=0 frozen_in_optimizer=0")
    print("T5/VAE requires_grad=false eval=true after mode restore=PASS")
    print("invalid LR ordering fail-fast=PASS")
    print("legacy FastWAM optimizer groups=1 compatibility=PASS")
    print("training_started=false backward_called=false")


if __name__ == "__main__":
    main()
