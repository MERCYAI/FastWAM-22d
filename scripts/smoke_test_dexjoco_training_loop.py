from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from fastwam.datasets.lerobot.dexjoco_contract import DEXJOCO_TASKS
from fastwam.datasets.lerobot.dexjoco_robot_video_dataset import (
    DexJoCoRobotVideoDataset,
)
from fastwam.datasets.lerobot.processors.dexjoco_processor import (
    DexJoCoFastWAMProcessor,
)
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.transforms.action_state_merger import ConcatLeftAlign
from fastwam.datasets.lerobot.transforms.image import ToTensor
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.mot import MoT
from fastwam.trainer import (
    DEXJOCO_DATASET_STATS,
    DEXJOCO_SELECTIVE_REPORT,
    DEXJOCO_TRAINING_CONFIG,
    DEXJOCO_TRAINING_MANIFEST,
    Wan22Trainer,
    build_lr_scheduler,
)
from fastwam.utils import misc
from fastwam.utils.fs import ensure_dir
from smoke_test_dexjoco_dual_action_model import (
    CountingActionScheduler,
    make_action_expert,
    make_dexjoco_model,
    make_video_expert,
)


DEFAULT_DATA_ROOT = Path("/tmp/dexjoco_phase1_smoke.Rlts3g/data")
DEFAULT_STATS = Path(
    "/tmp/dexjoco_phase1_smoke.Rlts3g/hydra_work/dataset_stats.json"
)


class ResumeSamplerProbe:
    def __init__(self) -> None:
        self.epoch = None
        self.batch_offset = None

    def set_epoch_offset(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_resume_batch_offset(self, offset: int) -> None:
        self.batch_offset = int(offset)

    def clear_resume_batch_offset(self) -> None:
        self.batch_offset = 0


class FailIfTensorStateLoads:
    def __init__(self, model) -> None:
        self.model = model

    def unwrap_model(self, _model):
        return self.model

    def load_state(self, **_kwargs):
        raise AssertionError("Tensor state load was reached before contract validation")


def compose_config() -> DictConfig:
    config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(
            config_name="train",
            overrides=["task=dexjoco_joint_2cam224_1e-4"],
        )


def shape_meta() -> dict:
    return {
        "images": [
            {"key": "primary", "raw_shape": [3, 640, 640], "shape": [3, 16, 16]},
            {"key": "wrist", "raw_shape": [3, 640, 640], "shape": [3, 16, 16]},
        ],
        "action": [{"key": "default", "raw_shape": 22, "shape": 22}],
        "state": [{"key": "default", "raw_shape": 23, "shape": 23}],
    }


def build_processor(meta: dict) -> DexJoCoFastWAMProcessor:
    image_transforms = [ToTensor(), transforms.Resize([16, 16])]
    return DexJoCoFastWAMProcessor(
        shape_meta=meta,
        num_obs_steps=5,
        num_output_cameras=2,
        action_output_dim=22,
        proprio_output_dim=23,
        delta_action_dim_mask=None,
        allow_non_production_stats=True,
        expected_tasks=DEXJOCO_TASKS,
        action_state_transforms=None,
        use_stepwise_action_norm=False,
        norm_default_mode="min/max",
        norm_exception_mode=None,
        action_state_merger=ConcatLeftAlign(),
        train_transforms=image_transforms,
        val_transforms=image_transforms,
    )


def write_smoke_contexts(dataset, cache_dir: Path, context_len: int = 3) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for source in dataset.lerobot_dataset.sources:
        instruction = source.episodes[0].instruction
        prompt = DEFAULT_PROMPT.format(task=instruction)
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        torch.save(
            {
                "context": torch.zeros(context_len, 32, dtype=torch.float32),
                "mask": torch.ones(context_len, dtype=torch.bool),
            },
            cache_dir / f"{digest}.t5_len{context_len}.wan22ti2v5b.pt",
        )


def first_task_indices(dataset) -> dict[str, int]:
    indices: dict[str, int] = {}
    start = 0
    for _, episode in dataset.lerobot_dataset.selected_episodes:
        indices.setdefault(episode.task_name, start)
        start += episode.length
    return indices


def build_six_task_batch(
    data_root: Path,
    stats_path: Path,
    work_dir: Path,
) -> tuple[dict, Path]:
    misc.register_work_dir(work_dir)
    meta = shape_meta()
    cache_dir = work_dir / "text_cache"
    dataset = DexJoCoRobotVideoDataset(
        dataset_dirs=[str(data_root / task) for task in DEXJOCO_TASKS],
        task_names=DEXJOCO_TASKS,
        split="train",
        shape_meta=meta,
        num_frames=5,
        video_size=[16, 32],
        global_sample_stride=1,
        action_video_freq_ratio=1,
        val_set_proportion=0.0,
        is_training_set=True,
        skip_padding_as_possible=False,
        concat_multi_camera="horizontal",
        return_camera_videos=True,
        processor=build_processor(meta),
        pretrained_norm_stats=str(stats_path),
        text_embedding_cache_dir=str(cache_dir),
        context_len=3,
    )
    write_smoke_contexts(dataset, cache_dir)
    task_indices = first_task_indices(dataset)
    batch = next(
        iter(
            DataLoader(
                Subset(dataset, [task_indices[task] for task in DEXJOCO_TASKS]),
                batch_size=len(DEXJOCO_TASKS),
                shuffle=False,
                num_workers=0,
            )
        )
    )
    assert tuple(batch["task_name"]) == DEXJOCO_TASKS
    assert batch["action"].shape == (6, 4, 22)
    assert batch["arm_action"].shape == (6, 4, 6)
    assert batch["hand_action"].shape == (6, 4, 16)
    assert batch["proprio"].shape == (6, 4, 23)
    assert batch["camera_videos"].shape == (6, 2, 3, 5, 16, 16)
    assert batch["video"].shape == (6, 3, 5, 16, 32)
    return batch, work_dir / DEXJOCO_DATASET_STATS


def make_legacy_checkpoint_payload() -> dict:
    video = make_video_expert(action_dim=7)
    action = make_action_expert(7)
    model = FastWAM(
        video_expert=video,
        action_expert=action,
        mot=MoT(
            mixtures={"video": video, "action": action},
            mot_checkpoint_mixed_attn=False,
        ),
        vae=nn.Linear(1, 1),
        text_dim=32,
        proprio_dim=None,
        device="cpu",
        torch_dtype=torch.float32,
    )
    return {"mot": model.mot.state_dict()}


def prepare_model(selective_report_path: Path):
    model = make_dexjoco_model()
    model.text_encoder = nn.Sequential(nn.Linear(3, 4), nn.Dropout(0.1))
    model.vae.freeze_probe = nn.Linear(2, 2)
    report = model.load_selective_pretrained_checkpoint(
        make_legacy_checkpoint_payload(),
        report_path=selective_report_path,
        source_name="<synthetic-old-fastwam-for-phase5-smoke>",
    )
    model.selective_checkpoint_report = report.to_dict()
    model.configure_joint_post_training()
    return model


def prepare_optimizer(model, cfg: DictConfig):
    trainer = object.__new__(Wan22Trainer)
    trainer.model = model
    trainer.cfg = cfg
    trainer.learning_rate = float(cfg.learning_rate)
    trainer.weight_decay = float(cfg.weight_decay)
    return Wan22Trainer._build_optimizer(trainer)


def gradient_categories(model) -> dict[str, list[tuple[str, nn.Parameter]]]:
    arm_backbone_keys = model.action_expert.backbone_key_set(
        dict(model.action_expert.named_parameters())
    )
    hand_backbone_keys = model.hand_expert.backbone_key_set(
        dict(model.hand_expert.named_parameters())
    )
    return {
        "video_dit": list(model.video_expert.named_parameters()),
        "arm_backbone": [
            (name, parameter)
            for name, parameter in model.action_expert.named_parameters()
            if name in arm_backbone_keys
        ],
        "arm_projection_head": [
            (name, parameter)
            for name, parameter in model.action_expert.named_parameters()
            if name not in arm_backbone_keys
        ],
        "hand_backbone": [
            (name, parameter)
            for name, parameter in model.hand_expert.named_parameters()
            if name in hand_backbone_keys
        ],
        "hand_projection_head": [
            (name, parameter)
            for name, parameter in model.hand_expert.named_parameters()
            if name not in hand_backbone_keys
        ],
        "proprio_encoder": list(model.proprio_encoder.named_parameters()),
    }


def snapshot_categories(categories) -> dict[str, dict[str, torch.Tensor]]:
    return {
        category: {
            name: parameter.detach().clone() for name, parameter in parameters
        }
        for category, parameters in categories.items()
    }


def audit_gradients(categories) -> dict[str, dict[str, float | int]]:
    summary = {}
    for category, parameters in categories.items():
        gradients = [parameter.grad for _, parameter in parameters]
        assert gradients and all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        nonzero = [gradient for gradient in gradients if bool(torch.count_nonzero(gradient))]
        assert nonzero, f"No nonzero gradient in {category}"
        norm = torch.sqrt(
            sum(gradient.detach().float().pow(2).sum() for gradient in gradients)
        )
        summary[category] = {
            "tensors": len(gradients),
            "nonzero_tensors": len(nonzero),
            "grad_norm": float(norm),
        }
    return summary


def audit_parameter_updates(categories, before) -> dict[str, int]:
    changed = {}
    for category, parameters in categories.items():
        count = sum(
            not torch.equal(parameter.detach(), before[category][name])
            for name, parameter in parameters
        )
        assert count > 0, f"No trainable parameter changed in {category}"
        changed[category] = count
    return changed


def make_trainer_shell(
    *,
    model,
    optimizer,
    scheduler,
    accelerator,
    cfg,
    output_dir: Path,
):
    trainer = object.__new__(Wan22Trainer)
    trainer.model = model
    trainer.optimizer = optimizer
    trainer.scheduler = scheduler
    trainer.accelerator = accelerator
    trainer.cfg = cfg
    trainer.output_dir = str(output_dir)
    trainer.checkpoint_root = str(output_dir / "checkpoints")
    trainer.weights_dir = str(output_dir / "checkpoints" / "weights")
    trainer.state_dir = str(output_dir / "checkpoints" / "state")
    trainer.global_step = 1
    trainer.epoch = 0
    trainer.batch_in_epoch = 1
    trainer.batch_size = 6
    trainer.train_sampler = ResumeSamplerProbe()
    for path in (trainer.checkpoint_root, trainer.weights_dir, trainer.state_dir):
        ensure_dir(path)
    return trainer


def assert_resume_fails_before_tensor_load(
    state_path: Path,
    model,
    cfg,
    expected_message: str,
) -> None:
    trainer = object.__new__(Wan22Trainer)
    trainer.model = model
    trainer.accelerator = FailIfTensorStateLoads(model)
    trainer.cfg = cfg
    trainer.train_sampler = ResumeSamplerProbe()
    try:
        trainer.load_training_state(str(state_path))
    except (ValueError, FileNotFoundError) as exc:
        assert expected_message in str(exc), str(exc)
    else:
        raise AssertionError("Invalid DexJoCo resume contract must fail")


def mutate_json(path: Path, mutation) -> None:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    mutation(payload)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=True, indent=2)
        file.write("\n")


def main() -> None:
    data_root = Path(os.environ.get("DEXJOCO_PHASE5_SMOKE_ROOT", DEFAULT_DATA_ROOT))
    stats_path = Path(os.environ.get("DEXJOCO_PHASE5_SMOKE_STATS", DEFAULT_STATS))
    if not data_root.is_dir() or not stats_path.is_file():
        raise FileNotFoundError(
            "Phase 5 smoke fixture is missing. Set DEXJOCO_PHASE5_SMOKE_ROOT and "
            "DEXJOCO_PHASE5_SMOKE_STATS to the six-task fixture and smoke statistics."
        )

    torch.manual_seed(17)
    with tempfile.TemporaryDirectory(prefix="fastwam_dexjoco_phase5_") as temp_dir:
        temp = Path(temp_dir)
        output_dir = temp / "run"
        output_dir.mkdir()
        accelerator = Accelerator(
            cpu=True,
            mixed_precision="no",
            step_scheduler_with_optimizer=False,
        )
        batch, copied_stats = build_six_task_batch(
            data_root.resolve(),
            stats_path.resolve(),
            output_dir,
        )
        selective_report_path = output_dir / "dexjoco_selective_load_report.json"
        model = prepare_model(selective_report_path)
        counting_scheduler = CountingActionScheduler(model.train_action_scheduler)
        model.train_action_scheduler = counting_scheduler

        cfg = compose_config()
        with open_dict(cfg):
            cfg.output_dir = str(output_dir)
            cfg.max_steps = 1
            cfg.num_workers = 0
            cfg.batch_size = 6
            cfg.mixed_precision = "no"
            cfg.data.dexjoco_root = str(data_root.resolve())
            cfg.data.train.pretrained_norm_stats = str(stats_path.resolve())
            cfg.data.train.processor.allow_non_production_stats = True
            cfg.model.selective_checkpoint_report_path = str(selective_report_path)

        optimizer = prepare_optimizer(model, cfg)
        scheduler = build_lr_scheduler(
            optimizer,
            scheduler_type="cosine",
            total_train_steps=4,
            warmup_steps=0,
        )
        categories = gradient_categories(model)
        before = snapshot_categories(categories)
        frozen_parameters = list(model.text_encoder.parameters()) + list(model.vae.parameters())
        assert frozen_parameters and all(not parameter.requires_grad for parameter in frozen_parameters)
        frozen_before = [parameter.detach().clone() for parameter in frozen_parameters]

        loss, losses, outputs = model.training_loss(batch, return_outputs=True)
        assert torch.isfinite(loss)
        assert set(losses) == {"loss_video", "loss_action"}
        assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
        torch.testing.assert_close(
            loss.detach().float(),
            torch.tensor(losses["loss_video"] + losses["loss_action"]),
        )
        assert outputs["arm_action"].shape == (6, 4, 6)
        assert outputs["hand_action"].shape == (6, 4, 16)
        assert outputs["action"].shape == (6, 4, 22)
        assert outputs["target_action"].shape == (6, 4, 22)
        assert outputs["timestep_action"].shape == (6,)
        assert all(
            torch.isfinite(outputs[key]).all()
            for key in ("video", "arm_action", "hand_action", "action", "target_action")
        )
        assert counting_scheduler.sample_calls == 1
        assert counting_scheduler.add_noise_shapes == [(6, 4, 22)]

        action_loss_token = F.mse_loss(
            outputs["action"].float(),
            outputs["target_action"].float(),
            reduction="none",
        ).mean(dim=2)
        valid = (~batch["action_is_pad"]).to(action_loss_token.dtype)
        action_per_sample = (action_loss_token * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        action_weight = counting_scheduler.training_weight(outputs["timestep_action"])
        manual_action_loss = (action_per_sample * action_weight).mean()
        torch.testing.assert_close(
            manual_action_loss,
            torch.tensor(losses["loss_action"]),
            rtol=1e-5,
            atol=1e-6,
        )

        loss.backward()
        gradient_summary = audit_gradients(categories)
        assert all(parameter.grad is None for parameter in frozen_parameters)
        optimizer.step()
        scheduler.step()
        changed_summary = audit_parameter_updates(categories, before)
        assert all(
            torch.equal(parameter.detach(), snapshot)
            for parameter, snapshot in zip(frozen_parameters, frozen_before, strict=True)
        )

        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
        trainer = make_trainer_shell(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            accelerator=accelerator,
            cfg=cfg,
            output_dir=output_dir,
        )
        checkpoint = trainer.save_checkpoint()
        state_path = Path(checkpoint["state_path"])
        weights_path = Path(checkpoint["weights_path"])
        assert weights_path.is_file()
        for filename in (
            DEXJOCO_TRAINING_MANIFEST,
            DEXJOCO_TRAINING_CONFIG,
            DEXJOCO_DATASET_STATS,
            DEXJOCO_SELECTIVE_REPORT,
            "trainer_state.json",
        ):
            assert (state_path / filename).is_file()
        accelerate_files = {path.name for path in state_path.rglob("*") if path.is_file()}
        assert any("model" in name for name in accelerate_files)
        assert any("optimizer" in name or "optim_states" in name for name in accelerate_files)
        assert any("scheduler" in name for name in accelerate_files)

        source_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in accelerator.unwrap_model(model).state_dict().items()
        }
        source_optimizer_state = copy.deepcopy(optimizer.state_dict())
        source_scheduler_state = copy.deepcopy(scheduler.state_dict())

        reload_report_path = temp / "reload_selective_report.json"
        reloaded_model = prepare_model(reload_report_path)
        reloaded_optimizer = prepare_optimizer(reloaded_model, cfg)
        reloaded_scheduler = build_lr_scheduler(
            reloaded_optimizer,
            scheduler_type="cosine",
            total_train_steps=4,
            warmup_steps=0,
        )
        reload_accelerator = Accelerator(
            cpu=True,
            mixed_precision="no",
            step_scheduler_with_optimizer=False,
        )
        reloaded_model, reloaded_optimizer, reloaded_scheduler = reload_accelerator.prepare(
            reloaded_model,
            reloaded_optimizer,
            reloaded_scheduler,
        )
        reload_trainer = make_trainer_shell(
            model=reloaded_model,
            optimizer=reloaded_optimizer,
            scheduler=reloaded_scheduler,
            accelerator=reload_accelerator,
            cfg=cfg,
            output_dir=output_dir,
        )
        reload_trainer.global_step = 0
        reload_trainer.load_training_state(str(state_path))
        assert reload_trainer.global_step == 1
        assert reload_trainer.batch_in_epoch == 1
        for name, tensor in reload_accelerator.unwrap_model(reloaded_model).state_dict().items():
            torch.testing.assert_close(tensor.detach().cpu(), source_state[name], rtol=0, atol=0)
        reloaded_optimizer_state = reloaded_optimizer.state_dict()
        assert len(reloaded_optimizer_state["state"]) == len(source_optimizer_state["state"])
        assert [group["name"] for group in reloaded_optimizer_state["param_groups"]] == [
            group["name"] for group in source_optimizer_state["param_groups"]
        ]
        assert [group["lr"] for group in reloaded_optimizer_state["param_groups"]] == [
            group["lr"] for group in source_optimizer_state["param_groups"]
        ]
        assert reloaded_scheduler.state_dict() == source_scheduler_state

        invalid_cases = (
            (
                "invalid_action_dim",
                DEXJOCO_TRAINING_MANIFEST,
                lambda payload: payload["dimensions"].__setitem__("action_dim", 21),
                "dimensions do not match",
            ),
            (
                "invalid_proprio_dim",
                DEXJOCO_TRAINING_MANIFEST,
                lambda payload: payload["dimensions"].__setitem__("proprio_dim", 22),
                "dimensions do not match",
            ),
            (
                "invalid_statistics_schema",
                DEXJOCO_DATASET_STATS,
                lambda payload: payload.__setitem__("schema_name", "libero.dataset_stats"),
                "statistics `schema_name`",
            ),
        )
        contract_model = reload_accelerator.unwrap_model(reloaded_model)
        for case_name, filename, mutation, expected_message in invalid_cases:
            invalid_state = temp / case_name
            shutil.copytree(state_path, invalid_state)
            mutate_json(invalid_state / filename, mutation)
            assert_resume_fails_before_tensor_load(
                invalid_state,
                contract_model,
                cfg,
                expected_message,
            )

        print(f"tasks={','.join(DEXJOCO_TASKS)}")
        print(
            "batch shapes: "
            f"action={tuple(batch['action'].shape)} state={tuple(batch['proprio'].shape)} "
            f"cameras={tuple(batch['camera_videos'].shape)} video={tuple(batch['video'].shape)}"
        )
        print(
            f"loss total={float(loss.detach()):.8f} "
            f"video={losses['loss_video']:.8f} action={losses['loss_action']:.8f}"
        )
        print(
            "predictions: "
            f"arm={tuple(outputs['arm_action'].shape)} "
            f"hand={tuple(outputs['hand_action'].shape)} "
            f"action={tuple(outputs['action'].shape)}"
        )
        print(
            "action diffusion: sample_t_calls=1 add_noise_shape=(6, 4, 22) "
            "shared_arm_hand_timestep=PASS full_22d_mse=PASS"
        )
        print(f"gradient summary={gradient_summary}")
        print(f"changed parameter tensors={changed_summary}")
        print("T5/VAE gradients=None parameters_unchanged=PASS")
        print(f"checkpoint accelerate files={sorted(accelerate_files)}")
        print(
            "checkpoint contents=model+optimizer+scheduler+config+dimensions+stats+selective_report=PASS"
        )
        print("save/reload tensor equality=PASS optimizer=PASS scheduler=PASS progress=PASS")
        print("resume fail-fast action_dim=PASS proprio_dim=PASS statistics_schema=PASS")
        print(f"smoke statistics copied={copied_stats} production=false")
        print("optimizer_steps=1 formal_training_started=false temporary_checkpoint_removed=true")


if __name__ == "__main__":
    main()
