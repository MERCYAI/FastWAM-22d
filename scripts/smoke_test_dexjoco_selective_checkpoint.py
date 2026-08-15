from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from fastwam.models.wan22.dexjoco_checkpoint import (
    REPORT_SCHEMA,
    REPORT_VERSION,
    SelectiveCheckpointError,
)
from fastwam.models.wan22.mot import MoT
from fastwam.utils.logging_config import setup_logging
from smoke_test_dexjoco_dual_action_model import (
    make_action_expert,
    make_dexjoco_model,
    make_video_expert,
)


SOURCE_PREFIX = "module.model.mot."


def make_old_fastwam_checkpoint() -> dict:
    video = make_video_expert(action_dim=7)
    action = make_action_expert(7)
    old_mot = MoT(
        mixtures={"video": video, "action": action},
        mot_checkpoint_mixed_attn=False,
    )

    mot_state = {}
    for index, (key, tensor) in enumerate(sorted(old_mot.state_dict().items())):
        fill_value = float(index + 1) / 1000.0
        mot_state[f"{SOURCE_PREFIX}{key}"] = torch.full_like(tensor, fill_value)

    # Old LIBERO proprio is deliberately incompatible with the new 23D input.
    # Both weight and bias remain policy-skipped even where bias shape matches.
    proprio_state = {
        "weight": torch.full((32, 8), 0.75),
        "bias": torch.full((32,), 0.5),
    }
    mot_state["module.model.mot.mixtures.aux.unused.weight"] = torch.ones(2, 3)
    return {
        "mot": mot_state,
        "proprio_encoder": proprio_state,
        "step": 123,
    }


def source_tensor(checkpoint: dict, local_key: str) -> torch.Tensor:
    return checkpoint["mot"][f"{SOURCE_PREFIX}{local_key}"]


def projection_target_names() -> set[str]:
    names = {
        f"{module}.{local_key}"
        for module in ("action", "hand")
        for local_key in (
            "action_encoder.weight",
            "action_encoder.bias",
            "head.weight",
            "head.bias",
        )
    }
    names.update(
        {
            "proprio_encoder.weight",
            "proprio_encoder.bias",
        }
    )
    return names


def assert_successful_selective_load(
    checkpoint: dict,
    checkpoint_path: Path,
    report_path: Path,
) -> dict[str, int]:
    model = make_dexjoco_model()
    with torch.no_grad():
        model.action_expert.action_encoder.weight.fill_(99.0)
        model.action_expert.action_encoder.bias.fill_(99.0)
        model.action_expert.head.weight.fill_(99.0)
        model.action_expert.head.bias.fill_(99.0)
        model.hand_expert.action_encoder.weight.fill_(99.0)
        model.hand_expert.action_encoder.bias.fill_(99.0)
        model.hand_expert.head.weight.fill_(99.0)
        model.hand_expert.head.bias.fill_(99.0)
        model.proprio_encoder.weight.fill_(99.0)
        model.proprio_encoder.bias.fill_(99.0)

    report = model.load_selective_pretrained_checkpoint(
        checkpoint_path,
        report_path=str(report_path),
    )
    assert report.applied is True
    assert report.checkpoint_format == "fastwam_mot_payload"
    assert report.summary["skipped_shape"] == 0
    assert report.summary["missing_in_checkpoint"] == 0
    assert report.summary["unexpected_in_checkpoint"] == 1

    video_key = "blocks.0.self_attn.q.weight"
    arm_key = "blocks.0.ffn.0.weight"
    hand_key = "blocks.0.cross_attn.k.weight"
    torch.testing.assert_close(
        model.video_expert.state_dict()[video_key],
        source_tensor(checkpoint, f"mixtures.video.{video_key}"),
    )
    torch.testing.assert_close(
        model.action_expert.state_dict()[arm_key],
        source_tensor(checkpoint, f"mixtures.action.{arm_key}"),
    )
    torch.testing.assert_close(
        model.hand_expert.state_dict()[hand_key],
        source_tensor(checkpoint, f"mixtures.action.{hand_key}"),
    )

    policy_targets = projection_target_names()
    skipped_policy_targets = {
        entry["target"] for entry in report.categories["skipped_policy"]
    }
    initialized_targets = {
        entry["target"] for entry in report.categories["newly_initialized"]
    }
    loaded_targets = {
        entry["target"]
        for category in ("loaded", "copied_to_hand")
        for entry in report.categories[category]
    }
    assert skipped_policy_targets == policy_targets
    assert initialized_targets == policy_targets
    assert loaded_targets.isdisjoint(policy_targets)
    assert all(
        entry["initialization_applied"]
        for entry in report.categories["newly_initialized"]
    )

    assert model.action_expert.action_encoder.weight.shape == (24, 6)
    assert model.action_expert.head.weight.shape == (6, 24)
    assert model.hand_expert.action_encoder.weight.shape == (24, 16)
    assert model.hand_expert.head.weight.shape == (16, 24)
    assert model.proprio_encoder.weight.shape == (32, 23)
    old_encoder_bias = source_tensor(
        checkpoint, "mixtures.action.action_encoder.bias"
    )
    assert old_encoder_bias.shape == model.action_expert.action_encoder.bias.shape
    assert old_encoder_bias.shape == model.hand_expert.action_encoder.bias.shape
    assert not torch.equal(old_encoder_bias, model.action_expert.action_encoder.bias)
    assert not torch.equal(old_encoder_bias, model.hand_expert.action_encoder.bias)
    assert not torch.all(model.action_expert.action_encoder.weight == 99.0)
    assert not torch.all(model.hand_expert.action_encoder.weight == 99.0)
    assert not torch.all(model.proprio_encoder.weight == 99.0)

    with report_path.open("r", encoding="utf-8") as file:
        saved_report = json.load(file)
    assert saved_report["schema"] == REPORT_SCHEMA
    assert saved_report["version"] == REPORT_VERSION
    assert saved_report["summary"] == report.summary
    assert saved_report["applied"] is True
    return report.summary


def assert_fail_fast_is_atomic(checkpoint: dict, report_path: Path) -> dict[str, int]:
    bad_state = {key: value.clone() for key, value in checkpoint["mot"].items()}
    missing_source_key = (
        f"{SOURCE_PREFIX}mixtures.video.blocks.0.self_attn.k.weight"
    )
    del bad_state[missing_source_key]
    mismatched_source_key = (
        f"{SOURCE_PREFIX}mixtures.action.blocks.0.self_attn.q.weight"
    )
    bad_state[mismatched_source_key] = bad_state[mismatched_source_key][:-1]

    model = make_dexjoco_model()
    untouched_key = "blocks.0.self_attn.v.weight"
    untouched_before = model.video_expert.state_dict()[untouched_key].clone()
    projection_before = model.action_expert.action_encoder.weight.clone()
    try:
        model.load_selective_pretrained_checkpoint(
            {
                "mot": bad_state,
                "proprio_encoder": {
                    key: value.clone()
                    for key, value in checkpoint["proprio_encoder"].items()
                },
            },
            report_path=str(report_path),
            source_name="synthetic-invalid-old-fastwam.pt",
        )
    except SelectiveCheckpointError as exc:
        report = exc.report
    else:
        raise AssertionError("Missing/shape-invalid backbone checkpoint must fail fast")

    assert report.applied is False
    assert report.summary["missing_in_checkpoint"] == 1
    assert report.summary["skipped_shape"] == 2
    assert all(
        not entry["initialization_applied"]
        for entry in report.categories["newly_initialized"]
    )
    torch.testing.assert_close(
        model.video_expert.state_dict()[untouched_key], untouched_before
    )
    torch.testing.assert_close(
        model.action_expert.action_encoder.weight, projection_before
    )
    return report.summary


def main() -> None:
    setup_logging()
    torch.manual_seed(11)
    checkpoint = make_old_fastwam_checkpoint()
    with tempfile.TemporaryDirectory(prefix="fastwam_phase3_") as temp_dir:
        temp_path = Path(temp_dir)
        checkpoint_path = temp_path / "synthetic_old_fastwam.pt"
        torch.save(checkpoint, checkpoint_path)
        success_summary = assert_successful_selective_load(
            checkpoint,
            checkpoint_path,
            temp_path / "success_report.json",
        )
        failure_summary = assert_fail_fast_is_atomic(
            checkpoint, temp_path / "failure_report.json"
        )

    print(f"successful classification counts={success_summary}")
    print(f"fail-fast classification counts={failure_summary}")
    print("video equality check=PASS key=blocks.0.self_attn.q.weight")
    print("arm equality check=PASS key=blocks.0.ffn.0.weight")
    print("hand remap equality check=PASS key=blocks.0.cross_attn.k.weight")
    print("policy projections absent from loaded/copied_to_hand=PASS")
    print("same-shape old/new action_encoder bias remains policy-skipped=PASS")
    print("projection shapes: arm_encoder=(24,6) arm_head=(6,24)")
    print("projection shapes: hand_encoder=(24,16) hand_head=(16,24)")
    print("missing/shape fail-fast before mutation=PASS")
    print(f"report schema={REPORT_SCHEMA}.v{REPORT_VERSION} JSON=PASS")


if __name__ == "__main__":
    main()
