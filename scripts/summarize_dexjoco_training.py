#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Sequence

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


LOSS_TAGS = {
    "total": "train/loss_total",
    "video": "train/loss_video",
    "action_22d": "train/loss_action_22d",
    "arm_6d_diagnostic": "train/loss_arm_6d",
    "hand_16d_diagnostic": "train/loss_hand_16d",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize DexJoCo TensorBoard loss convergence diagnostics."
    )
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--min-points", type=int, default=20)
    parser.add_argument("--decrease-threshold", type=float, default=0.05)
    parser.add_argument("--diverge-threshold", type=float, default=0.05)
    parser.add_argument("--plateau-relative-slope", type=float, default=1e-3)
    parser.add_argument("--unstable-cv", type=float, default=0.5)
    return parser.parse_args()


def _window_stats(values: Sequence[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "median": float(median(values))}


def summarize_series(
    events,
    *,
    window: int,
    min_points: int,
    decrease_threshold: float,
    diverge_threshold: float,
    plateau_relative_slope: float,
    unstable_cv: float,
) -> dict:
    raw_steps = [int(event.step) for event in events]
    raw_values = [float(event.value) for event in events]
    finite = [
        (step, value)
        for step, value in zip(raw_steps, raw_values, strict=True)
        if math.isfinite(value)
    ]
    nonfinite_count = len(raw_values) - len(finite)
    result = {
        "num_points": len(raw_values),
        "num_valid_points": len(finite),
        "nan_inf_count": nonfinite_count,
        "last_valid_step": finite[-1][0] if finite else None,
        "first_window": None,
        "last_window": None,
        "relative_reduction": None,
        "final_window_slope": None,
        "final_window_relative_slope": None,
        "final_window_coefficient_of_variation": None,
        "status": "insufficient_data",
    }
    if not finite:
        return result

    effective_window = min(int(window), len(finite))
    first = finite[:effective_window]
    last = finite[-effective_window:]
    first_values = [value for _, value in first]
    last_values = [value for _, value in last]
    first_stats = _window_stats(first_values)
    last_stats = _window_stats(last_values)
    denominator = max(abs(first_stats["mean"]), 1e-12)
    relative_reduction = (first_stats["mean"] - last_stats["mean"]) / denominator

    if len(last) >= 2:
        steps = np.asarray([step for step, _ in last], dtype=np.float64)
        values = np.asarray(last_values, dtype=np.float64)
        slope = float(np.polyfit(steps, values, 1)[0])
    else:
        slope = 0.0
    relative_slope = slope / max(abs(last_stats["mean"]), 1e-12)
    coefficient_of_variation = float(
        np.std(last_values) / max(abs(last_stats["mean"]), 1e-12)
    )

    result.update(
        {
            "first_window": first_stats,
            "last_window": last_stats,
            "relative_reduction": float(relative_reduction),
            "final_window_slope": slope,
            "final_window_relative_slope": float(relative_slope),
            "final_window_coefficient_of_variation": coefficient_of_variation,
        }
    )
    if len(finite) < min_points or len(finite) < 2 * window:
        return result
    if nonfinite_count > 0 or coefficient_of_variation > unstable_cv:
        result["status"] = "unstable"
    elif relative_reduction <= -diverge_threshold or relative_slope > plateau_relative_slope:
        result["status"] = "diverging"
    elif abs(relative_slope) <= plateau_relative_slope:
        result["status"] = "plateau/converged"
    elif relative_reduction >= decrease_threshold and relative_slope < 0:
        result["status"] = "decreasing"
    else:
        result["status"] = "unstable"
    return result


def load_and_summarize(logdir: Path, args: argparse.Namespace) -> dict:
    if not logdir.is_dir():
        raise FileNotFoundError(f"TensorBoard log directory does not exist: {logdir}")
    accumulator = EventAccumulator(str(logdir), size_guidance={"scalars": 0})
    accumulator.Reload()
    available_tags = set(accumulator.Tags().get("scalars", []))
    losses = {}
    for name, tag in LOSS_TAGS.items():
        events = accumulator.Scalars(tag) if tag in available_tags else []
        losses[name] = {
            "tag": tag,
            **summarize_series(
                events,
                window=args.window,
                min_points=args.min_points,
                decrease_threshold=args.decrease_threshold,
                diverge_threshold=args.diverge_threshold,
                plateau_relative_slope=args.plateau_relative_slope,
                unstable_cv=args.unstable_cv,
            ),
        }
    return {
        "schema": "fastwam.dexjoco.training_summary@1",
        "logdir": str(logdir),
        "engineering_diagnostic_only": True,
        "parameters": {
            "window": args.window,
            "min_points": args.min_points,
            "decrease_threshold": args.decrease_threshold,
            "diverge_threshold": args.diverge_threshold,
            "plateau_relative_slope": args.plateau_relative_slope,
            "unstable_cv": args.unstable_cv,
        },
        "losses": losses,
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# DexJoCo Training Convergence Summary",
        "",
        f"- TensorBoard logdir: `{summary['logdir']}`",
        "- Scope: engineering loss-curve diagnostics only; this does not replace simulator success metrics.",
        "",
        "| Loss | First mean / median | Last mean / median | Relative reduction | Final slope | Final CV | Non-finite | Last step | Status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, metrics in summary["losses"].items():
        first = metrics["first_window"]
        last = metrics["last_window"]
        first_text = "n/a" if first is None else f"{first['mean']:.6g} / {first['median']:.6g}"
        last_text = "n/a" if last is None else f"{last['mean']:.6g} / {last['median']:.6g}"
        reduction = metrics["relative_reduction"]
        slope = metrics["final_window_slope"]
        cv = metrics["final_window_coefficient_of_variation"]
        lines.append(
            "| {name} | {first} | {last} | {reduction} | {slope} | {cv} | {bad} | {step} | {status} |".format(
                name=name,
                first=first_text,
                last=last_text,
                reduction="n/a" if reduction is None else f"{reduction:.2%}",
                slope="n/a" if slope is None else f"{slope:.6g}",
                cv="n/a" if cv is None else f"{cv:.4f}",
                bad=metrics["nan_inf_count"],
                step=metrics["last_valid_step"] if metrics["last_valid_step"] is not None else "n/a",
                status=metrics["status"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.window <= 0 or args.min_points <= 0:
        raise ValueError("`--window` and `--min-points` must be positive.")
    logdir = args.logdir.expanduser().resolve()
    summary = load_and_summarize(logdir, args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=True, indent=2)
        stream.write("\n")
    args.output_markdown.write_text(render_markdown(summary), encoding="utf-8")
    print(f"json={args.output_json.resolve()}")
    print(f"markdown={args.output_markdown.resolve()}")
    for name, metrics in summary["losses"].items():
        print(
            f"{name}: status={metrics['status']} points={metrics['num_points']} "
            f"nan_inf={metrics['nan_inf_count']} last_step={metrics['last_valid_step']}"
        )


if __name__ == "__main__":
    main()
