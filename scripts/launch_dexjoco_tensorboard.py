#!/usr/bin/env python
from __future__ import annotations

import argparse
import signal
from pathlib import Path


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Launch TensorBoard for a DexJoCo run.")
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6006)
    args, extra = parser.parse_known_args()
    forbidden = ("--logdir", "--host", "--port", "--bind_all")
    for value in extra:
        if value in forbidden or any(value.startswith(f"{flag}=") for flag in forbidden):
            parser.error(f"Pass {value.split('=', 1)[0]} through the dedicated launcher option.")
    return args, extra


def main() -> None:
    args, extra = parse_args()
    logdir = args.logdir.expanduser().resolve()
    if not logdir.is_dir():
        raise FileNotFoundError(f"TensorBoard log directory does not exist: {logdir}")
    if not 1 <= args.port <= 65535:
        raise ValueError(f"TensorBoard port must be in [1, 65535], got {args.port}.")

    try:
        from tensorboard import program
    except ImportError as exc:
        raise ImportError(
            "TensorBoard is unavailable. Install the pinned project dependencies from "
            "pyproject.toml before running this launcher."
        ) from exc

    tensorboard = program.TensorBoard()
    tensorboard.configure(
        argv=[
            "tensorboard",
            "--logdir",
            str(logdir),
            "--host",
            args.host,
            "--port",
            str(args.port),
            *extra,
        ]
    )
    url = tensorboard.launch()
    print(f"TensorBoard logdir: {logdir}", flush=True)
    print(f"TensorBoard URL: {url}", flush=True)
    try:
        signal.pause()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
