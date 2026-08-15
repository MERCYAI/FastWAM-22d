#!/usr/bin/env python
"""Serve a trained DexJoCo dual-action checkpoint over websocket."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fastwam.inference.dexjoco_policy import (
    DEFAULT_TEXT_ENCODER_ID,
    DexJoCoInferencePolicy,
    checkpoint_statistics_path,
    load_dexjoco_inference_model,
)
from fastwam.inference.websocket_server import DexJoCoWebsocketServer
from fastwam.utils.logging_config import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve versioned, denormalized DexJoCo 22D action chunks."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Phase 5 state directory containing manifest, model, and dataset_stats.json.",
    )
    parser.add_argument("--text-cache-dir", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--text-encoder-id", default=DEFAULT_TEXT_ENCODER_ID)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--sigma-shift", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rand-device", default="cpu")
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--max-request-mib", type=int, default=32)
    parser.add_argument(
        "--allow-non-production-statistics",
        action="store_true",
        help="Smoke only: permit production=false dataset statistics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(log_level=logging.INFO)
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    model = load_dexjoco_inference_model(
        checkpoint_dir,
        device=args.device,
        dtype=args.dtype,
        allow_non_production_statistics=args.allow_non_production_statistics,
    )
    policy = DexJoCoInferencePolicy(
        model,
        statistics_path=checkpoint_statistics_path(checkpoint_dir),
        text_cache_dir=args.text_cache_dir,
        horizon=args.horizon,
        image_size=(args.image_height, args.image_width),
        context_len=args.context_len,
        text_encoder_id=args.text_encoder_id,
        num_inference_steps=args.num_inference_steps,
        sigma_shift=args.sigma_shift,
        seed=args.seed,
        rand_device=args.rand_device,
        tiled=args.tiled,
        allow_non_production_statistics=args.allow_non_production_statistics,
    )
    server = DexJoCoWebsocketServer(
        policy,
        host=args.host,
        port=args.port,
        max_request_bytes=args.max_request_mib * 1024 * 1024,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
