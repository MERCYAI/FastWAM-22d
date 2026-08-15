"""OpenPI-compatible websocket server for DexJoCo 22D action chunks."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import websockets
from websockets.asyncio.server import ServerConnection, serve

from fastwam.utils.logging_config import get_logger

from . import msgpack_numpy


logger = get_logger(__name__)


class DexJoCoWebsocketServer:
    def __init__(
        self,
        policy: Any,
        *,
        host: str = "0.0.0.0",
        port: int = 8000,
        max_request_bytes: int = 32 * 1024 * 1024,
    ) -> None:
        if not isinstance(host, str) or not host:
            raise ValueError("Websocket host must be non-empty.")
        if not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError("Websocket port must be in [0, 65535].")
        if not isinstance(max_request_bytes, int) or max_request_bytes <= 0:
            raise ValueError("max_request_bytes must be a positive integer.")
        metadata = policy.metadata
        if not isinstance(metadata, dict):
            raise TypeError("DexJoCo policy metadata must be a mapping.")
        self.policy = policy
        self.host = host
        self.port = port
        self.max_request_bytes = max_request_bytes
        self.metadata = metadata
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def create_server(self):
        """Return an async context manager; exposed for bounded integration tests."""
        return serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=self.max_request_bytes,
        )

    async def _handler(self, websocket: ServerConnection) -> None:
        logger.info("DexJoCo websocket connection opened from %s", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self.metadata))
        previous_total_ms = None
        try:
            async for message in websocket:
                if not isinstance(message, bytes):
                    raise TypeError("DexJoCo websocket requests must be binary msgpack frames.")
                start = time.monotonic()
                payload = msgpack_numpy.unpackb(message)
                infer_start = time.monotonic()
                response = self.policy.infer(payload)
                infer_ms = (time.monotonic() - infer_start) * 1000.0
                response["server_timing"] = {"infer_ms": infer_ms}
                if previous_total_ms is not None:
                    response["server_timing"]["prev_total_ms"] = previous_total_ms
                await websocket.send(packer.pack(response))
                previous_total_ms = (time.monotonic() - start) * 1000.0
        except websockets.ConnectionClosed:
            pass
        except Exception as exc:
            logger.exception("DexJoCo websocket request failed")
            if websocket.state.name == "OPEN":
                await websocket.send(f"{type(exc).__name__}: {exc}")
                await websocket.close(code=1011, reason="DexJoCo inference request failed")
        finally:
            logger.info("DexJoCo websocket connection closed from %s", websocket.remote_address)

    async def run(self) -> None:
        async with self.create_server() as server:
            sockets = server.sockets or []
            addresses = [socket.getsockname() for socket in sockets]
            logger.info("Serving FastWAM DexJoCo policy on %s", addresses)
            await server.serve_forever()

    def serve_forever(self) -> None:
        asyncio.run(self.run())
