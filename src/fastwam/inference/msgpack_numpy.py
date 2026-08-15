"""Small, pickle-free NumPy extension for msgpack.

The wire representation intentionally matches OpenPI's ``msgpack_numpy``
module so the existing DexJoCo OpenPI websocket client remains reusable.
"""

from __future__ import annotations

import functools
from typing import Any

import msgpack
import numpy as np


def pack_array(value: Any) -> Any:
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported NumPy dtype: {value.dtype}.")
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            b"__ndarray__": True,
            b"data": array.tobytes(),
            b"dtype": array.dtype.str,
            b"shape": array.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    return value


def unpack_array(value: dict) -> Any:
    if b"__ndarray__" in value:
        dtype = np.dtype(value[b"dtype"])
        if dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported NumPy dtype: {dtype}.")
        shape = tuple(int(size) for size in value[b"shape"])
        if any(size < 0 for size in shape):
            raise ValueError(f"Invalid NumPy shape: {shape}.")
        expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        data = value[b"data"]
        if expected_bytes != len(data):
            raise ValueError(
                f"NumPy payload byte count mismatch: expected {expected_bytes}, got {len(data)}."
            )
        return np.ndarray(buffer=data, dtype=dtype, shape=shape)
    if b"__npgeneric__" in value:
        dtype = np.dtype(value[b"dtype"])
        if dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported NumPy dtype: {dtype}.")
        return dtype.type(value[b"data"])
    return value


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)
