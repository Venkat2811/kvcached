# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""
ZeroMQ-based broadcast implementation for benchmarking.
This directly uses the ZeroMQ transport layer for IPC.
"""

import asyncio
from typing import List

# Import the ZeroMQ implementation directly
from kvcached.tp_ipc_zmq import (
    _broadcast_map_to_kv_tensors,
    _broadcast_unmap_from_kv_tensors,
)


async def broadcast_map_to_kv_tensors(tp_size: int, offsets: List[int]) -> None:
    """
    Broadcast map operation using ZeroMQ transport.
    This is an async implementation that leverages ZMQ's concurrent messaging.
    """
    await _broadcast_map_to_kv_tensors(tp_size, offsets)
