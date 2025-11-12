# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""
ZeroMQ-based IPC transport layer for kvcached.

This module provides an alternative to UNIX domain sockets for inter-process
communication between workers. ZeroMQ offers better performance for:
- Large message sizes (8-32MB KV cache blocks)
- Multi-worker broadcast scenarios
- Zero-copy message passing

Inspired by vLLM's ZeroMQ implementation for high-performance IPC.

NOTE: Future Enhancement - Shared Memory Ring Buffer
------------------------------------------------------
For even better performance on same-node communication, consider implementing
a shared memory ring buffer similar to vLLM's ShmRingBuffer. This would provide:
- True zero-copy for local workers (no serialization overhead)
- Lock-free single-writer, multiple-reader design
- Fallback to ZeroMQ for messages exceeding buffer size
- Hybrid approach: SHM for local + ZMQ for remote workers

See vLLM's implementation at:
vllm/distributed/device_communicators/shm_broadcast.py
"""

import asyncio
import os
import pickle
import struct
from typing import Any, Dict, List, Optional, cast

import zmq
import zmq.asyncio

from kvcached.vmm_ops import kv_tensors_created, map_to_kv_tensors, unmap_from_kv_tensors

# ZMQ configuration
ZMQ_IPC_DIR = "/tmp/kvcached-zmq-ipc"
ZMQ_MAX_CHUNK_SIZE_MB = int(os.getenv("KVCACHED_ZMQ_MAX_CHUNK_SIZE_MB", "32"))
ZMQ_MAX_CHUNK_SIZE = ZMQ_MAX_CHUNK_SIZE_MB * 1024 * 1024  # Convert to bytes

# Message types
Message = Dict[str, Any]


def get_worker_zmq_path(rank: int) -> str:
    """
    Get the IPC path for the worker's ZeroMQ socket.
    Uses IPC transport (UNIX domain sockets) for same-machine communication.
    """
    return f"ipc://{ZMQ_IPC_DIR}/worker_{rank}.ipc"


class ZMQMessageSerializer:
    """
    Handles serialization of messages with support for zero-copy of large payloads.

    For messages with large offset lists (common with KV cache blocks), we can
    use ZeroMQ's multipart messaging to avoid copying large arrays.
    """

    @staticmethod
    def serialize(msg: Message) -> List[bytes]:
        """
        Serialize a message into a list of frames for ZeroMQ multipart sending.

        For large offset arrays, we extract them and send as separate frames
        to enable zero-copy transmission.

        Returns:
            List of byte frames: [metadata_frame, data_frame1, data_frame2, ...]
        """
        # Check if message contains large offset arrays
        offsets = msg.get("offsets")

        if offsets and isinstance(offsets, list) and len(offsets) > 100:
            # Large offset array - send as separate frame for potential zero-copy
            # Create metadata frame without offsets
            metadata = msg.copy()
            metadata["offsets"] = None  # Placeholder
            metadata["_has_offsets_frame"] = True

            # Serialize metadata and offsets separately
            metadata_bytes = pickle.dumps(metadata)

            # Convert offsets list to bytes efficiently using struct
            # Format: 4 bytes for count, then count * 8 bytes for int64 offsets
            offsets_bytes = struct.pack(f"<I{len(offsets)}q", len(offsets), *offsets)

            return [metadata_bytes, offsets_bytes]
        else:
            # Small message - single frame
            return [pickle.dumps(msg)]

    @staticmethod
    def deserialize(frames: List[bytes]) -> Message:
        """
        Deserialize a message from ZeroMQ frames.

        Args:
            frames: List of byte frames received from ZeroMQ

        Returns:
            Deserialized message dictionary
        """
        if len(frames) == 1:
            # Single frame - standard pickle deserialization
            return cast(Message, pickle.loads(frames[0]))
        else:
            # Multi-frame message with separate offset array
            metadata = cast(Message, pickle.loads(frames[0]))

            if metadata.get("_has_offsets_frame"):
                # Deserialize offset array from second frame
                offsets_bytes = frames[1]
                count = struct.unpack("<I", offsets_bytes[:4])[0]
                offsets = list(struct.unpack(f"<{count}q", offsets_bytes[4:]))
                metadata["offsets"] = offsets
                del metadata["_has_offsets_frame"]

            return metadata


class ZMQWorkerListener:
    """
    ZeroMQ-based worker listener that handles incoming commands.
    Uses ROUTER socket for request-reply pattern with multiple clients.
    """

    def __init__(self, rank: int):
        self.rank = rank
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVHWM, 0)  # Unlimited receive high water mark
        self.socket.setsockopt(zmq.SNDHWM, 0)  # Unlimited send high water mark
        self.running = False

    async def start(self):
        """Start the ZMQ worker listener."""
        os.makedirs(ZMQ_IPC_DIR, exist_ok=True)
        socket_path = get_worker_zmq_path(self.rank)

        # Remove existing socket file if it exists
        ipc_file = socket_path.replace("ipc://", "")
        if os.path.exists(ipc_file):
            try:
                os.remove(ipc_file)
            except OSError as e:
                print(f"Error removing existing ZMQ socket {ipc_file}: {e}")

        self.socket.bind(socket_path)
        self.running = True
        print(f"Worker {self.rank} ZMQ listener started at {socket_path}")

        # Start listening loop
        asyncio.create_task(self._listen_loop())

    async def _listen_loop(self):
        """Main loop that processes incoming messages."""
        serializer = ZMQMessageSerializer()

        while self.running:
            try:
                # ROUTER socket receives: [identity, empty_frame, message_frames...]
                frames = await self.socket.recv_multipart()

                if len(frames) < 2:
                    print(f"Worker {self.rank}: Invalid message format")
                    continue

                identity = frames[0]
                # frames[1] is the empty delimiter frame
                msg_frames = frames[2:]

                # Deserialize message
                msg = serializer.deserialize(msg_frames)

                # Process command
                response = await self._process_command(msg)

                # Send response back to client
                response_frames = serializer.serialize(response)
                await self.socket.send_multipart([identity, b""] + response_frames)

            except zmq.ZMQError as e:
                if self.running:
                    print(f"Worker {self.rank} ZMQ error: {e}")
            except Exception as e:
                print(f"Worker {self.rank} error processing message: {e}")
                # Try to send error response
                try:
                    error_response = serializer.serialize({
                        "status": "error",
                        "message": str(e)
                    })
                    await self.socket.send_multipart([identity, b""] + error_response)
                except:
                    pass

    async def _process_command(self, msg: Message) -> Message:
        """Process a command and return response."""
        cmd = msg.get("cmd")

        if cmd == "map_to_kv_tensors":
            map_to_kv_tensors(msg["offsets"])
            return {"status": "success"}
        elif cmd == "unmap_from_kv_tensors":
            unmap_from_kv_tensors(msg["offsets"])
            return {"status": "success"}
        elif cmd == "kv_tensors_created":
            created = kv_tensors_created()
            return {"status": "success", "created": created}
        else:
            return {"status": "error", "message": f"Unknown command: {cmd}"}

    async def stop(self):
        """Stop the listener and cleanup resources."""
        self.running = False
        self.socket.close()
        self.context.term()


def start_worker_listener_thread(rank: int):
    """
    Start a ZeroMQ worker listener in a background thread.
    This is the entry point called by workers during initialization.
    """
    listener = ZMQWorkerListener(rank)

    # Create new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Start listener
    loop.run_until_complete(listener.start())

    # Keep loop running
    try:
        loop.run_forever()
    finally:
        loop.run_until_complete(listener.stop())
        loop.close()


# Global ZMQ context - reuse across all calls for better performance
_zmq_context = None
_zmq_sockets = {}  # Cache sockets per rank


def _get_zmq_context():
    """Get or create the global ZMQ context."""
    global _zmq_context
    if _zmq_context is None:
        _zmq_context = zmq.asyncio.Context()
    return _zmq_context


async def _send_and_receive_message(rank: int, message: Message) -> Message:
    """
    Send a message to a worker and receive response using ZeroMQ.
    Uses DEALER socket for client-side communication.

    NOTE: This creates a new socket per call but reuses the global context.
    For even better performance, consider socket pooling in future.
    """
    context = _get_zmq_context()
    socket = context.socket(zmq.DEALER)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVHWM, 0)
    socket.setsockopt(zmq.SNDHWM, 0)

    try:
        socket_path = get_worker_zmq_path(rank)
        socket.connect(socket_path)

        # Serialize and send message
        serializer = ZMQMessageSerializer()
        msg_frames = serializer.serialize(message)

        # DEALER sends: [empty_frame, message_frames...]
        await socket.send_multipart([b""] + msg_frames)

        # Receive response: [empty_frame, response_frames...]
        frames = await socket.recv_multipart()

        if len(frames) < 1:
            raise RuntimeError("Invalid response format")

        # Skip empty delimiter frame
        response_frames = frames[1:] if len(frames) > 1 else frames

        # Deserialize response
        response = serializer.deserialize(response_frames)
        return response

    finally:
        socket.close()
        # Don't terminate context - it's global and reused


async def _broadcast_map_to_kv_tensors(tp_size: int, offsets: List[int]) -> None:
    """
    Broadcast the "map_to_kv_tensors" operation to all workers concurrently using ZeroMQ.
    """
    map_message = {"cmd": "map_to_kv_tensors", "offsets": offsets}
    tasks = [
        _send_and_receive_message(rank, map_message) for rank in range(tp_size)
    ]

    responses = await asyncio.gather(*tasks, return_exceptions=True)
    for rank, response in enumerate(responses):
        if isinstance(response, Exception):
            raise RuntimeError(f"Worker {rank} failed to map: {response}")
        elif not isinstance(response, dict) or response.get("status") != "success":
            raise RuntimeError(f"Worker {rank} failed to map: {response}")


async def _broadcast_unmap_from_kv_tensors(tp_size: int, offsets: List[int]) -> None:
    """
    Broadcast the "unmap_from_kv_tensors" operation to all workers concurrently using ZeroMQ.
    """
    unmap_message = {"cmd": "unmap_from_kv_tensors", "offsets": offsets}
    tasks = [
        _send_and_receive_message(rank, unmap_message) for rank in range(tp_size)
    ]

    responses = await asyncio.gather(*tasks, return_exceptions=True)
    for rank, response in enumerate(responses):
        if isinstance(response, Exception):
            raise RuntimeError(f"Worker {rank} failed to unmap: {response}")
        elif not isinstance(response, dict) or response.get("status") != "success":
            raise RuntimeError(f"Worker {rank} failed to unmap: {response}")


async def _broadcast_kv_tensors_created(tp_size: int) -> bool:
    """
    Broadcast the "kv_tensors_created" operation to all workers concurrently using ZeroMQ.
    Returns True if all workers report that KV tensors are created, False otherwise.
    """
    check_message = {"cmd": "kv_tensors_created"}
    tasks = [
        _send_and_receive_message(rank, check_message) for rank in range(tp_size)
    ]

    responses = await asyncio.gather(*tasks, return_exceptions=True)
    all_created = True
    for rank, response in enumerate(responses):
        if isinstance(response, Exception):
            raise RuntimeError(
                f"Worker {rank} failed to check KV tensors created: {response}"
            )
        elif not isinstance(response, dict) or response.get("status") != "success":
            raise RuntimeError(
                f"Worker {rank} failed to check KV tensors created: {response}"
            )
        elif not response.get("created", False):
            all_created = False

    return all_created


# Wrapper functions to call the async functions from sync code
def broadcast_map_to_kv_tensors(tp_size: int, offsets: List[int]) -> None:
    """Broadcast map operation to all workers (sync wrapper)."""
    asyncio.run(_broadcast_map_to_kv_tensors(tp_size, offsets))


def broadcast_unmap_from_kv_tensors(tp_size: int, offsets: List[int]) -> None:
    """Broadcast unmap operation to all workers (sync wrapper)."""
    asyncio.run(_broadcast_unmap_from_kv_tensors(tp_size, offsets))


def broadcast_kv_tensors_created(tp_size: int) -> bool:
    """Check if KV tensors are created on all workers (sync wrapper)."""
    return asyncio.run(_broadcast_kv_tensors_created(tp_size))
