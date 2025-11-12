#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""
Non-GPU benchmark for comparing UNIX sockets vs ZeroMQ IPC performance.

This tests the raw message passing overhead without requiring CUDA or GPUs.
"""

import argparse
import asyncio
import multiprocessing as mp
import os
import pickle
import socket
import struct
import time
from typing import List

import zmq
import zmq.asyncio


# ============================================================================
# Mock Worker Listener (UNIX Sockets)
# ============================================================================

def unix_worker_listener(rank: int, socket_dir: str, ready_queue: mp.Queue):
    """Worker that listens on UNIX socket and echoes messages back."""
    socket_path = os.path.join(socket_dir, f"worker_{rank}.sock")

    if os.path.exists(socket_path):
        os.remove(socket_path)

    server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_sock.bind(socket_path)
    server_sock.listen()

    ready_queue.put(rank)

    while True:
        conn, _ = server_sock.accept()
        try:
            # Read length
            length_bytes = conn.recv(4)
            if not length_bytes:
                break
            length = int.from_bytes(length_bytes, 'big')

            # Read data
            data = b""
            while len(data) < length:
                chunk = conn.recv(length - len(data))
                if not chunk:
                    break
                data += chunk

            msg = pickle.loads(data)

            # Echo back
            response = {"status": "success", "rank": rank}
            response_data = pickle.dumps(response)
            conn.sendall(len(response_data).to_bytes(4, 'big') + response_data)
        except Exception as e:
            print(f"Worker {rank} error: {e}")
        finally:
            conn.close()


# ============================================================================
# Mock Worker Listener (ZeroMQ)
# ============================================================================

def zmq_worker_listener(rank: int, socket_dir: str, ready_queue: mp.Queue):
    """Worker that listens on ZeroMQ socket and echoes messages back."""
    socket_path = f"ipc://{socket_dir}/worker_{rank}.ipc"

    # Remove existing socket file
    ipc_file = socket_path.replace("ipc://", "")
    if os.path.exists(ipc_file):
        os.remove(ipc_file)

    context = zmq.Context()
    socket_obj = context.socket(zmq.ROUTER)
    socket_obj.setsockopt(zmq.LINGER, 0)
    socket_obj.bind(socket_path)

    ready_queue.put(rank)

    while True:
        try:
            frames = socket_obj.recv_multipart()
            if len(frames) < 2:
                continue

            identity = frames[0]
            msg_frames = frames[2:]

            # Deserialize
            if len(msg_frames) == 1:
                msg = pickle.loads(msg_frames[0])
            else:
                # Multi-frame message
                msg = pickle.loads(msg_frames[0])
                if msg.get("_has_offsets_frame"):
                    offsets_bytes = msg_frames[1]
                    count = struct.unpack("<I", offsets_bytes[:4])[0]
                    offsets = list(struct.unpack(f"<{count}q", offsets_bytes[4:]))
                    msg["offsets"] = offsets
                    del msg["_has_offsets_frame"]

            # Echo back
            response = {"status": "success", "rank": rank}
            response_data = pickle.dumps(response)
            socket_obj.send_multipart([identity, b"", response_data])
        except Exception as e:
            print(f"ZMQ Worker {rank} error: {e}")


# ============================================================================
# Benchmark Functions
# ============================================================================

def send_unix_message(rank: int, socket_dir: str, msg: dict) -> dict:
    """Send message via UNIX socket."""
    socket_path = os.path.join(socket_dir, f"worker_{rank}.sock")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(socket_path)

    try:
        # Send message
        data = pickle.dumps(msg)
        sock.sendall(len(data).to_bytes(4, 'big') + data)

        # Receive response
        length_bytes = sock.recv(4)
        length = int.from_bytes(length_bytes, 'big')
        data = b""
        while len(data) < length:
            chunk = sock.recv(length - len(data))
            if not chunk:
                break
            data += chunk

        return pickle.loads(data)
    finally:
        sock.close()


# Global ZMQ context for benchmark - reuse for performance
_bench_zmq_context = None


def _get_bench_zmq_context():
    """Get or create the global ZMQ context for benchmark."""
    global _bench_zmq_context
    if _bench_zmq_context is None:
        _bench_zmq_context = zmq.asyncio.Context()
    return _bench_zmq_context


async def send_zmq_message_async(rank: int, socket_dir: str, msg: dict) -> dict:
    """Send message via ZeroMQ async."""
    socket_path = f"ipc://{socket_dir}/worker_{rank}.ipc"

    context = _get_bench_zmq_context()
    socket_obj = context.socket(zmq.DEALER)
    socket_obj.setsockopt(zmq.LINGER, 0)
    socket_obj.connect(socket_path)

    try:
        # Serialize message
        offsets = msg.get("offsets")
        if offsets and isinstance(offsets, list) and len(offsets) > 100:
            # Multi-frame for large messages
            metadata = msg.copy()
            metadata["offsets"] = None
            metadata["_has_offsets_frame"] = True
            metadata_bytes = pickle.dumps(metadata)
            offsets_bytes = struct.pack(f"<I{len(offsets)}q", len(offsets), *offsets)
            await socket_obj.send_multipart([b"", metadata_bytes, offsets_bytes])
        else:
            # Single frame
            msg_bytes = pickle.dumps(msg)
            await socket_obj.send_multipart([b"", msg_bytes])

        # Receive response
        frames = await socket_obj.recv_multipart()
        response_frames = frames[1:] if len(frames) > 1 else frames
        response = pickle.loads(response_frames[0])

        return response
    finally:
        socket_obj.close()
        # Don't terminate context - it's global and reused


def send_zmq_message(rank: int, socket_dir: str, msg: dict) -> dict:
    """Sync wrapper for ZMQ message."""
    return asyncio.run(send_zmq_message_async(rank, socket_dir, msg))


async def broadcast_zmq_async(workers: int, socket_dir: str, msg: dict) -> List[dict]:
    """Broadcast to all workers concurrently via ZeroMQ."""
    tasks = [send_zmq_message_async(rank, socket_dir, msg) for rank in range(workers)]
    responses = await asyncio.gather(*tasks)
    return responses


def broadcast_zmq(workers: int, socket_dir: str, msg: dict) -> List[dict]:
    """Sync wrapper for ZMQ broadcast."""
    return asyncio.run(broadcast_zmq_async(workers, socket_dir, msg))


def broadcast_unix_sequential(workers: int, socket_dir: str, msg: dict) -> List[dict]:
    """Broadcast to all workers sequentially via UNIX sockets."""
    responses = []
    for rank in range(workers):
        response = send_unix_message(rank, socket_dir, msg)
        responses.append(response)
    return responses


# ============================================================================
# Main Benchmark
# ============================================================================

def run_benchmark(
    backend: str,
    workers: int,
    iterations: int,
    message_size: str,
    verbose: bool,
):
    """Run IPC benchmark."""
    socket_dir = f"/tmp/kvcached-bench-{backend}"
    os.makedirs(socket_dir, exist_ok=True)

    # Create test message
    if message_size == "small":
        msg = {"cmd": "test", "data": list(range(10))}
    elif message_size == "medium":
        msg = {"cmd": "test", "data": list(range(1000))}
    elif message_size == "large":
        # Large offset array (like real KV cache operations)
        msg = {"cmd": "map_to_kv_tensors", "offsets": list(range(0, 100 * 2097152, 2097152))}
    else:
        msg = {"cmd": "test"}

    # Start workers
    ready_queue = mp.Queue()
    procs = []

    listener_func = zmq_worker_listener if backend == "zmq" else unix_worker_listener

    for rank in range(workers):
        p = mp.Process(
            target=listener_func,
            args=(rank, socket_dir, ready_queue),
            daemon=True
        )
        p.start()
        procs.append(p)

    # Wait for all workers to be ready
    for _ in range(workers):
        ready_queue.get()

    time.sleep(0.5)  # Let sockets stabilize

    if verbose:
        print(f"\nStarting benchmark: backend={backend}, workers={workers}, "
              f"iterations={iterations}, message_size={message_size}")

    # Run benchmark
    times = []
    broadcast_func = broadcast_zmq if backend == "zmq" else broadcast_unix_sequential

    for i in range(iterations):
        t0 = time.time()
        responses = broadcast_func(workers, socket_dir, msg)
        t1 = time.time()

        elapsed = t1 - t0
        times.append(elapsed)

        if verbose and i % 10 == 0:
            print(f"  Iteration {i:3d}: {elapsed*1000:.3f} ms")

        # Verify responses
        if len(responses) != workers:
            print(f"ERROR: Expected {workers} responses, got {len(responses)}")

    # Compute statistics
    times = [t * 1000 for t in times]  # Convert to ms
    mean_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    p95_time = sorted(times)[int(len(times) * 0.95)]

    # Print results
    print(f"\n{'='*60}")
    print(f"Benchmark Results: {backend.upper()}")
    print(f"{'='*60}")
    print(f"Backend:          {backend}")
    print(f"Workers:          {workers}")
    print(f"Iterations:       {iterations}")
    print(f"Message Size:     {message_size}")
    print(f"-" * 60)
    print(f"Mean Latency:     {mean_time:.3f} ms")
    print(f"Min Latency:      {min_time:.3f} ms")
    print(f"Max Latency:      {max_time:.3f} ms")
    print(f"P95 Latency:      {p95_time:.3f} ms")
    print(f"Per-worker Mean:  {mean_time/workers:.3f} ms")
    print(f"{'='*60}\n")

    # Cleanup
    for p in procs:
        p.terminate()
        p.join(timeout=1)

    return {
        "backend": backend,
        "workers": workers,
        "mean": mean_time,
        "min": min_time,
        "max": max_time,
        "p95": p95_time,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Non-GPU IPC benchmark comparing UNIX sockets vs ZeroMQ"
    )
    parser.add_argument(
        "--backend",
        choices=["unix", "zmq", "both"],
        default="both",
        help="Which backend to test (default: both)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes (default: 4)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of iterations (default: 50)"
    )
    parser.add_argument(
        "--message-size",
        choices=["small", "medium", "large"],
        default="large",
        help="Message size to test (default: large)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print per-iteration details"
    )

    args = parser.parse_args()

    results = []

    if args.backend == "both":
        backends = ["unix", "zmq"]
    else:
        backends = [args.backend]

    for backend in backends:
        result = run_benchmark(
            backend=backend,
            workers=args.workers,
            iterations=args.iterations,
            message_size=args.message_size,
            verbose=args.verbose,
        )
        results.append(result)

    # Print comparison if both backends tested
    if len(results) == 2:
        unix_result = results[0] if results[0]["backend"] == "unix" else results[1]
        zmq_result = results[0] if results[0]["backend"] == "zmq" else results[1]

        print(f"\n{'='*60}")
        print(f"COMPARISON: UNIX vs ZeroMQ")
        print(f"{'='*60}")
        print(f"{'Metric':<20} {'UNIX':<12} {'ZeroMQ':<12} {'Improvement':<12}")
        print(f"{'-'*60}")

        for metric in ["mean", "min", "max", "p95"]:
            unix_val = unix_result[metric]
            zmq_val = zmq_result[metric]
            improvement = ((unix_val - zmq_val) / unix_val) * 100
            print(f"{metric.capitalize():<20} {unix_val:>10.3f}ms {zmq_val:>10.3f}ms "
                  f"{improvement:>10.1f}%")

        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
