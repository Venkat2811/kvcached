# ZeroMQ IPC Implementation for kvcached

## Overview

This document describes the ZeroMQ-based IPC (Inter-Process Communication) implementation added to kvcached, inspired by vLLM's approach to high-performance worker communication.

## Motivation

The original kvcached implementation uses UNIX domain sockets for IPC between tensor-parallel workers. While functional, this approach has limitations when handling:
- Large message sizes (8-32MB KV cache blocks)
- Multi-worker broadcast scenarios
- High-throughput concurrent communication

ZeroMQ provides significant performance improvements by offering:
- **Zero-copy messaging** for large payloads via multipart messages
- **Asynchronous I/O** with concurrent message processing
- **Built-in patterns** (ROUTER/DEALER) optimized for distributed communication
- **Better scalability** for tensor-parallel workloads

## Architecture

### Key Components

1. **`kvcached/tp_ipc_zmq.py`** - ZeroMQ transport layer
   - `ZMQMessageSerializer` - Handles zero-copy serialization
   - `ZMQWorkerListener` - ROUTER socket for receiving commands
   - Broadcast functions - DEALER sockets for sending to workers

2. **`kvcached/tp_ipc_util.py`** - Unified IPC interface
   - Backend selection via `KVCACHED_IPC_BACKEND` env variable
   - Transparent switching between UNIX sockets and ZeroMQ

3. **Benchmark Integration** - `benchmarks/bench_tp_ipc/`
   - New `--map-impl zmq` option for performance testing
   - Direct comparison with existing implementations

### Message Flow

```
Controller/Master Process
         |
         v
    [DEALER Socket] ----> [ROUTER Socket] Worker 0
         |                [ROUTER Socket] Worker 1
         |                [ROUTER Socket] Worker N
         v
   Concurrent Broadcast
   (map/unmap operations)
```

### Zero-Copy Serialization

For large offset arrays (>100 elements), the implementation:
1. Splits message into metadata + data frames
2. Sends offset array as separate ZMQ frame
3. Enables zero-copy transmission at ZMQ layer

```python
# Before: Single pickle frame (copy overhead)
msg = {"cmd": "map_to_kv_tensors", "offsets": [large_array]}
send(pickle.dumps(msg))

# After: Multi-frame with zero-copy potential
metadata = {"cmd": "map_to_kv_tensors", "_has_offsets_frame": True}
offsets_bytes = struct.pack(f"<I{len(offsets)}q", len(offsets), *offsets)
send_multipart([pickle.dumps(metadata), offsets_bytes])
```

## Usage

### Enable ZeroMQ Backend

Set the environment variable before starting workers:

```bash
export KVCACHED_IPC_BACKEND=zmq
```

Default is `unix` for backward compatibility.

### Configuration Options

- **`KVCACHED_IPC_BACKEND`**: `"unix"` or `"zmq"` (default: `"unix"`)
- **`KVCACHED_ZMQ_MAX_CHUNK_SIZE_MB`**: Max message chunk size in MB (default: 32)

### Running Benchmarks

Compare performance of different IPC implementations:

```bash
# Benchmark UNIX sockets (original)
python benchmarks/bench_tp_ipc/kvcached_tp_ipc_benchmark.py \
    --tp-size 4 --iters 20 --pages-per-iter 10 --map-impl async

# Benchmark ZeroMQ
python benchmarks/bench_tp_ipc/kvcached_tp_ipc_benchmark.py \
    --tp-size 4 --iters 20 --pages-per-iter 10 --map-impl zmq
```

### Integration with Existing Code

No code changes required! The implementation is transparent:

```python
from kvcached.tp_ipc_util import (
    broadcast_map_to_kv_tensors,
    broadcast_unmap_from_kv_tensors,
    start_worker_listener_thread
)

# Works with both UNIX sockets and ZeroMQ
# Backend selected via environment variable
start_worker_listener_thread(rank=0)
broadcast_map_to_kv_tensors(tp_size=4, offsets=[...])
```

## Performance Expectations

Based on vLLM's experience with ZeroMQ, expected improvements:

1. **Latency Reduction**: 20-40% lower latency for large messages
2. **Throughput**: 2-3x higher throughput under concurrent load
3. **Scalability**: Better performance as `tp_size` increases
4. **CPU Efficiency**: Lower CPU usage due to zero-copy paths

Run the benchmarks to validate these improvements on your hardware!

## Future Enhancements

### Shared Memory Ring Buffer (Planned)

For even better performance on same-node communication:

- **Zero-copy local communication**: No serialization for workers on same node
- **Lock-free design**: Single-writer, multiple-reader ring buffer
- **Hybrid approach**: SHM for local + ZMQ for remote workers
- **Automatic fallback**: Switch to ZMQ for messages exceeding buffer size

Reference implementation: `vllm/distributed/device_communicators/shm_broadcast.py`

This would bring performance closer to vLLM's optimized IPC layer while maintaining kvcached's flexibility.

## Comparison with vLLM

### Similarities
- ZeroMQ IPC transport for worker communication
- ROUTER/DEALER socket pattern
- Multipart messaging for large payloads
- Configurable chunk sizes

### Differences
- **kvcached**: Simple direct implementation, easier to maintain
- **vLLM**: Includes shared memory ring buffer for local optimization
- **kvcached**: Environment variable configuration
- **vLLM**: More complex hybrid SHM+ZMQ architecture

## Troubleshooting

### ZeroMQ socket already in use

```bash
# Clean up stale socket files
rm -rf /tmp/kvcached-zmq-ipc/*
```

### Import errors

```bash
# Ensure pyzmq is installed
pip install pyzmq>=25.0.0
```

### Performance not improving

- Check CPU/GPU affinity settings
- Verify `--pages-per-iter` is large enough to show benefits
- Monitor with `htop` to ensure workers aren't CPU-bound
- Try different `tp_size` values

## Testing

Run the test suite to verify ZeroMQ functionality:

```bash
# Test both backends
KVCACHED_IPC_BACKEND=unix python -m pytest tests/
KVCACHED_IPC_BACKEND=zmq python -m pytest tests/
```

## References

1. **vLLM ZeroMQ Implementation**: https://github.com/vllm-project/vllm
   - `vllm/v1/engine/core_client.py` - Client-side ZMQ usage
   - `vllm/v1/serial_utils.py` - Zero-copy tensor serialization
   - `vllm/distributed/device_communicators/shm_broadcast.py` - Shared memory ring buffer

2. **ZeroMQ Documentation**: https://zeromq.org/
   - Guide: https://zguide.zeromq.org/
   - Python bindings: https://pyzmq.readthedocs.io/

3. **kvcached Blog Post**: https://yifanqiao.notion.site/Solve-the-GPU-Cost-Crisis-with-kvcached-289da9d1f4d68034b17bf2774201b141

## Contributing

Contributions to improve the ZeroMQ implementation are welcome! Key areas:

1. **Shared memory ring buffer** - Implement local zero-copy optimization
2. **Performance tuning** - Optimize socket options and buffer sizes
3. **Error handling** - Improve robustness and error recovery
4. **Testing** - Add comprehensive test coverage
5. **Documentation** - Expand usage examples and best practices

Please open an issue or PR on the kvcached repository.
