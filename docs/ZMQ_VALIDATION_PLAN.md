# ZeroMQ Implementation - Validation & Testing Plan

## 📍 Where ZeroMQ Has Been Added

### ✅ Single Integration Point: Tensor-Parallel Worker IPC

ZeroMQ has been added to **ONE specific location** in kvcached - the tensor-parallel worker communication layer.

#### Files Modified:
1. **`kvcached/tp_ipc_zmq.py`** (NEW) - Complete ZeroMQ transport implementation
2. **`kvcached/tp_ipc_util.py`** (MODIFIED) - Backend selection logic
3. **`pyproject.toml`** (MODIFIED) - Added `pyzmq>=25.0.0` dependency

#### What It Does:
When `tp_size > 1` (multi-GPU tensor parallelism), workers need to communicate for:
- **`broadcast_map_to_kv_tensors()`** - Map KV cache pages to physical GPU memory
- **`broadcast_unmap_from_kv_tensors()`** - Unmap KV cache pages
- **`broadcast_kv_tensors_created()`** - Check if KV tensors initialized

#### How It's Used:
```python
# In KVCacheManager when tp_size > 1
from kvcached.tp_ipc_util import broadcast_map_to_kv_tensors

# This now automatically uses ZMQ or UNIX sockets based on:
# KVCACHED_IPC_BACKEND environment variable
broadcast_map_to_kv_tensors(tp_size=4, offsets=[0, 2097152, 4194304, ...])
```

### ❌ Where ZeroMQ Is NOT Used

- **Controller/Frontend** (`controller/*.py`) - Uses HTTP REST APIs via aiohttp
- **Single GPU mode** (`tp_size=1`) - No IPC needed
- **Client-server communication** - Standard HTTP/OpenAI API

## 🎯 Benchmark Validation Plan

### Tier 1: Direct IPC Performance (PRIMARY)

#### 1. `benchmarks/bench_tp_ipc/` ⭐ **MOST IMPORTANT**

**Purpose**: Directly measures IPC overhead for map/unmap operations

**What to run**:
```bash
# Baseline: UNIX sockets with async implementation
python benchmarks/bench_tp_ipc/kvcached_tp_ipc_benchmark.py \
    --tp-size 4 \
    --iters 50 \
    --pages-per-iter 20 \
    --map-impl async \
    --verbose

# Test: ZeroMQ implementation
python benchmarks/bench_tp_ipc/kvcached_tp_ipc_benchmark.py \
    --tp-size 4 \
    --iters 50 \
    --pages-per-iter 20 \
    --map-impl zmq \
    --verbose

# Compare different implementations
for impl in seq thread async zmq; do
    echo "Testing: $impl"
    python benchmarks/bench_tp_ipc/kvcached_tp_ipc_benchmark.py \
        --tp-size 4 \
        --iters 50 \
        --pages-per-iter 20 \
        --map-impl $impl
done
```

**Expected results**:
- **Latency**: ZMQ should show 20-40% lower mean/p95 latency vs async
- **Scalability**: ZMQ benefit increases with `--tp-size` (test 2, 4, 8)
- **Large messages**: ZMQ shines with `--pages-per-iter 50+`

**Key metrics to compare**:
- Mean map/unmap time (ms)
- P95 latency (ms)
- Per-page overhead (ms)

---

#### 2. `benchmarks/bench_map_parallelism/` ⭐ **IMPORTANT**

**Purpose**: Tests parallel map operations across multiple workers with IPC

**What to run**:
```bash
# Baseline: UNIX sockets (default)
export KVCACHED_IPC_BACKEND=unix
python benchmarks/bench_map_parallelism/kvcached_map_parallel_benchmark.py \
    --procs 4 \
    --pages-per-proc 10 \
    --iters 20 \
    --verbose

# Test: ZeroMQ backend
export KVCACHED_IPC_BACKEND=zmq
python benchmarks/bench_map_parallelism/kvcached_map_parallel_benchmark.py \
    --procs 4 \
    --pages-per-proc 10 \
    --iters 20 \
    --verbose
```

**Expected results**:
- Faster map/unmap operations with ZMQ
- Lower variance in per-iteration times
- Better scaling with `--procs` increase

**Key metrics**:
- Per-iteration map time
- Total benchmark runtime
- Min/max/mean across iterations

---

### Tier 2: End-to-End Serving Performance (SECONDARY)

#### 3. `benchmarks/bench_latency_benefit/` 🎯 **END-TO-END**

**Purpose**: Measures real-world serving performance with kvcached

**What to run**:
```bash
# Baseline: UNIX sockets
export KVCACHED_IPC_BACKEND=unix
export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1

# Start vLLM server with TP=2 or TP=4
vllm serve meta-llama/Llama-3.1-8B \
    --tensor-parallel-size 4 \
    --disable-log-requests \
    --no-enable-prefix-caching \
    --port 12346

# Run benchmark
python benchmarks/bench_latency_benefit/bench_kvcached_vllm.py \
    --model meta-llama/Llama-3.1-8B \
    --request-rate 10 \
    --num-prompts 1000

# Test: ZeroMQ backend
export KVCACHED_IPC_BACKEND=zmq
# Restart server and re-run benchmark
```

**Expected results**:
- Lower TTFT (Time To First Token) with ZMQ
- Higher throughput under load
- Better P99 latency

**Key metrics**:
- TTFT (mean, p50, p99)
- Throughput (tokens/sec)
- Request latency distribution

---

#### 4. `benchmarks/gsm8k/` 📚 **QUALITY CHECK**

**Purpose**: Ensures correctness of outputs with ZMQ backend

**What to run**:
```bash
# Test with both backends to ensure same results
export KVCACHED_IPC_BACKEND=unix
python benchmarks/gsm8k/bench_vllm.py --model meta-llama/Llama-3.1-8B

export KVCACHED_IPC_BACKEND=zmq
python benchmarks/gsm8k/bench_vllm.py --model meta-llama/Llama-3.1-8B
```

**Expected results**:
- Identical accuracy/outputs between backends
- No correctness regressions

---

### Tier 3: VMM and Overhead Tests (OPTIONAL)

#### 5. `benchmarks/bench_vmm/` - Low-level memory operations
#### 6. `benchmarks/bench_kvcached_overhead/` - Overhead measurements

These test low-level operations that don't directly use IPC, but good for completeness.

---

## 🔬 Detailed Test Matrix

### Configuration Variables to Test

| Variable | Values | Why |
|----------|--------|-----|
| `KVCACHED_IPC_BACKEND` | `unix`, `zmq` | Core comparison |
| `--tp-size` / `--procs` | 2, 4, 8 | Scalability |
| `--pages-per-iter` | 1, 10, 20, 50 | Message size impact |
| `--iters` | 20, 50, 100 | Statistical significance |
| Model size | 7B, 8B, 13B, 70B | Different KV cache sizes |

### Recommended Test Sequence

#### Phase 1: Quick Validation (5-10 minutes)
```bash
# 1. Sanity check - does it work?
export KVCACHED_IPC_BACKEND=zmq
python benchmarks/bench_tp_ipc/kvcached_tp_ipc_benchmark.py \
    --tp-size 2 --iters 10 --pages-per-iter 5 --map-impl zmq

# 2. Compare with baseline
export KVCACHED_IPC_BACKEND=unix
python benchmarks/bench_tp_ipc/kvcached_tp_ipc_benchmark.py \
    --tp-size 2 --iters 10 --pages-per-iter 5 --map-impl async
```

#### Phase 2: Performance Validation (30-60 minutes)
```bash
# Test all implementations with meaningful iteration counts
for impl in seq thread async zmq; do
    for tp_size in 2 4; do
        for pages in 10 20; do
            echo "=== Testing impl=$impl tp=$tp_size pages=$pages ==="
            python benchmarks/bench_tp_ipc/kvcached_tp_ipc_benchmark.py \
                --tp-size $tp_size \
                --iters 50 \
                --pages-per-iter $pages \
                --map-impl $impl \
                | tee results_${impl}_tp${tp_size}_p${pages}.txt
        done
    done
done
```

#### Phase 3: End-to-End Validation (2-4 hours)
```bash
# Run full serving benchmarks with both backends
# Compare TTFT, throughput, and latency
```

---

## 📊 Expected Performance Improvements

Based on vLLM's ZeroMQ results and our implementation:

### Direct IPC (bench_tp_ipc)
- **Latency**: 20-40% reduction in mean latency
- **P95 latency**: 30-50% reduction
- **Throughput**: 2-3x for concurrent operations
- **Scaling**: Better performance as tp_size increases

### Real-world Serving (bench_latency_benefit)
- **TTFT**: 5-15% improvement (depends on workload)
- **Throughput**: 10-20% improvement under high load
- **P99 latency**: 15-25% improvement
- **CPU usage**: 5-10% reduction

---

## 🐛 Debugging Failed Tests

### ZeroMQ socket errors
```bash
# Clean up stale sockets
rm -rf /tmp/kvcached-zmq-ipc/*
rm -rf /tmp/kvcached-ipc/*
```

### Import errors
```bash
# Reinstall with ZeroMQ
pip install -e . --no-build-isolation --no-cache-dir
pip list | grep zmq  # Should show pyzmq>=25.0.0
```

### No performance improvement
- Check: Are you using `tp_size > 1`? ZMQ only used for multi-worker
- Check: Is `--pages-per-iter` large enough? (try 20+)
- Check: Monitor CPU/GPU with `htop` - are workers CPU-bound?
- Check: Disable prefix caching if enabled

### Verification
```bash
# Verify backend is actually being used
python -c "
import os
os.environ['KVCACHED_IPC_BACKEND'] = 'zmq'
from kvcached.tp_ipc_util import IPC_BACKEND
print(f'Using backend: {IPC_BACKEND}')
"
```

---

## 📈 Results Analysis Template

Create a results comparison table:

| Metric | UNIX (async) | ZeroMQ | Improvement |
|--------|--------------|--------|-------------|
| Mean latency (ms) | X.XX | Y.YY | ZZ% |
| P95 latency (ms) | X.XX | Y.YY | ZZ% |
| Per-page (ms) | X.XX | Y.YY | ZZ% |
| Throughput (ops/s) | XXX | YYY | ZZ% |

---

## ✅ Success Criteria

The ZeroMQ implementation is validated if:

1. ✅ **Correctness**: All benchmarks run without errors on both backends
2. ✅ **Performance**: ZMQ shows measurable improvement in IPC benchmarks
3. ✅ **Scalability**: ZMQ benefit increases with tp_size
4. ✅ **Quality**: No accuracy regressions in GSM8K tests
5. ✅ **Real-world**: TTFT improvements in serving benchmarks

---

## 📝 Next Steps After Validation

1. **Document Results**: Update README with benchmark results
2. **Tune Parameters**: Optimize `KVCACHED_ZMQ_MAX_CHUNK_SIZE_MB` based on testing
3. **CI Integration**: Add ZMQ backend to CI test matrix
4. **User Guide**: Add usage examples to documentation
5. **Future Work**: Consider shared memory ring buffer implementation

---

## 🔗 Reference Commands

### Quick Start
```bash
# Install
pip install -e . --no-build-isolation

# Run primary benchmark
export KVCACHED_IPC_BACKEND=zmq
python benchmarks/bench_tp_ipc/kvcached_tp_ipc_benchmark.py \
    --tp-size 4 --iters 50 --pages-per-iter 20 --map-impl zmq
```

### Environment Variables
```bash
export KVCACHED_IPC_BACKEND=zmq              # Enable ZeroMQ
export KVCACHED_ZMQ_MAX_CHUNK_SIZE_MB=32     # Chunk size (default: 32)
export ENABLE_KVCACHED=true                   # Enable kvcached
export KVCACHED_AUTOPATCH=1                   # Auto-patch engines
```
