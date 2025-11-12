#!/usr/bin/env python3
"""
Direct comparison benchmark: Current vs Optimized ZMQ implementations.
Runs both side-by-side with identical test conditions.
"""

import multiprocessing as mp
import os
import sys
import time
import importlib.util
import types
from typing import List

# Mock vmm_ops for non-GPU testing
vmm_ops = types.ModuleType('vmm_ops')
vmm_ops.map_to_kv_tensors = lambda x: None
vmm_ops.unmap_from_kv_tensors = lambda x: None
vmm_ops.kv_tensors_created = lambda: True
sys.modules['kvcached.vmm_ops'] = vmm_ops

def load_implementation(name, filepath):
    """Load ZMQ implementation from file."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def run_benchmark(impl_name, impl_module, workers=4, iterations=30):
    """Run benchmark with specific implementation."""
    print(f"\n{'='*60}")
    print(f"Testing: {impl_name.upper()} Implementation")
    print(f"{'='*60}")

    socket_dir = f"/tmp/kvcached-bench-{impl_name}"
    os.makedirs(socket_dir, exist_ok=True)

    # Start worker listeners
    ready_queue = mp.Queue()

    def worker_process(rank):
        impl_module.start_worker_listener_thread(rank)
        ready_queue.put(rank)
        while True:
            time.sleep(3600)

    procs = []
    for rank in range(workers):
        p = mp.Process(target=worker_process, args=(rank,), daemon=True)
        p.start()
        procs.append(p)

    # Wait for all workers to be ready
    for _ in range(workers):
        ready_queue.get()

    time.sleep(0.5)  # Let sockets stabilize

    # Run benchmark
    offsets = list(range(0, 100 * 2097152, 2097152))  # Large message (100 offsets)
    times = []

    print(f"Running {iterations} iterations...")
    for i in range(iterations):
        t0 = time.time()
        impl_module.broadcast_map_to_kv_tensors(workers, offsets)
        t1 = time.time()
        times.append((t1 - t0) * 1000)  # Convert to ms

        if i % 10 == 0:
            print(f"  Progress: {i}/{iterations}")

    # Calculate statistics
    import numpy as np
    mean = np.mean(times)
    min_t = np.min(times)
    max_t = np.max(times)
    p95 = np.percentile(times, 95)
    per_worker = mean / workers

    # Print results
    print(f"\nResults:")
    print(f"  Mean Latency:     {mean:.3f} ms")
    print(f"  Min Latency:      {min_t:.3f} ms")
    print(f"  Max Latency:      {max_t:.3f} ms")
    print(f"  P95 Latency:      {p95:.3f} ms")
    print(f"  Per-worker Mean:  {per_worker:.3f} ms")

    # Cleanup
    for p in procs:
        p.terminate()
        p.join(timeout=1)

    # Clean socket directory
    import shutil
    shutil.rmtree(socket_dir, ignore_errors=True)

    return {
        'mean': mean,
        'min': min_t,
        'max': max_t,
        'p95': p95,
        'per_worker': per_worker,
    }


def main():
    print("="*60)
    print("ZeroMQ Implementation Comparison Benchmark")
    print("="*60)
    print()
    print("Configuration:")
    print(f"  Workers: 4")
    print(f"  Iterations: 30")
    print(f"  Message Size: Large (100 offsets, ~800 bytes)")
    print()

    # Load implementations
    print("Loading implementations...")
    current = load_implementation(
        'tp_ipc_zmq_current',
        'kvcached/tp_ipc_zmq.py'
    )
    optimized = load_implementation(
        'tp_ipc_zmq_optimized',
        'kvcached/tp_ipc_zmq_optimized.py'
    )
    print("✓ Both implementations loaded\n")

    # Run benchmarks
    current_results = run_benchmark('current', current, workers=4, iterations=30)
    time.sleep(2)  # Let system settle
    optimized_results = run_benchmark('optimized', optimized, workers=4, iterations=30)

    # Compare results
    print(f"\n\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}\n")

    print(f"{'Metric':<20} {'Current':<12} {'Optimized':<12} {'Change':<12}")
    print(f"{'-'*60}")

    for metric in ['mean', 'min', 'max', 'p95']:
        curr_val = current_results[metric]
        opt_val = optimized_results[metric]
        change = ((curr_val - opt_val) / curr_val) * 100

        print(f"{metric.capitalize():<20} {curr_val:>10.3f}ms {opt_val:>10.3f}ms {change:>+10.1f}%")

    print(f"{'-'*60}")

    # Overall assessment
    improvement = ((current_results['mean'] - optimized_results['mean']) / current_results['mean']) * 100

    print(f"\n{'='*60}")
    if improvement > 5:
        print(f"✓ OPTIMIZED VERSION IS FASTER: {improvement:+.1f}%")
        print(f"  Recommendation: Use tp_ipc_zmq_optimized.py")
    elif improvement < -5:
        print(f"✗ CURRENT VERSION IS FASTER: {improvement:+.1f}%")
        print(f"  Recommendation: Keep tp_ipc_zmq.py")
    else:
        print(f"≈ PERFORMANCE IS SIMILAR: {improvement:+.1f}%")
        print(f"  Recommendation: Either version acceptable")
    print(f"{'='*60}\n")

    # Additional details
    print("Key Differences:")
    print("  Current Implementation:")
    print("    - Global context reuse")
    print("    - Socket created per call")
    print("    - Basic buffer settings")
    print()
    print("  Optimized Implementation:")
    print("    - Global context with io_threads=2")
    print("    - Socket pooling (persistent connections)")
    print("    - Dynamic buffer sizing (512MB on high-mem)")
    print("    - Comprehensive socket configuration")
    print()

    return improvement


if __name__ == "__main__":
    try:
        improvement = main()
        sys.exit(0 if improvement > 0 else 1)
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted")
        sys.exit(1)
