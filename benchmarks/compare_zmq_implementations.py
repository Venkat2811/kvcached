#!/usr/bin/env python3
"""
Comprehensive benchmark comparing current vs optimized ZeroMQ implementations.
"""

import sys
import time
import importlib.util
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def load_module_from_file(name, filepath):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def run_benchmark_with_implementation(impl_name, impl_module, workers=4, iterations=50):
    """Run benchmark with a specific implementation."""
    import subprocess
    import os

    # Create a test script that uses the specific implementation
    test_script = f"""
import sys
import time
import multiprocessing as mp
sys.path.insert(0, '/home/user/kvcached')

# Import the specific implementation
{'from kvcached.tp_ipc_zmq import *' if impl_name == 'current' else 'from kvcached.tp_ipc_zmq_optimized import *'}

# Mock the vmm_ops module for non-GPU testing
import types
vmm_ops = types.ModuleType('vmm_ops')
vmm_ops.map_to_kv_tensors = lambda x: None
vmm_ops.unmap_from_kv_tensors = lambda x: None
vmm_ops.kv_tensors_created = lambda: True
sys.modules['kvcached.vmm_ops'] = vmm_ops

# Run the benchmark
import asyncio
import os

socket_dir = "/tmp/kvcached-bench-{impl_name}"
os.makedirs(socket_dir, exist_ok=True)

# Start workers
from kvcached.{'tp_ipc_zmq' if impl_name == 'current' else 'tp_ipc_zmq_optimized'} import start_worker_listener_thread
ready_queue = mp.Queue()

def mock_worker(rank, ready_queue):
    start_worker_listener_thread(rank)
    ready_queue.put(rank)
    import time
    while True:
        time.sleep(3600)

procs = []
for rank in range({workers}):
    p = mp.Process(target=mock_worker, args=(rank, ready_queue), daemon=True)
    p.start()
    procs.append(p)

# Wait for workers
for _ in range({workers}):
    ready_queue.get()
time.sleep(1)

# Benchmark
from kvcached.{'tp_ipc_zmq' if impl_name == 'current' else 'tp_ipc_zmq_optimized'} import broadcast_map_to_kv_tensors

offsets = list(range(0, 100 * 2097152, 2097152))
times = []

for i in range({iterations}):
    t0 = time.time()
    broadcast_map_to_kv_tensors({workers}, offsets)
    t1 = time.time()
    times.append((t1 - t0) * 1000)

import numpy as np
print(f"RESULT:{{np.mean(times):.3f}}:{{np.min(times):.3f}}:{{np.max(times):.3f}}:{{np.percentile(times, 95):.3f}}")

# Cleanup
for p in procs:
    p.terminate()
"""

    # Write and run test script
    script_path = f"/tmp/bench_{impl_name}.py"
    with open(script_path, 'w') as f:
        f.write(test_script)

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=120
        )

        # Parse results
        for line in result.stdout.split('\n'):
            if line.startswith('RESULT:'):
                parts = line.split(':')
                return {
                    'mean': float(parts[1]),
                    'min': float(parts[2]),
                    'max': float(parts[3]),
                    'p95': float(parts[4])
                }

        print(f"Error: Could not parse results from {impl_name}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return None

    except Exception as e:
        print(f"Error running {impl_name}: {e}")
        return None

if __name__ == "__main__":
    print("=" * 70)
    print("ZeroMQ Implementation Comparison Benchmark")
    print("=" * 70)
    print()

    configurations = [
        (2, 30),
        (4, 50),
        (8, 30),
    ]

    results = {}

    for workers, iters in configurations:
        print(f"\n{'='*70}")
        print(f"Configuration: {workers} workers, {iters} iterations")
        print(f"{'='*70}\n")

        results[(workers, iters)] = {}

        # Test current implementation
        print(f"Testing CURRENT implementation...")
        current_results = run_benchmark_with_implementation('current', None, workers, iters)
        if current_results:
            results[(workers, iters)]['current'] = current_results
            print(f"  Mean: {current_results['mean']:.3f} ms")
            print(f"  Min:  {current_results['min']:.3f} ms")
            print(f"  Max:  {current_results['max']:.3f} ms")
            print(f"  P95:  {current_results['p95']:.3f} ms")

        print()

        # Test optimized implementation
        print(f"Testing OPTIMIZED implementation...")
        opt_results = run_benchmark_with_implementation('optimized', None, workers, iters)
        if opt_results:
            results[(workers, iters)]['optimized'] = opt_results
            print(f"  Mean: {opt_results['mean']:.3f} ms")
            print(f"  Min:  {opt_results['min']:.3f} ms")
            print(f"  Max:  {opt_results['max']:.3f} ms")
            print(f"  P95:  {opt_results['p95']:.3f} ms")

        print()

        # Compare
        if current_results and opt_results:
            improvement = ((current_results['mean'] - opt_results['mean']) / current_results['mean']) * 100
            print(f"{'='*70}")
            print(f"COMPARISON ({workers} workers)")
            print(f"{'='*70}")
            print(f"Mean latency improvement: {improvement:+.1f}%")
            print(f"  Current:   {current_results['mean']:.3f} ms")
            print(f"  Optimized: {opt_results['mean']:.3f} ms")
            print(f"{'='*70}")

    # Final summary
    print(f"\n\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}\n")

    print(f"{'Workers':<10} {'Current (ms)':<15} {'Optimized (ms)':<15} {'Improvement':<15}")
    print(f"{'-'*70}")

    for (workers, iters), data in results.items():
        if 'current' in data and 'optimized' in data:
            curr = data['current']['mean']
            opt = data['optimized']['mean']
            imp = ((curr - opt) / curr) * 100
            print(f"{workers:<10} {curr:<15.3f} {opt:<15.3f} {imp:>+14.1f}%")

    print(f"{'-'*70}")
