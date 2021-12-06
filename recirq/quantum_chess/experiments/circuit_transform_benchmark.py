"""
Benchmarking tool for logical-to-hardware circuit transformations.

To run this:
  python recirq/quantum_chess/experiments/circuit_transform_benchmark.py \
      file1.qasm [file2.qasm [...]]

This script takes 1 or more positional arguments which are the paths to qasm
circuit files.

For each file, this script will:
  1) load the file into a cirq Circuit
  2) optimize/decompose the circuit for the Sycamore chip
  3) apply the CircuitTransformer
  4) validate the resulting circuit for the Sycamore chip

Some statistics about the transformation in step (3) will be printed, including
timing and increases to the circuit size.

Any qasm format circuit file should work.
See 'circuits' in this directory, which contains the circuit files used for
evaluation of https://ieeexplore.ieee.org/abstract/document/8976109.

For example, to run the benchmark on all the circuits in that directory:
  python recirq/quantum_chess/experiments/circuit_transform_benchmark.py \
      recirq/quantum_chess/experiments/circuits/*
NOTE however that the current dynamic look-ahead transformer implementation
contains a bug and will never terminate on some of the circuits in that
directory (https://github.com/quantumlib/ReCirq/issues/163).
"""

import cirq
import cirq_google as cg
import cProfile
import pstats
import sys

import recirq.quantum_chess.circuit_transformer as ct
from cirq.contrib.qasm_import import circuit_from_qasm
from typing import List
from timeit import default_timer as timer


def print_stats(time_sec: float, circuit: cirq.Circuit) -> None:
    """Prints some statistics about the circuit size.

    Args:
        time_sec: the time it took to produce the circuit
        circuit: print stats about this circuit
    """
    n_ops = len(list(circuit.all_operations()))
    print(f" qubits={len(circuit.all_qubits())}")
    print(f"    ops={n_ops}")
    print(f"moments={len(circuit.moments)}", flush=True)
    print(f"   time={time_sec:0.3f}s ({time_sec * 1e6 / n_ops :0.1f}us/op)")


def load_circuit_file(input_path: str) -> cirq.Circuit:
    """Reads a QASM circuit file and returns it as a cirq Circuit."""
    print(f"reading: {input_path}", flush=True)
    start = timer()
    with open(input_path, "r") as handle:
        contents = handle.read()
    circuit = circuit_from_qasm(contents)
    stop = timer()
    print_stats(stop - start, circuit)
    return circuit


def optimize(name: str, circuit: cirq.Circuit) -> cirq.Circuit:
    """Applies sycamore circuit decompositions/optimizations.

    Args:
        name: the name of the circuit for printing messages
        circuit: the circuit to optimize_for_sycamore
    """
    print(f"optimizing: {name}", flush=True)
    start = timer()
    optimized = cg.optimized_for_sycamore(circuit)
    stop = timer()
    print_stats(stop - start, optimized)
    return optimized


def benchmark_transform(
    name: str, circuit: cirq.Circuit, transformer: ct.CircuitTransformer
) -> cirq.Circuit:
    """Applies a transformation with profiling.

    Prints the (truncated) profile and some statistics about the size of the
    overhead in the transformed circuit.

    Args:
        name: the name of the circuit for printing messages
        circuit: the circuit to transform
        transformer: the transformation to apply
    Returns:
        the result of transformation.
    """
    ops_before = len(list(circuit.all_operations()))
    moments_before = len(circuit.moments)
    qubits_before = len(circuit.all_qubits())

    print(f"transforming: {name}", flush=True)
    start = timer()
    profile = cProfile.Profile()
    profile.enable()
    transformed = transformer.transform(circuit)
    profile.disable()
    stop = timer()
    print(f"finished: {name}", flush=True)
    print_stats(stop - start, transformed)

    ops_after = len(list(transformed.all_operations()))
    moments_after = len(transformed.moments)
    qubits_after = len(transformed.all_qubits())
    print("overhead:")
    print(f" qubits={qubits_after/qubits_before:0.3f}x")
    print(f"    ops={ops_after/ops_before:0.3f}x")
    print(f"moments={moments_after/moments_before:0.3f}x")
    print("profile:")
    stats = pstats.Stats(profile)
    stats.strip_dirs()
    stats.sort_stats("cumtime")
    stats.print_stats(40)
    sys.stdout.flush()
    return transformed


def main(input_files: List[str]) -> None:
    device = cg.Sycamore
    transformers = {
        "dynamic look-ahead placement": ct.DynamicLookAheadHeuristicCircuitTransformer(
            device
        ),
        "middle-out placement": ct.ConnectivityHeuristicCircuitTransformer(device),
    }
    for input_file in input_files:
        circuit = load_circuit_file(input_file)
        optimized = optimize(input_file, circuit)
        for name, transformer in transformers.items():
            suffix = f"{input_file} with {name}"
            try:
                transformed = benchmark_transform(suffix, optimized, transformer)
                device.validate_circuit(transformed)
            except Exception as e:
                print(f"failed: {suffix}")
                print(e)


if __name__ == "__main__":
    inputs = sys.argv[1:]
    inputs.sort()
    main(inputs)
