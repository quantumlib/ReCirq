import itertools

import cirq
import numpy as np
import pytest
from recirq.third_party import quaff


@pytest.mark.parametrize("n, seed", ((n, quaff.random_seed()) for n in range(1, 6)))
def test_parameter_resolution(n, seed):
    rng = np.random.default_rng(seed)
    qubits = cirq.LineQubit.range(n)
    parameterized_circuit = cirq.Circuit(
        quaff.get_parameterized_truncated_clifford_ops(qubits)
    )

    for _ in range(10):
        gate = quaff.TruncatedCliffordGate.random(n, rng)
        resolver = quaff.get_truncated_clifford_resolver(gate)
        resolved_circuit = cirq.resolve_parameters(parameterized_circuit, resolver)
        assert not cirq.is_parameterized(resolved_circuit)
        expected = cirq.unitary(gate(*qubits))
        actual = cirq.unitary(resolved_circuit)
        assert np.allclose(actual, expected)


@pytest.mark.parametrize(
    "part_sizes, seed",
    (
        (part_sizes, quaff.random_seed())
        for part_sizes in [(4, 4), (3, 3, 3), (1, 3, 2)]
    ),
)
def test_parameter_resolution_partitioned(part_sizes, seed):
    rng = np.random.default_rng(seed)
    num_qubits = sum(part_sizes)
    qubits = cirq.LineQubit.range(num_qubits)
    partition = [
        qubits[i : i + part_size]
        for i, part_size in zip(
            itertools.accumulate(part_sizes[:-1], initial=0), part_sizes
        )
    ]
    for part, part_size in zip(partition, part_sizes):
        assert len(part) == part_size

    parameterized_circuit = cirq.Circuit(
        quaff.get_parameterized_truncated_cliffords_ops(partition)
    )

    for _ in range(10):
        gates = [
            quaff.TruncatedCliffordGate.random(part_size, rng)
            for part_size in part_sizes
        ]
        resolver = quaff.get_truncated_cliffords_resolver(gates)
        resolved_circuit = cirq.resolve_parameters(parameterized_circuit, resolver)
        assert not cirq.is_parameterized(resolved_circuit)
        expected = cirq.Circuit(
            gate(*part) for gate, part in zip(gates, partition)
        ).unitary()
        actual = cirq.unitary(resolved_circuit)
        assert np.allclose(actual, expected)
