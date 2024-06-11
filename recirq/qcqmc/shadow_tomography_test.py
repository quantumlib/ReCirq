import cirq
import numpy as np

from recirq.qcqmc import quaff
from recirq.qcqmc.bitstrings import get_bitstrings_a_b
from recirq.qcqmc.shadow_tomography import (
    get_amplitudes_from_samples_via_big_unitary,
    get_amplitudes_from_samples_via_clifford_simulation,
    get_amplitudes_from_samples_via_simulation,
)


def test_get_amplitudes_from_samples():
    rng = np.random.default_rng(52)
    nq = 4
    qs = cirq.LineQubit.range(nq)
    qubit_to_index = {q: i for i, q in enumerate(qs)}
    qubit_partition = [[qs[0], qs[1]], [qs[2], qs[3]]]

    clifford_gates = [
        quaff.TruncatedCliffordGate.random(num_qubits=len(qubits), seed=rng)
        for qubits in qubit_partition
    ]

    clifford_circuits = [
        cirq.Circuit(clifford.on(*part))
        for clifford, part in zip(clifford_gates, qubit_partition)
    ]
    inverse_cliffords = [cirq.inverse(cliff) for cliff in clifford_circuits]

    raw_samples = rng.choice([0, 1], size=(200, nq))
    valid_configurations = list(get_bitstrings_a_b(n_elec=nq // 2, n_orb=nq // 2))

    amps3 = get_amplitudes_from_samples_via_clifford_simulation(
        inverse_cliffords=inverse_cliffords,
        qubit_partition=qubit_partition,
        raw_samples_subset=raw_samples,
        valid_configurations=valid_configurations,
        qubit_to_index=qubit_to_index,
    )

    amps1 = get_amplitudes_from_samples_via_big_unitary(
        inverse_cliffords=inverse_cliffords,
        qubit_partition=qubit_partition,
        raw_samples_subset=raw_samples,
        valid_configurations=valid_configurations,
        qubit_to_index=qubit_to_index,
    )

    amps2 = get_amplitudes_from_samples_via_simulation(
        inverse_cliffords=inverse_cliffords,
        qubit_partition=qubit_partition,
        raw_samples_subset=raw_samples,
        valid_configurations=valid_configurations,
        qubit_to_index=qubit_to_index,
    )

    np.testing.assert_allclose(amps1, amps2)
    np.testing.assert_allclose(amps1, amps3)


def test_get_amplitudes_from_samples_2():
    rng = np.random.default_rng(52)
    nq = 4
    qs = cirq.LineQubit.range(nq)
    qubit_to_index = {q: i for i, q in enumerate(qs)}
    qubit_partition = [[qs[0], qs[1], qs[2], qs[3]]]

    clifford_gates = [
        quaff.TruncatedCliffordGate.random(num_qubits=len(qubits), seed=rng)
        for qubits in qubit_partition
    ]

    clifford_circuits = [
        cirq.Circuit(clifford.on(*part))
        for clifford, part in zip(clifford_gates, qubit_partition)
    ]
    inverse_cliffords = [cirq.inverse(cliff) for cliff in clifford_circuits]

    raw_samples = rng.choice([0, 1], size=(200, nq))
    valid_configurations = list(get_bitstrings_a_b(n_elec=nq // 2, n_orb=nq // 2))

    amps3 = get_amplitudes_from_samples_via_clifford_simulation(
        inverse_cliffords=inverse_cliffords,
        qubit_partition=qubit_partition,
        raw_samples_subset=raw_samples,
        valid_configurations=valid_configurations,
        qubit_to_index=qubit_to_index,
    )

    amps1 = get_amplitudes_from_samples_via_big_unitary(
        inverse_cliffords=inverse_cliffords,
        qubit_partition=qubit_partition,
        raw_samples_subset=raw_samples,
        valid_configurations=valid_configurations,
        qubit_to_index=qubit_to_index,
    )

    amps2 = get_amplitudes_from_samples_via_simulation(
        inverse_cliffords=inverse_cliffords,
        qubit_partition=qubit_partition,
        raw_samples_subset=raw_samples,
        valid_configurations=valid_configurations,
        qubit_to_index=qubit_to_index,
    )

    np.testing.assert_allclose(amps1, amps2)
    np.testing.assert_allclose(amps1, amps3)
