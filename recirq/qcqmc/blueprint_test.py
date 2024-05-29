from typing import Iterable, Tuple

import cirq
import numpy as np
import pytest

from recirq.qcqmc.blueprint import (
    BlueprintParamsTrialWf,
    _get_truncated_cliffords,
    build_blueprint,
    build_blueprint_from_base_circuit,
)
from recirq.qcqmc.hamiltonian import HamiltonianData, LoadFromFileHamiltonianParams
from recirq.qcqmc.trial_wf import (
    TrialWavefunctionData,
    TrialWavefunctionParams,
    _get_qubits_a_b_reversed,
)


class DummyTrialWfParams(TrialWavefunctionParams):
    @property
    def bitstrings(self) -> Iterable[Tuple[bool, ...]]:
        raise NotImplementedError()

    @property
    def path_string(self) -> str:
        raise NotImplementedError()

    @property
    def qubits_jordan_wigner_ordered(self) -> Tuple[cirq.GridQubit, ...]:
        return tuple(cirq.GridQubit.rect(4, 1))

    @property
    def qubits_linearly_connected(self) -> Tuple[cirq.GridQubit, ...]:
        return tuple(cirq.GridQubit.rect(4, 1))


def test_build_blueprint_from_base_circuit():
    trial_wf_params = DummyTrialWfParams(
        name="dummy",
        hamiltonian_params=LoadFromFileHamiltonianParams("dummy", "dummy", 0, 0),
    )
    blueprint_params = BlueprintParamsTrialWf(
        name="blueprint_test",
        trial_wf_params=trial_wf_params,
        n_cliffords=10,
        qubit_partition=tuple(
            (q,) for q in trial_wf_params.qubits_jordan_wigner_ordered
        ),
        seed=1,
    )
    bp_data = build_blueprint_from_base_circuit(
        blueprint_params, base_circuit=cirq.Circuit()
    )
    assert isinstance(bp_data.compiled_circuit, cirq.AbstractCircuit)
    assert len(bp_data.resolvers) == 10


def test_small_blueprint(
    fixture_4_qubit_ham_and_trial_wf: Tuple[HamiltonianData, TrialWavefunctionData]
):
    _, trial_wf_data = fixture_4_qubit_ham_and_trial_wf
    trial_wf_params = trial_wf_data.params

    blueprint_params = BlueprintParamsTrialWf(
        name="blueprint_test",
        trial_wf_params=trial_wf_params,
        n_cliffords=5,
        qubit_partition=(tuple(_get_qubits_a_b_reversed(n_orb=trial_wf_params.n_orb)),),
        seed=1,
    )
    blueprint = build_blueprint(
        blueprint_params, dependencies={trial_wf_params: trial_wf_data}
    )

    assert len(list(blueprint.resolvers)) == 5

    resolved_circuits = list(blueprint.resolved_clifford_circuits)
    assert len(resolved_circuits) == 5
    for circuit_tuple in resolved_circuits:
        assert len(circuit_tuple) == 1
        for circuit, qubits in zip(circuit_tuple, blueprint_params.qubit_partition):
            assert len(circuit.all_qubits()) == len(qubits)
            assert set(circuit.all_qubits()) == set(qubits)


def test_small_blueprint_2(
    fixture_4_qubit_ham_and_trial_wf: Tuple[HamiltonianData, TrialWavefunctionData]
):
    _, trial_wf_data = fixture_4_qubit_ham_and_trial_wf
    trial_wf_params = trial_wf_data.params

    blueprint_params = BlueprintParamsTrialWf(
        name="blueprint_test_2",
        trial_wf_params=trial_wf_params,
        n_cliffords=5,
        qubit_partition=tuple(
            (qubit,) for qubit in trial_wf_params.qubits_jordan_wigner_ordered
        ),
        seed=1,
    )

    blueprint = build_blueprint(
        blueprint_params, dependencies={trial_wf_params: trial_wf_data}
    )

    assert len(list(blueprint.resolvers)) == 5

    resolved_circuits = list(blueprint.resolved_clifford_circuits)
    assert len(resolved_circuits) == 5
    for circuit_tuple in resolved_circuits:
        assert len(circuit_tuple) == 4
        for circuit, qubits in zip(circuit_tuple, blueprint_params.qubit_partition):
            assert len(circuit.all_qubits()) == len(qubits)
            assert set(circuit.all_qubits()) == set(qubits)


@pytest.mark.parametrize("seed", range(0, 3, 2))
def test_quaff_respects_seed(seed):
    n_cliffords = 500
    from recirq.qcqmc.quaff import (
        TruncatedCliffordGate,
        get_parameterized_truncated_cliffords_ops,
        get_truncated_cliffords_resolver,
    )

    rng_1a = np.random.default_rng(seed)
    rng_1b = np.random.default_rng(seed)
    rng_2a = np.random.default_rng(seed + 1)
    rng_2b = np.random.default_rng(seed + 1)

    qubits = cirq.LineQubit.range(4)
    qubit_partition = ((qubits[0], qubits[1]), (qubits[2], qubits[3]))

    parameterized_clifford_circuit = cirq.Circuit(
        get_parameterized_truncated_cliffords_ops(qubit_partition)
    )

    truncated_cliffords_1a = [
        [
            TruncatedCliffordGate.random(len(qubits), rng_1a)
            for qubits in qubit_partition
        ]
        for _ in range(n_cliffords)
    ]

    truncated_cliffords_1b = [
        [
            TruncatedCliffordGate.random(len(qubits), rng_1b)
            for qubits in qubit_partition
        ]
        for _ in range(n_cliffords)
    ]

    truncated_cliffords_2a = [
        [
            TruncatedCliffordGate.random(len(qubits), rng_2a)
            for qubits in qubit_partition
        ]
        for _ in range(n_cliffords)
    ]

    truncated_cliffords_2b = [
        [
            TruncatedCliffordGate.random(len(qubits), rng_2b)
            for qubits in qubit_partition
        ]
        for _ in range(n_cliffords)
    ]

    resolvers_1a = [
        get_truncated_cliffords_resolver(gates) for gates in truncated_cliffords_1a
    ]
    resolvers_1b = [
        get_truncated_cliffords_resolver(gates) for gates in truncated_cliffords_1b
    ]
    resolvers_2a = [
        get_truncated_cliffords_resolver(gates) for gates in truncated_cliffords_2a
    ]
    resolvers_2b = [
        get_truncated_cliffords_resolver(gates) for gates in truncated_cliffords_2b
    ]

    assert resolvers_1a == resolvers_1b
    assert resolvers_2a == resolvers_2b
    assert resolvers_1a != resolvers_2a

    assert isinstance(resolvers_1a[0]["h_0_0"], np.integer)


def test_truncated_cliffords_partitioning():
    qubit_partition = [cirq.LineQubit.range(2), cirq.LineQubit.range(2, 6)]
    n_cliffords = 7
    seed = 3

    truncated_cliffords = list(
        _get_truncated_cliffords(
            n_cliffords=n_cliffords, qubit_partition=qubit_partition, seed=seed
        )
    )

    assert len(truncated_cliffords) == 7
    for clifford_set in truncated_cliffords:
        assert len(clifford_set) == 2
        for part, clifford in zip(qubit_partition, clifford_set):
            assert len(part) == clifford._num_qubits
