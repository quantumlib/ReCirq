# Copyright 2024 Google
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Iterable, Tuple

import cirq
import numpy as np
import pytest
from recirq.third_party import quaff

from recirq.qcqmc.blueprint import (BlueprintData, BlueprintParamsTrialWf,
                                    _get_truncated_cliffords)
from recirq.qcqmc.hamiltonian import HamiltonianData, HamiltonianFileParams
from recirq.qcqmc.qubit_maps import get_qubits_a_b_reversed
from recirq.qcqmc.trial_wf import (TrialWavefunctionData,
                                   TrialWavefunctionParams)


class FakeTrialWfParams(TrialWavefunctionParams):
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
    trial_wf_params = FakeTrialWfParams(
        name="fake",
        hamiltonian_params=HamiltonianFileParams("fake", "fake", 0, 0),
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
    bp_data = BlueprintData.build_blueprint_from_base_circuit(
        blueprint_params, base_circuit=cirq.Circuit()
    )
    assert isinstance(bp_data.compiled_circuit, cirq.AbstractCircuit)
    assert len(bp_data.resolvers) == 10


def test_small_blueprint(
    fixture_4_qubit_ham_and_trial_wf: Tuple[HamiltonianData, TrialWavefunctionData]
):
    _, trial_wf_data = fixture_4_qubit_ham_and_trial_wf
    trial_wf_params = trial_wf_data.params
    import attrs

    for k, v in attrs.asdict((trial_wf_params)).items():
        print(k, v)

    blueprint_params = BlueprintParamsTrialWf(
        name="blueprint_test",
        trial_wf_params=trial_wf_params,
        n_cliffords=5,
        qubit_partition=(tuple(get_qubits_a_b_reversed(n_orb=trial_wf_params.n_orb)),),
        seed=1,
    )
    blueprint = BlueprintData.build_blueprint_from_dependencies(
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

    blueprint = BlueprintData.build_blueprint_from_dependencies(
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


@pytest.mark.slow()
def test_medium(
    fixture_8_qubit_ham_and_trial_wf: Tuple[HamiltonianData, TrialWavefunctionData]
):
    _, trial_wf_data = fixture_8_qubit_ham_and_trial_wf
    trial_wf_params = trial_wf_data.params

    blueprint_params = BlueprintParamsTrialWf(
        name="blueprint_test_medium",
        trial_wf_params=trial_wf_params,
        n_cliffords=3,
        qubit_partition=(
            tuple(qubit for qubit in trial_wf_params.qubits_jordan_wigner_ordered),
        ),
        seed=1,
    )

    blueprint = BlueprintData.build_blueprint_from_dependencies(
        blueprint_params, dependencies={trial_wf_params: trial_wf_data}
    )

    assert len(list(blueprint.resolvers)) == 3

    resolved_circuits = list(blueprint.resolved_clifford_circuits)
    assert len(resolved_circuits) == 3
    for circuit_tuple in resolved_circuits:
        print(len(circuit_tuple), len(circuit_tuple[0]))
        assert len(circuit_tuple) == 1
        for circuit, qubits in zip(circuit_tuple, blueprint_params.qubit_partition):
            assert len(circuit.all_qubits()) == len(qubits)
            assert set(circuit.all_qubits()) == set(qubits)


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(0, 3, 2))
def test_quaff_respects_seed(seed):
    n_cliffords = 500

    rng_1a = np.random.default_rng(seed)
    rng_1b = np.random.default_rng(seed)
    rng_2a = np.random.default_rng(seed + 1)
    rng_2b = np.random.default_rng(seed + 1)

    qubits = cirq.LineQubit.range(4)
    qubit_partition = ((qubits[0], qubits[1]), (qubits[2], qubits[3]))

    cirq.Circuit(quaff.get_parameterized_truncated_cliffords_ops(qubit_partition))

    truncated_cliffords_1a = [
        [
            quaff.TruncatedCliffordGate.random(len(qubits), rng_1a)
            for qubits in qubit_partition
        ]
        for _ in range(n_cliffords)
    ]

    truncated_cliffords_1b = [
        [
            quaff.TruncatedCliffordGate.random(len(qubits), rng_1b)
            for qubits in qubit_partition
        ]
        for _ in range(n_cliffords)
    ]

    truncated_cliffords_2a = [
        [
            quaff.TruncatedCliffordGate.random(len(qubits), rng_2a)
            for qubits in qubit_partition
        ]
        for _ in range(n_cliffords)
    ]

    truncated_cliffords_2b = [
        [
            quaff.TruncatedCliffordGate.random(len(qubits), rng_2b)
            for qubits in qubit_partition
        ]
        for _ in range(n_cliffords)
    ]

    resolvers_1a = [
        quaff.get_truncated_cliffords_resolver(gates)
        for gates in truncated_cliffords_1a
    ]
    resolvers_1b = [
        quaff.get_truncated_cliffords_resolver(gates)
        for gates in truncated_cliffords_1b
    ]
    resolvers_2a = [
        quaff.get_truncated_cliffords_resolver(gates)
        for gates in truncated_cliffords_2a
    ]
    resolvers_2b = [
        quaff.get_truncated_cliffords_resolver(gates)
        for gates in truncated_cliffords_2b
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
