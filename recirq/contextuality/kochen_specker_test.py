# Copyright 2025 Google
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

import math

import cirq
import numpy as np
import pytest

import recirq.contextuality.kochen_specker as ks


@pytest.mark.parametrize("square", [ks.EASY_SQUARE, ks.HARD_SQUARE])
@pytest.mark.parametrize("axis", ["row", "col"])
@pytest.mark.parametrize("idx", range(3))
def test_satisfy_game_rule(square: ks.MagicSquare, axis: ks.Axis, idx: int) -> None:
    q0, anc, q1 = cirq.LineQubit.range(3)
    circuit = square.build_circuit_for_context(
        trios=[(q0, q1, anc)],
        context=(axis, idx),
    )
    result = cirq.Simulator().run(circuit, repetitions=100)

    paulis = (
        square.paulis[idx] if axis == "row" else [row[idx] for row in square.paulis]
    )

    parity = np.mean(sum(result.records[f"{pauli}_{axis}"] for pauli in paulis) % 2)
    assert parity == (0 if axis == "row" else 1)


@pytest.mark.parametrize("square", [ks.EASY_SQUARE, ks.HARD_SQUARE])
@pytest.mark.parametrize(
    "qnd_strategy", ["map_all_paulis_to_ancilla", "map_2q_paulis_to_ancilla"]
)
def test_estimate_chi(square: ks.MagicSquare, qnd_strategy: ks.QndStrategy) -> None:
    q0, anc, q1 = cirq.LineQubit.range(3)
    expt = square.kochen_specker_expt(
        q0, q1, anc, num_contexts=36, seed=0, qnd_strategy=qnd_strategy
    )
    sim = cirq.Simulator()
    result = expt.run(sim, repetitions=10)
    assert result.estimate_chi()[0] == 6


@pytest.mark.parametrize("square", [ks.EASY_SQUARE, ks.HARD_SQUARE])
@pytest.mark.parametrize(
    "qnd_strategy", ["map_all_paulis_to_ancilla", "map_2q_paulis_to_ancilla"]
)
def test_estimate_parities(
    square: ks.MagicSquare, qnd_strategy: ks.QndStrategy
) -> None:
    q0, anc, q1 = cirq.LineQubit.range(3)
    expt = square.kochen_specker_expt(
        q0, q1, anc, num_contexts=36, seed=0, qnd_strategy=qnd_strategy
    )
    sim = cirq.Simulator()
    result = expt.run(sim, repetitions=10)

    axes: list[ks.Axis] = ["col", "row"]
    for axis in axes:
        for idx in range(3):
            assert result.estimate_parities()[axis, idx][0] == (
                1 if axis == "row" else -1
            )


@pytest.mark.parametrize("square", [ks.EASY_SQUARE, ks.HARD_SQUARE])
@pytest.mark.parametrize(
    "qnd_strategy", ["map_all_paulis_to_ancilla", "map_2q_paulis_to_ancilla"]
)
def test_estimate_coincidences(
    square: ks.MagicSquare, qnd_strategy: ks.QndStrategy
) -> None:
    q0, anc, q1 = cirq.LineQubit.range(3)
    magic_square_experiment = square.kochen_specker_expt(
        q0, q1, anc, num_contexts=40, seed=0, qnd_strategy=qnd_strategy
    )
    sim = cirq.Simulator()
    result = magic_square_experiment.run(sim, repetitions=50)
    compatible, non_compatible, _ = result.estimate_coincidences()

    assert all(
        float(np.mean(coincidences_for_pauli)) == 1.0
        for coincidences_for_pauli in compatible.values()
    )

    assert all(
        math.isclose(np.mean(coincidence_prob_for_pauli), 0.5, abs_tol=0.1)
        for coincidence_prob_for_pauli in non_compatible.values()
    )
