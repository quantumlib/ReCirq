# Copyright 2022 Google
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
from typing import Iterator, List
import itertools
import cirq
import numpy as np
import pytest

from . import optimizers
from . import toric_code_rectangle as tcr
from . import toric_code_state_prep as tcsp


def iter_cirq_single_qubit_cliffords() -> Iterator[cirq.SingleQubitCliffordGate]:
    """Iterates all 24 distinct (up to global phase) single qubit Clifford operations."""
    paulis: List[cirq.Pauli] = [cirq.X, cirq.Y, cirq.Z]
    for x_to_axis, z_to_axis in itertools.permutations(paulis, 2):
        for x_flip, z_flip in itertools.product([False, True], repeat=2):
            yield cirq.SingleQubitCliffordGate.from_xz_map(
                x_to=(x_to_axis, x_flip), z_to=(z_to_axis, z_flip)
            )


class TestConvertCnotMomentsToCzAndSimplifyHadamards:
    @classmethod
    def setup_class(cls):
        cls.control = cirq.LineQubit(0)
        cls.target = cirq.LineQubit(1)
        cls.cnot_moment = cirq.Moment([cirq.CNOT(cls.control, cls.target)])
        cls.cz_moment = cirq.Moment([cirq.CZ(cls.control, cls.target)])

        cls.x_qubit = cirq.LineQubit(0)
        cls.hadamard_qubit = cirq.LineQubit(1)
        cls.single_qubit_moment = cirq.Moment(
            [cirq.X(cls.x_qubit), cirq.H(cls.hadamard_qubit)]
        )
        cls.x_moment = cirq.Moment([cirq.X(cls.x_qubit)])
        cls.hadamard_moment = cirq.Moment([cirq.H(cls.hadamard_qubit)])

    def test_optimize_circuit_single_cnot(self):
        circuit = cirq.Circuit(self.cnot_moment)
        optimized = optimizers.convert_cnot_moments_to_cz_and_simplify_hadamards(
            circuit
        )
        assert optimized == cirq.Circuit(
            [self.hadamard_moment, self.cz_moment, self.hadamard_moment]
        )

    def test_optimize_circuit_single_cz(self):
        circuit = cirq.Circuit(self.cz_moment)
        optimized = optimizers.convert_cnot_moments_to_cz_and_simplify_hadamards(
            circuit
        )
        assert optimized == cirq.Circuit(self.cz_moment)

    def test_optimize_circuit_repeated_cnot(self):
        circuit = cirq.Circuit([self.cnot_moment, self.cnot_moment])
        optimized = optimizers.convert_cnot_moments_to_cz_and_simplify_hadamards(
            circuit
        )
        assert optimized == cirq.Circuit(
            [self.hadamard_moment, self.cz_moment, self.cz_moment, self.hadamard_moment]
        )

    def test_optimize_circuit_maintain_2q_layers(self):
        def q(i):
            return cirq.LineQubit(i)

        layers_of_pairs = [[(q(0), q(1)), (q(2), q(3))], [(q(1), q(2)), (q(3), q(4))]]
        cnot_layers = [
            cirq.Moment(cirq.CNOT(*pair) for pair in layer) for layer in layers_of_pairs
        ]

        circuit = cirq.Circuit()
        circuit += cirq.X(q(4))
        circuit += cnot_layers[0]
        circuit += cirq.X(q(0))
        circuit += cnot_layers[1]
        circuit += cirq.X(q(1))

        optimized = optimizers.convert_cnot_moments_to_cz_and_simplify_hadamards(
            circuit
        )

        layers_of_targets = [[pair[1] for pair in layer] for layer in layers_of_pairs]
        cz_layers = [
            cirq.Moment(cirq.CZ(*pair) for pair in layer) for layer in layers_of_pairs
        ]
        expected_circuit = cirq.Circuit()
        expected_circuit += cirq.Moment(
            [cirq.X(q(4))] + [cirq.H(t) for t in layers_of_targets[0]]
        )
        expected_circuit += cz_layers[0]
        expected_circuit += cirq.Moment(
            [cirq.X(q(0))]
            + [cirq.H(t) for t in layers_of_targets[0]]
            + [cirq.H(t) for t in layers_of_targets[1]]
        )
        expected_circuit += cz_layers[1]
        expected_circuit += cirq.Moment(
            [cirq.X(q(1))] + [cirq.H(t) for t in layers_of_targets[1]]
        )

        assert optimized == expected_circuit

    def test_moment_has_cnot(self):
        assert optimizers._has_cnot(self.cnot_moment)
        assert not optimizers._has_cnot(self.cz_moment)
        assert not optimizers._has_cnot(self.single_qubit_moment)

    def test_moment_has_2q_gates(self):
        assert optimizers._has_2q_gates(self.cnot_moment)
        assert optimizers._has_2q_gates(self.cz_moment)
        assert not optimizers._has_2q_gates(self.single_qubit_moment)

    def test_moment_is_exclusively_1q_gates(self):
        assert not optimizers._is_exclusively_1q_gates(self.cnot_moment)
        assert not optimizers._is_exclusively_1q_gates(self.cz_moment)
        assert optimizers._is_exclusively_1q_gates(self.single_qubit_moment)

    def test_qubits_with_hadamard(self):
        assert optimizers._qubits_with_hadamard(self.cnot_moment) == set()
        assert optimizers._qubits_with_hadamard(self.cz_moment) == set()
        assert optimizers._qubits_with_hadamard(self.single_qubit_moment) == {
            self.hadamard_qubit
        }

    def test_break_up_cnots(self):
        h0, cz, h1 = optimizers._break_up_cnots(self.cnot_moment)
        assert h0 == cirq.Moment([cirq.H(self.target)])
        assert cz == self.cz_moment
        assert h1 == cirq.Moment([cirq.H(self.target)])

        h0, cz, h1 = optimizers._break_up_cnots(self.cz_moment)
        assert h0 == cirq.Moment()
        assert cz == self.cz_moment
        assert h1 == cirq.Moment()

    def test_merge_hadamards(self):
        moments = optimizers._merge_hadamards(
            self.single_qubit_moment, self.single_qubit_moment
        )
        assert len(moments) == 2
        assert moments[0] == self.x_moment
        assert moments[1] == self.x_moment

        moments = optimizers._merge_hadamards(
            self.hadamard_moment, self.single_qubit_moment
        )
        assert len(moments) == 1
        assert moments[0] == self.x_moment

    def test_gates_are_close_identical(self):
        assert optimizers._gates_are_close(cirq.I, cirq.I)
        assert optimizers._gates_are_close(cirq.H, cirq.H)
        assert optimizers._gates_are_close(cirq.CNOT, cirq.CNOT)

    def test_gates_are_close_equivalent(self):
        assert optimizers._gates_are_close(cirq.I, cirq.HPowGate(exponent=0.0))
        assert optimizers._gates_are_close(cirq.H, cirq.HPowGate())
        assert optimizers._gates_are_close(cirq.CNOT, cirq.CNotPowGate())

    def test_gates_are_not_close(self):
        assert not optimizers._gates_are_close(cirq.I, cirq.H)
        assert not optimizers._gates_are_close(cirq.H, cirq.CNOT)


class TestDeferSingleQubitGates:
    def test_simple_example(self):
        qubits = [cirq.LineQubit(idx) for idx in range(5)]
        circuit = cirq.Circuit(
            cirq.Moment(cirq.H(q) for q in qubits),
            cirq.Moment(cirq.CZ(qubits[0], qubits[1])),
            cirq.Moment(cirq.X(qubits[0])),
            cirq.Moment(cirq.CZ(qubits[1], qubits[2])),
            cirq.Moment(cirq.X(qubits[0])),
            cirq.Moment(cirq.CZ(qubits[2], qubits[3])),
        )

        deferred_circuit = optimizers.defer_single_qubit_gates(circuit)
        expected = """
0: ───H───@───X───────X───────
          │
1: ───H───@───────@───────────
                  │
2: ───────────H───@───────@───
                          │
3: ───────────────────H───@───

4: ───────────────────H───────
        """.strip()
        assert deferred_circuit.to_text_diagram(qubit_order=qubits) == expected

    def test_mixed_moments(self):
        """Don't edit mixed moments, but pay attention to gates in them."""
        qubits = [cirq.LineQubit(idx) for idx in range(4)]
        circuit = cirq.Circuit(
            cirq.Moment(cirq.H(qubits[2])),
            cirq.Moment(cirq.H(qubits[3]), cirq.X(qubits[0])),
            cirq.Moment(cirq.CZ(qubits[0], qubits[1]), cirq.X(qubits[2])),
            cirq.Moment(cirq.X(qubits[0])),
            cirq.Moment(cirq.CZ(qubits[1], qubits[2])),
        )
        deferred_circuit = optimizers.defer_single_qubit_gates(circuit)
        expected = """
0: ───X───@───X───────
          │
1: ───────@───────@───
                  │
2: ───H───X───────@───

3: ───────────H───────
        """.strip()
        assert deferred_circuit.to_text_diagram(qubit_order=qubits) == expected

    def test_multiple_deferrals(self):
        qubits = [cirq.LineQubit(idx) for idx in range(2)]
        circuit = cirq.Circuit(
            cirq.Moment(cirq.X(qubits[0]), cirq.H(qubits[1])),
            cirq.Moment(cirq.X(qubits[0])),
            cirq.Moment(cirq.X(qubits[0]), cirq.Y(qubits[1])),
            cirq.Moment(cirq.X(qubits[0])),
            cirq.Moment(cirq.X(qubits[0]), cirq.Z(qubits[1])),
            cirq.Moment(cirq.X(qubits[0])),
            cirq.Moment(cirq.CZ(qubits[0], qubits[1])),
        )
        deferred_circuit = optimizers.defer_single_qubit_gates(circuit)
        expected = """
0: ───X───X───X───X───X───X───@───
                              │
1: ───────────────H───Y───Z───@───
        """.strip()
        assert deferred_circuit.to_text_diagram(qubit_order=qubits) == expected

    def test_toric_code(self):
        code = tcr.ToricCodeRectangle(cirq.GridQubit(0, 0), (1, 1), 1, 3)
        circuit = tcsp.toric_code_cnot_circuit(code)
        deferred_circuit = optimizers.defer_single_qubit_gates(circuit)
        assert cirq.linalg.allclose_up_to_global_phase(
            cirq.unitary(circuit), cirq.unitary(deferred_circuit)
        )


class TestInsertEchosOnActiveQubits:
    def test_simple_example(self):
        qubits = [cirq.LineQubit(idx) for idx in range(3)]
        circuit = cirq.Circuit(
            cirq.Moment(cirq.H(qubits[0])),
            cirq.Moment(cirq.CZ(qubits[0], qubits[1])),
            cirq.Moment(cirq.H(qubits[2])),
            cirq.Moment(cirq.CZ(qubits[0], qubits[2])),
            cirq.Moment(cirq.X(qubits[2])),
        )
        echo_circuit = optimizers.insert_echos_on_idle_qubits(circuit)
        expected = """
0: ───H───@───X───@───X───
          │       │
1: ───────@───X───┼───X───
                  │
2: ───────────H───@───Y───
        """.strip()
        assert echo_circuit.to_text_diagram(qubit_order=qubits) == expected

    def test_measurement(self):
        """Should stop manipulations once we hit a measurement."""
        qubits = [cirq.LineQubit(idx) for idx in range(3)]
        circuit = cirq.Circuit(
            cirq.Moment(cirq.H(q) for q in qubits),
            cirq.Moment(cirq.CZ(qubits[0], qubits[1])),
            cirq.Moment(cirq.H(qubits[0])),
            cirq.Moment(cirq.H(qubits[0])),
            cirq.Moment(cirq.measure(qubits[0])),
            cirq.Moment(cirq.H(qubits[0])),
            cirq.Moment(cirq.H(qubits[0])),
        )
        echo_circuit = optimizers.insert_echos_on_idle_qubits(circuit)
        expected = """
0: ───H───@───H───H───M───H───H───
          │
1: ───H───@───X───X───────────────

2: ───H───────X───X───────────────
                """.strip()
        assert echo_circuit.to_text_diagram(qubit_order=qubits) == expected

    @pytest.mark.parametrize(
        "gates, expected",
        [
            ([cirq.S, cirq.S], cirq.Z),
            ([cirq.X, cirq.Z], cirq.Y),
            ([cirq.H, cirq.X, cirq.H], cirq.Z),
        ],
    )
    def test_combine_1q_cliffords(self, gates, expected):
        assert optimizers._combine_1q_cliffords(*gates) == expected

    @pytest.mark.parametrize(
        "moment, expected",
        [
            (
                cirq.Moment(cirq.measure(cirq.LineQubit(0)), cirq.X(cirq.LineQubit(1))),
                True,
            ),
            (cirq.Moment(cirq.X(cirq.LineQubit(0))), False),
        ],
    )
    def test_moment_has_measurement(self, moment, expected):
        assert optimizers._has_measurement(moment) == expected

    def test_invalid_echo(self):
        with pytest.raises(ValueError):
            _ = optimizers.insert_echos_on_idle_qubits(cirq.Circuit(), echo=cirq.H)

    @pytest.mark.parametrize("resolve_to_hadamard", [False, True])
    def test_toric_code(self, resolve_to_hadamard: bool):
        code = tcr.ToricCodeRectangle(cirq.GridQubit(0, 0), (1, 1), 1, 3)
        circuit = tcsp.toric_code_cnot_circuit(code)
        echo_circuit = optimizers.insert_echos_on_idle_qubits(
            circuit, resolve_to_hadamard=resolve_to_hadamard
        )
        assert cirq.linalg.allclose_up_to_global_phase(
            cirq.unitary(circuit), cirq.unitary(echo_circuit)
        )

    @pytest.mark.parametrize("gate", iter_cirq_single_qubit_cliffords())
    def test_resolve_gate_to_hadamard_unitaries(self, gate: cirq.Gate):
        qubit = cirq.GridQubit(0, 0)
        circuit = cirq.Circuit(gate.on(qubit))

        decomposition = optimizers.resolve_gate_to_hadamard(gate)
        decomposition_circuit = cirq.Circuit(g.on(qubit) for g in decomposition)

        assert cirq.allclose_up_to_global_phase(
            cirq.unitary(circuit), cirq.unitary(decomposition_circuit)
        )

    @pytest.mark.parametrize("gate", iter_cirq_single_qubit_cliffords())
    def test_resolve_gate_to_hadamard_gives_hadamards(self, gate: cirq.Gate):
        """All "pi/2 XY" gates should have a hadamard in the middle of the decomposition."""
        decomposition = optimizers.resolve_gate_to_hadamard(gate)
        xz_gate = cirq.PhasedXZGate.from_matrix(cirq.unitary(gate))
        if np.isclose(abs(xz_gate.x_exponent), 0.5):
            assert decomposition[1] == cirq.H

    @pytest.mark.parametrize("gate", iter_cirq_single_qubit_cliffords())
    def test_resolve_gate_to_hadamard_xy_rotations_localized(self, gate: cirq.Gate):
        """Decomposition index 0 and 2 should not have XY rotations."""
        decomposition = optimizers.resolve_gate_to_hadamard(gate)
        for idx in [0, 2]:
            z_rotation = decomposition[idx]
            xz_gate = cirq.PhasedXZGate.from_matrix(cirq.unitary(z_rotation))
            assert np.isclose(float(xz_gate.x_exponent), 0.0)

    def test_resolve_moment_to_hadamard_passthrough_hadamards(self):
        q0, q1 = cirq.GridQubit.rect(1, 2)
        moment = cirq.Moment(cirq.H(q0), cirq.H(q1))
        assert optimizers.resolve_moment_to_hadamard(moment) == cirq.Circuit(moment)

    @pytest.mark.parametrize("gate", iter_cirq_single_qubit_cliffords())
    def test_resolve_moment_to_hadamard_each_clifford(self, gate: cirq.Gate):
        qubit = cirq.GridQubit(0, 0)
        clifford_moment = cirq.Moment(gate.on(qubit))
        resolved_circuit = optimizers.resolve_moment_to_hadamard(clifford_moment)
        assert cirq.allclose_up_to_global_phase(
            cirq.unitary(cirq.Circuit(clifford_moment)), cirq.unitary(resolved_circuit)
        )
        assert len(resolved_circuit) <= 3  # Allow up to 3 moments (e.g., --Z--H--Z--)
        assert not any(
            op.gate == cirq.I for op in resolved_circuit.all_operations()
        )  # Skip I's

        # Verify no SingleQubitCliffordGate sneaks through for pi/2 XY rotation cases
        xz_gate = cirq.PhasedXZGate.from_matrix(cirq.unitary(gate))
        if np.isclose(abs(xz_gate.x_exponent), 0.5):
            assert not any(
                isinstance(op.gate, cirq.SingleQubitCliffordGate)
                for op in resolved_circuit.all_operations()
            )
            assert any(op.gate == cirq.H for op in resolved_circuit.all_operations())
