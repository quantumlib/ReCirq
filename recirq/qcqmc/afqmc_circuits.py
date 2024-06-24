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

"""Trial wavefunction circuit ansatz primitives."""

import itertools
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import attrs
import cirq
import numpy as np
import openfermion
from openfermion.circuits.primitives.state_preparation import (
    _ops_from_givens_rotations_circuit_description,
)
from openfermion.linalg.givens_rotations import givens_decomposition
from openfermion.linalg.sparse_tools import jw_sparse_givens_rotation

from recirq.qcqmc import layer_spec as lspec
from recirq.qcqmc import qubit_maps


class GeminalStatePreparationGate(cirq.Gate):
    def __init__(self, angle: float, inline_control: bool = False):
        """A 4-qubit gate that takes to produce a geminal state

        This state takes |00b0〉to sin(θ) |0011〉+ cos(θ) |1100〉and |00(1-b)0〉 to |0000〉.

        The perfect pairing operator T_pp (eqn C5) is a sum over pairs. Each
        pair consists of four qubits (two spin states; occupied and unoccupied
        for each), and is a rotation of the form implemented by this gate.

        Given qubits a, b, c, d: This composite gate decomposes into a
        Givens rotation on qubits a, c followed by two CNOTs copying a to b
        and c to d.

        See: https://github.com/quantumlib/ReCirq/issues/348 for tracking issue
        on how inline_control is used in experiment.

        Args:
            angle: The angle θ.
            inline_control: When True, the state preparation is controlled by the
                state of the third qubit.indicator,
                If inline_control is False, the GeminalStatePreparationGate acts on
                |0000> to prepare the geminal state and maps |0010> to |0000>.
                If inline_control is True, then the GeminalStatePreparationGate acts
                on |0010> to prepare the geminal state and maps |0000> to
                |0000>.
        """
        self.angle = -angle + np.pi / 2
        self.inline_control = inline_control

    def _decompose_(self, qubits):
        # assumes linear connectivity: a - c - d - b

        a, b, c, d = qubits
        if not self.inline_control:
            yield cirq.X(c)
        yield cirq.ZPowGate(global_shift=-1)(c)
        yield cirq.FSimGate(-self.angle + np.pi, 0)(c, a)
        yield cirq.ZPowGate(exponent=-0.5, global_shift=2)(a)
        yield cirq.CNOT(a, b)
        yield cirq.CNOT(c, d)

    def num_qubits(self):
        return 4

    def _circuit_diagram_info_(
        self, args: "cirq.CircuitDiagramInfoArgs"
    ) -> Tuple[str, ...]:
        return ("G",) * self.num_qubits()


class ControlledGeminalStatesPreparationGate(cirq.Gate):
    """A multi-qubit gate that prepares N geminal states conditioned on the state of the 0th qubit.

    Given an initial state a|0> + b|1> for the 0th qubit, a series of CNOT and
    SWAP gates create a len(self.angles)-qubit GHZ state distributed between the
    0th qubit and the 4n-1st qubit for n=1, 2, ..., len(self.angles).  This GHZ
    state controls the action of each GeminalStatePreparationGate, resulting in
    a final state that is a|0>|0....0> + b|1>|geminal 1>|geminal 2>...

    Args:
        angles: The N angles for the geminal states.
    """

    def __init__(self, angles: Iterable[float]):
        self.angles = tuple(angles)

    def _decompose_(self, qubits):
        # assumes connectivity
        # 2 4 6 8 10 12 .......
        # 1 3 5 7 9  11 .......
        #                  0
        ancilla = qubits[0]
        comp_qubits = qubits[1:]
        middle = get_ancilla_col(self.n_pairs) // 2
        yield cirq.CNOT(ancilla, comp_qubits[4 * middle + 2])
        for i in range(middle - 1, -1, -1):
            yield cirq.CNOT(comp_qubits[4 * i + 6], comp_qubits[4 * i + 4])
            yield cirq.SWAP(comp_qubits[4 * i + 4], comp_qubits[4 * i + 2])
        for i in range(middle + 1, self.n_pairs):
            yield cirq.CNOT(comp_qubits[4 * i - 2], comp_qubits[4 * i])
            yield cirq.SWAP(comp_qubits[4 * i], comp_qubits[4 * i + 2])
        for i, angle in enumerate(self.angles):
            yield GeminalStatePreparationGate(angle, 1)(
                *comp_qubits[4 * i : 4 * (i + 1)]
            )

    def num_qubits(self):
        return 1 + 4 * len(self.angles)

    @property
    def n_pairs(self) -> int:
        return len(self.angles)


class SlaterDeterminantPreparationGate(cirq.Gate):
    def __init__(self, orbitals: np.ndarray, from_reference: bool = False):
        """Prepares a Slater determinant using the Jordan Wigner representation.

        Args:
            orbitals: An η × N array whose rows are the occupied orbitals.
            from_reference: Whether the initial state is the reference state
                |1⋯0〉or the zero state |0⋯0〉. Defaults to zero state.
        """
        self.orbitals = orbitals
        self.from_reference = from_reference

    def _decompose_(self, qubits):
        n_occupied = self.orbitals.shape[0]

        decomposition, left_unitary, diagonal = givens_decomposition(self.orbitals)
        circuit_description = list(reversed(decomposition))
        phase = np.linalg.det(left_unitary).conj() * diagonal.prod()
        assert np.isclose(abs(phase), 1)
        exponent = np.real_if_close(-1j * np.log(phase) / np.pi)
        assert np.isclose(np.exp(1j * np.pi * exponent), phase)

        if not self.from_reference:
            for qubit in qubits[:n_occupied]:
                yield cirq.X(qubit)
        yield cirq.ZPowGate(exponent=exponent)(qubits[0])
        yield _ops_from_givens_rotations_circuit_description(
            qubits, circuit_description
        )

    def num_qubits(self):
        return self.orbitals.shape[1]


class ControlledSlaterDeterminantPreparationGate(cirq.Gate):
    def __init__(self, orbitals: np.ndarray):
        """Prepares a conditional Slater determinant under JW.

        The ancilla is the first qubit.

        Args:
            orbitals: An η × N array whose rows are the occupied orbitals.
        """
        self.orbitals = orbitals

    def _decompose_(self, qubits):
        n_occupied, n_orb = self.orbitals.shape
        if not (n_occupied and n_orb):
            return

        ancilla, comp_qubits = qubits[0], qubits[1:]
        ancilla_col = max(0, get_ancilla_col(n_orb // 4))
        assert 0 <= ancilla_col < n_orb

        yield cirq.CNOT(ancilla, comp_qubits[ancilla_col])

        for col in range(ancilla_col, n_occupied - 1, -1):
            yield cirq.SWAP(comp_qubits[col], comp_qubits[col - 1])

        for col in reversed(range(max(ancilla_col, n_occupied - 1))):
            yield cirq.CNOT(comp_qubits[col + 1], comp_qubits[col])

        for col in range(ancilla_col + 1, n_occupied):
            yield cirq.CNOT(comp_qubits[col - 1], comp_qubits[col])

        yield SlaterDeterminantPreparationGate(self.orbitals, True)(*qubits[1:])

    def num_qubits(self):
        return self.orbitals.shape[1] + 1


def get_phase_state_preparation_gate(angle: float) -> cirq.Gate:
    """A gate that takes |0〉to (|0〉+ exp(i θ)|1〉)/sqrt(2)."""
    return cirq.MatrixGate(
        np.sqrt(1 / 2) * np.array(((1, np.exp(-1j * angle)), (np.exp(1j * angle), -1)))
    )


###############################################################################
# States
###############################################################################


def get_geminal_state(angle: float) -> np.ndarray:
    """Returns the geminal state cos(θ) |0011〉+ sin(θ) |1100〉.

    Args:
        angle: The angle θ.
    """
    return sum(
        cirq.one_hot(index=i, shape=16, value=a, dtype=float)
        for i, a in [(0b0011, np.sin(angle)), (0b1100, np.cos(angle))]
    )  # type: ignore


def get_geminal_states_prod(angles: Iterable[float]) -> np.ndarray:
    """Returns the Kronecker product of geminal states with angles (θ_i)_i.

    Args:
        angles: The angles (θ_1, ...).
    """
    prod = np.ones(())
    for angle in angles:
        prod = np.kron(prod, get_geminal_state(angle))
    return prod


def jw_slater_determinant(slater_determinant_matrix):
    """A version of openfermion.jw_slater_determinant that preserves phase."""
    decomposition, left_unitary, diagonal = givens_decomposition(
        slater_determinant_matrix
    )
    circuit_description = list(reversed(decomposition))
    start_orbitals = range(slater_determinant_matrix.shape[0])
    n_qubits = slater_determinant_matrix.shape[1]

    state = openfermion.jw_configuration_state(start_orbitals, n_qubits)
    for parallel_ops in circuit_description:
        for operation in parallel_ops:
            i, j, theta, phi = operation
            state = jw_sparse_givens_rotation(i, j, theta, phi, n_qubits).dot(state)

    return state * np.linalg.det(left_unitary).conj() * diagonal.prod()


###############################################################################
# Circuit utils
###############################################################################


def get_ancilla_col(n_pairs: int) -> int:
    """The column of the ancilla qubit.

    Args:
        n_pairs: The number of pairs of spatial orbitals.
    """
    return 1 + 2 * ((n_pairs - 1) // 2)


def get_computational_qubits_and_ancilla(
    n_spin_orbitals: int,
) -> Tuple[List[cirq.GridQubit], cirq.GridQubit]:
    """The computational and ancilla qubits.

    All are GridQubits. The even orbitals are mapped to row 0 and odd to row 1.
    The ancilla is in row -1.

    Like this:

        1 3 5 7 9 11 .......
        0 2 4 6 8 10 .......
                        a

    Args:
        n_spin_orbitals: The number of spin orbitals.
    """
    return (
        [cirq.GridQubit(i % 2, i // 2) for i in range(n_spin_orbitals)],
        cirq.GridQubit(-1, get_ancilla_col(n_spin_orbitals // 4)),
    )


def get_jordan_wigner_string(n_pairs: int, reverse: bool = False) -> Iterable[int]:
    """Returns the JW ordering.

    The i-th element is the index of the orbital mapped to the i-th qubit.

    Args:
        n_pairs: The number of pairs of spatial orbitals.
        reverse: Whether to return the inverse permutation. Defaults to False.
    """
    if reverse:
        return itertools.chain(
            range(0, 4 * n_pairs, 2), reversed(range(1, 4 * n_pairs, 2))
        )
    return itertools.chain(
        *zip(range(2 * n_pairs), reversed(range(2 * n_pairs, 4 * n_pairs)))
    )


###############################################################################
# Circuits and simulation
###############################################################################


def get_geminal_and_slater_det_overlap(
    *, angles: Iterable[float], orbitals: np.ndarray
) -> complex:
    """Returns the overlap between geminal states and a Slater determinant.

        angles: The angles defining the geminal states.
        orbitals: An η × N array whose rows are the occupied orbitals.

    The Slater determinant state is transformed using Jordan-Wigner.
    """

    return np.dot(
        get_geminal_states_prod(angles), jw_slater_determinant(orbitals)
    ).item()


def get_geminal_and_slater_det_overlap_circuit(
    *,
    ancilla: cirq.Qid,
    comp_qubits: Iterable[cirq.Qid],
    orbitals: np.ndarray,
    angles: np.ndarray,
    phase: float = 0,
    decompose: bool = False,
    fold_in_measurement: bool = False,
) -> cirq.OP_TREE:
    """
    Args:
        ancilla: The ancilla qubit.
        comp_qubits: The computational qubits.
        orbitals: An η × N array.
        angles: An iterable of N / 4 angles.
        phase: The phase to measure. Defaults to 0 (real part).
        decompose: Whether or not to decompose into 2-qubit gates. Defaults to
            False.
        fold_in_measurement: Whether or not to leave out final Hadamard on
            ancilla and neighboring CNOTs. Defaults to False.


    Let r·exp(i φ) be the overlap 〈geminal | SD 〉. When the returned circuit
    is applied to the all-zero state and the ancilla qubit measured, the
    difference between the probability of 0 and that of 1 is equal to
    r·cos(φ + θ), where θ is the given phase.
    """
    _, n_orbs = orbitals.shape
    n_pairs = n_orbs // 4
    assert 4 * n_pairs == n_orbs
    assert angles.shape == (n_pairs,)

    comp_qubits = tuple(comp_qubits)
    qubits = (ancilla,) + comp_qubits
    yield get_phase_state_preparation_gate(phase)(ancilla)

    reverse_jw_indices = tuple(get_jordan_wigner_string(n_pairs, True))
    basis_change_op = ControlledSlaterDeterminantPreparationGate(
        orbitals[:, reverse_jw_indices]
    )(ancilla, *(comp_qubits[i] for i in reverse_jw_indices))

    def keep(operation):
        return len(operation.qubits) <= 2

    if fold_in_measurement:
        inverse_geminal_prep_ops = [
            cirq.inverse(GeminalStatePreparationGate(angle))(*comp_qubits[i : i + 4])
            for i, angle in zip(range(0, n_orbs, 4), angles)
        ]
    else:
        inverse_geminal_prep_ops = [
            cirq.inverse(ControlledGeminalStatesPreparationGate(angles))(*qubits)
        ]

    if decompose:
        yield cirq.decompose(basis_change_op, keep=keep)
        for operation in inverse_geminal_prep_ops:
            yield cirq.decompose(operation, keep=keep)
    else:
        yield basis_change_op
        yield inverse_geminal_prep_ops

    if not fold_in_measurement:
        yield cirq.H(ancilla)


def get_geminal_and_slater_det_overlap_via_simulation(
    *,
    angles: np.ndarray,
    orbitals: np.ndarray,
    phase: float = 0,
    fold_in_measurement: bool = False,
) -> complex:
    """
    Gets the overlap by simulating the Hadamard test.

    Args:
        orbitals: An η × N array.
        angles: An iterable of N / 4 angles.
        phase: The phase to measure. Defaults to 0 (real part).
        fold_in_measurement: Whether or not to fold measurements into the
            circuit. Defaults to False.

    See get_geminal_and_slater_det_overlap and
    get_geminal_and_slater_det_overlap_circuit.
    """
    n_orbs = orbitals.shape[1]
    comp_qubits, ancilla = get_computational_qubits_and_ancilla(n_orbs)
    qubits = [ancilla] + comp_qubits
    circuit = cirq.Circuit(
        get_geminal_and_slater_det_overlap_circuit(
            ancilla=ancilla,
            comp_qubits=comp_qubits,
            orbitals=orbitals,
            angles=angles,
            phase=phase,
            fold_in_measurement=fold_in_measurement,
        )
    )
    final_state = (
        cirq.Simulator().simulate(circuit, qubit_order=qubits).final_state_vector
    )

    if fold_in_measurement:
        measurement = cirq.Circuit(
            (cirq.X(ancilla), (cirq.X(comp_qubits[i + 2]) for i in range(0, n_orbs, 4)))
        ).unitary(qubit_order=qubits)
    else:
        measurement = cirq.Circuit(cirq.Z(ancilla)).unitary(qubit_order=qubits)
    return final_state.T.conj().dot(measurement.dot(final_state))


def get_givens_gate(qubits: Tuple[cirq.Qid, ...], param: float) -> cirq.Operation:
    """Get a the givens rotation gate on two qubits.

    Args:
        qubits: The two qubits to apply the gate to.
        param: The parameter for the givens rotation.

    Returns:
        The givens rotation gate.
    """
    return cirq.givens(param).on(qubits[0], qubits[1])


def get_charge_charge_gate(
    qubits: Tuple[cirq.Qid, ...], param: float
) -> cirq.Operation:
    """Get the cirq charge-charge gate.

    Args:
        qubits: Two qubits you want to apply the gate to.
        param: The parameter for the charge-charge interaction.

    Returns:
        The charge-charge gate.
    """
    return cirq.CZ(qubits[0], qubits[1]) ** (-param / np.pi)


def get_layer_gates(
    layer_spec: lspec.LayerSpec,
    n_elec: int,
    params: np.ndarray,
    fermion_index_to_qubit_map: Dict[int, cirq.GridQubit],
) -> List[cirq.Operation]:
    """Gets the gates for a hardware efficient layer of the ansatz.

    Args:
        layer_spec: The layer specification.
        n_elec: The number of electrons.
        params: The variational parameters for the hardware efficient gate layer.
        fermion_index_to_qubit_map: A mapping between fermion mode indices and qubits.

    Returns:
        A list of gates for the layer.
    """

    indices_list = lspec.get_layer_indices(layer_spec, n_elec)

    gate_funcs = {"givens": get_givens_gate, "charge_charge": get_charge_charge_gate}
    gate_func = gate_funcs[layer_spec.base_gate]

    gates = []
    for indices, param in zip(indices_list, params):
        qubits = tuple(fermion_index_to_qubit_map[ind] for ind in indices)
        gates.append(gate_func(qubits, param))

    return gates


def get_heuristic_circuit(
    layer_specs: Sequence[lspec.LayerSpec],
    n_elec: int,
    params: np.ndarray,
    fermion_index_to_qubit_map: Dict[int, cirq.GridQubit],
) -> cirq.Circuit:
    """Get a circuit for the heuristic ansatz.

    Args:
        layer_specs: The layer specs for the heuristic layers.
        n_elec: The number of electrons.
        params: The variational parameters for the circuit.
        fermion_index_to_qubit_map: A mapping between fermion mode indices and qubits.

    Returns:
        A circuit for the heuristic ansatz.
    """
    gates: List[cirq.Operation] = []

    for layer_spec in layer_specs:
        params_slice = params[len(gates) :]
        gates += get_layer_gates(
            layer_spec, n_elec, params_slice, fermion_index_to_qubit_map
        )

    return cirq.Circuit(gates)


def get_4_qubit_pp_circuits(
    *,
    two_body_params: np.ndarray,
    n_elec: int,
    heuristic_layers: Tuple[lspec.LayerSpec, ...],
) -> Tuple[cirq.Circuit, cirq.Circuit]:
    """A helper function that builds the circuits for the four qubit ansatz.

    We map the fermionic orbitals to grid qubits like so:
    3 1
    2 0
    """
    assert n_elec == 2

    fermion_index_to_qubit_map = qubit_maps.get_4_qubit_fermion_qubit_map()
    geminal_gate = GeminalStatePreparationGate(two_body_params[0], inline_control=True)

    ansatz_circuit = cirq.Circuit(
        cirq.decompose(
            geminal_gate.on(
                fermion_index_to_qubit_map[0],
                fermion_index_to_qubit_map[1],
                fermion_index_to_qubit_map[2],
                fermion_index_to_qubit_map[3],
            )
        )
    )

    heuristic_layer_circuit = get_heuristic_circuit(
        heuristic_layers, n_elec, two_body_params[1:], fermion_index_to_qubit_map
    )

    ansatz_circuit += heuristic_layer_circuit

    indicator = fermion_index_to_qubit_map[2]
    superposition_circuit = cirq.Circuit([cirq.H(indicator) + ansatz_circuit])
    ansatz_circuit = cirq.Circuit([cirq.X(indicator) + ansatz_circuit])

    return superposition_circuit, ansatz_circuit


def get_8_qubit_circuits(
    *,
    two_body_params: np.ndarray,
    n_elec: int,
    heuristic_layers: Tuple[lspec.LayerSpec, ...],
) -> Tuple[cirq.Circuit, cirq.Circuit]:
    """A helper function that builds the circuits for the four qubit ansatz.

    We map the fermionic orbitals to grid qubits like so:
    3 5 1 7
    2 4 0 6
    """
    fermion_index_to_qubit_map = qubit_maps.get_8_qubit_fermion_qubit_map()

    geminal_gate_1 = GeminalStatePreparationGate(
        two_body_params[0], inline_control=True
    )
    geminal_gate_2 = GeminalStatePreparationGate(
        two_body_params[1], inline_control=True
    )

    # We'll add the initial bit flips later.
    ansatz_circuit = cirq.Circuit(
        cirq.decompose(
            geminal_gate_1.on(
                fermion_index_to_qubit_map[2],
                fermion_index_to_qubit_map[3],
                fermion_index_to_qubit_map[4],
                fermion_index_to_qubit_map[5],
            )
        ),
        cirq.decompose(
            geminal_gate_2.on(
                fermion_index_to_qubit_map[0],
                fermion_index_to_qubit_map[1],
                fermion_index_to_qubit_map[6],
                fermion_index_to_qubit_map[7],
            )
        ),
    )

    heuristic_layer_circuit = get_heuristic_circuit(
        heuristic_layers, n_elec, two_body_params[2:], fermion_index_to_qubit_map
    )

    ansatz_circuit += heuristic_layer_circuit

    superposition_circuit = (
        cirq.Circuit(
            [
                cirq.H(fermion_index_to_qubit_map[0]),
                cirq.CNOT(fermion_index_to_qubit_map[0], fermion_index_to_qubit_map[6]),
                cirq.SWAP(fermion_index_to_qubit_map[0], fermion_index_to_qubit_map[4]),
            ]
        )
        + ansatz_circuit
    )

    ansatz_circuit = (
        cirq.Circuit(
            [
                cirq.X(fermion_index_to_qubit_map[4]),
                cirq.X(fermion_index_to_qubit_map[6]),
            ]
        )
        + ansatz_circuit
    )

    return superposition_circuit, ansatz_circuit


def get_12_qubit_circuits(
    *,
    two_body_params: np.ndarray,
    n_elec: int,
    heuristic_layers: Tuple[lspec.LayerSpec, ...],
) -> Tuple[cirq.Circuit, cirq.Circuit]:
    """A helper function that builds the circuits for the four qubit ansatz.

    We map the fermionic orbitals to grid qubits like so:
    5 7 3 9 1 11
    4 6 2 8 0 10
    """

    fermion_index_to_qubit_map = qubit_maps.get_12_qubit_fermion_qubit_map()

    geminal_gate_1 = GeminalStatePreparationGate(
        two_body_params[0], inline_control=True
    )
    geminal_gate_2 = GeminalStatePreparationGate(
        two_body_params[1], inline_control=True
    )
    geminal_gate_3 = GeminalStatePreparationGate(
        two_body_params[2], inline_control=True
    )

    # We'll add the initial bit flips later.
    ansatz_circuit = cirq.Circuit(
        cirq.decompose(
            geminal_gate_1.on(
                fermion_index_to_qubit_map[4],
                fermion_index_to_qubit_map[5],
                fermion_index_to_qubit_map[6],
                fermion_index_to_qubit_map[7],
            )
        ),
        cirq.decompose(
            geminal_gate_2.on(
                fermion_index_to_qubit_map[2],
                fermion_index_to_qubit_map[3],
                fermion_index_to_qubit_map[8],
                fermion_index_to_qubit_map[9],
            )
        ),
        cirq.decompose(
            geminal_gate_3.on(
                fermion_index_to_qubit_map[0],
                fermion_index_to_qubit_map[1],
                fermion_index_to_qubit_map[10],
                fermion_index_to_qubit_map[11],
            )
        ),
    )

    heuristic_layer_circuit = get_heuristic_circuit(
        heuristic_layers, n_elec, two_body_params[3:], fermion_index_to_qubit_map
    )

    ansatz_circuit += heuristic_layer_circuit

    superposition_circuit = (
        cirq.Circuit(
            [
                cirq.H(fermion_index_to_qubit_map[8]),
                cirq.CNOT(fermion_index_to_qubit_map[8], fermion_index_to_qubit_map[0]),
                cirq.CNOT(fermion_index_to_qubit_map[8], fermion_index_to_qubit_map[2]),
                cirq.SWAP(
                    fermion_index_to_qubit_map[0], fermion_index_to_qubit_map[10]
                ),
                cirq.SWAP(fermion_index_to_qubit_map[2], fermion_index_to_qubit_map[6]),
            ]
        )
        + ansatz_circuit
    )

    ansatz_circuit = (
        cirq.Circuit(
            [
                cirq.X(fermion_index_to_qubit_map[6]),
                cirq.X(fermion_index_to_qubit_map[8]),
                cirq.X(fermion_index_to_qubit_map[10]),
            ]
        )
        + ansatz_circuit
    )

    return superposition_circuit, ansatz_circuit


def get_16_qubit_circuits(
    *,
    two_body_params: np.ndarray,
    n_elec: int,
    heuristic_layers: Tuple[lspec.LayerSpec, ...],
) -> Tuple[cirq.Circuit, cirq.Circuit]:
    """A helper function that builds the circuits for the four qubit ansatz.

    We map the fermionic orbitals to grid qubits like so:
    7 9 5 11 3 13 1 15
    6 8 4 10 2 12 0 14
    """
    fermion_index_to_qubit_map = qubit_maps.get_16_qubit_fermion_qubit_map()

    geminal_gate_1 = GeminalStatePreparationGate(
        two_body_params[0], inline_control=True
    )
    geminal_gate_2 = GeminalStatePreparationGate(
        two_body_params[1], inline_control=True
    )
    geminal_gate_3 = GeminalStatePreparationGate(
        two_body_params[2], inline_control=True
    )
    geminal_gate_4 = GeminalStatePreparationGate(
        two_body_params[3], inline_control=True
    )

    # We'll add the initial bit flips later.
    ansatz_circuit = cirq.Circuit(
        cirq.decompose(
            geminal_gate_1.on(
                fermion_index_to_qubit_map[6],
                fermion_index_to_qubit_map[7],
                fermion_index_to_qubit_map[8],
                fermion_index_to_qubit_map[9],
            )
        ),
        cirq.decompose(
            geminal_gate_2.on(
                fermion_index_to_qubit_map[4],
                fermion_index_to_qubit_map[5],
                fermion_index_to_qubit_map[10],
                fermion_index_to_qubit_map[11],
            )
        ),
        cirq.decompose(
            geminal_gate_3.on(
                fermion_index_to_qubit_map[2],
                fermion_index_to_qubit_map[3],
                fermion_index_to_qubit_map[12],
                fermion_index_to_qubit_map[13],
            )
        ),
        cirq.decompose(
            geminal_gate_4.on(
                fermion_index_to_qubit_map[0],
                fermion_index_to_qubit_map[1],
                fermion_index_to_qubit_map[14],
                fermion_index_to_qubit_map[15],
            )
        ),
    )

    heuristic_layer_circuit = get_heuristic_circuit(
        heuristic_layers, n_elec, two_body_params[4:], fermion_index_to_qubit_map
    )

    ansatz_circuit += heuristic_layer_circuit

    superposition_circuit = (
        cirq.Circuit(
            [
                cirq.H(fermion_index_to_qubit_map[10]),
                cirq.CNOT(
                    fermion_index_to_qubit_map[10], fermion_index_to_qubit_map[2]
                ),
                cirq.SWAP(
                    fermion_index_to_qubit_map[2], fermion_index_to_qubit_map[12]
                ),
                cirq.CNOT(
                    fermion_index_to_qubit_map[10], fermion_index_to_qubit_map[4]
                ),
                cirq.CNOT(
                    fermion_index_to_qubit_map[12], fermion_index_to_qubit_map[0]
                ),
                cirq.SWAP(fermion_index_to_qubit_map[4], fermion_index_to_qubit_map[8]),
                cirq.SWAP(
                    fermion_index_to_qubit_map[0], fermion_index_to_qubit_map[14]
                ),
            ]
        )
        + ansatz_circuit
    )

    ansatz_circuit = (
        cirq.Circuit(
            [
                cirq.X(fermion_index_to_qubit_map[8]),
                cirq.X(fermion_index_to_qubit_map[10]),
                cirq.X(fermion_index_to_qubit_map[12]),
                cirq.X(fermion_index_to_qubit_map[14]),
            ]
        )
        + ansatz_circuit
    )

    return superposition_circuit, ansatz_circuit


def get_circuits(
    *,
    two_body_params: np.ndarray,
    n_orb: int,
    n_elec: int,
    heuristic_layers: Tuple[lspec.LayerSpec, ...],
) -> Tuple[cirq.Circuit, cirq.Circuit]:
    """A function that runs a specialized method to get the ansatz circuits."""

    if n_orb != n_elec:
        raise ValueError("n_orb must equal n_elec.")

    circ_funcs = {
        2: get_4_qubit_pp_circuits,
        4: get_8_qubit_circuits,
        6: get_12_qubit_circuits,
        8: get_16_qubit_circuits,
    }
    try:
        circ_func = circ_funcs[n_orb]
    except KeyError:
        raise NotImplementedError(f"No circuits for n_orb = {n_orb}")

    return circ_func(
        two_body_params=two_body_params,
        n_elec=n_elec,
        heuristic_layers=heuristic_layers,
    )
