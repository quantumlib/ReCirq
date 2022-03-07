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
"""Circuit optimizers for NISQ CZ circuits."""

import copy
from collections import defaultdict
from typing import cast, Dict, List, Optional, Set, Tuple

import cirq
import numpy as np

UNITARY_ATOL = 1e-8


def _gates_are_close(gate0: cirq.Gate, gate1: cirq.Gate) -> bool:
    """Determine whether two gates' unitaries are equal up to global phase and tolerance."""
    # Sanity check the shapes agree
    if cirq.qid_shape(gate0) != cirq.qid_shape(gate1):
        return False
    # Evaluate unitaries
    return cirq.allclose_up_to_global_phase(
        cirq.unitary(gate0), cirq.unitary(gate1), atol=UNITARY_ATOL
    )


def _qubits_with_hadamard(moment: cirq.Moment) -> Set[cirq.Qid]:
    """Returns the set of qubits in a moment acted on by a hadamard."""
    return {op.qubits[0] for op in moment if _gates_are_close(op.gate, cirq.H)}


def _has_measurement(moment: cirq.Moment) -> bool:
    """Returns True if a moment has a measurement."""
    return any(cirq.is_measurement(op) for op in moment)


def _has_2q_gates(moment: cirq.Moment) -> bool:
    """Returns True if any two-qubit operations with unitaries exist in the moment.

    Note that this excludes measurement."""
    return any(len(op.qubits) == 2 and cirq.has_unitary(op) for op in moment)


def _is_exclusively_1q_gates(moment: cirq.Moment) -> bool:
    """Returns True if all ops in the moment are single-qubit and unitary.

    Note that this excludes measurement."""
    return all(len(op.qubits) == 1 and cirq.has_unitary(op) for op in moment)


def _has_cnot(moment: cirq.Moment) -> bool:
    """Returns True if any operations in the moment are CNOT."""
    return any(_gates_are_close(op.gate, cirq.CNOT) for op in moment)


def _merge_hadamards(moment0: cirq.Moment, moment1: cirq.Moment) -> List[cirq.Moment]:
    """Remove consecutive Hadamards acting on the same qubit.

    Args:
        moment0: First moment of operations
        moment1: Second moment of operations

    Returns:
        One or two  new moments equivalent to (moment0, moment1) but with consecutive Hadamards
            combined/removed. If possible, all operations are combined into the first moment.
    """
    double_hadamard_qubits = _qubits_with_hadamard(moment0) & _qubits_with_hadamard(
        moment1
    )
    moment0_merged_hadamards = moment0.without_operations_touching(
        double_hadamard_qubits
    )
    moment1_merged_hadamards = moment1.without_operations_touching(
        double_hadamard_qubits
    )

    if moment0_merged_hadamards.qubits & moment1_merged_hadamards.qubits:
        return [moment0_merged_hadamards, moment1_merged_hadamards]
    elif moment0_merged_hadamards.qubits | moment1_merged_hadamards.qubits:
        return [moment0_merged_hadamards + moment1_merged_hadamards]
    return []


def _break_up_cnots(
    moment: cirq.Moment,
) -> Tuple[cirq.Moment, cirq.Moment, cirq.Moment]:
    """Break up a moment, replacing CNOTs with CZs and Hadamards.

    Args:
        moment: Moment to check for CNOTs to convert to CZs

    Returns:
        Three moments:
        - Moment of H's for CNOT targets (may be empty)
        - Copy of moment with CNOTs replaced by CZ
        - Moment of H's for CNOT targets (may be empty)
    """
    cnot_control_target_pairs: Set[Tuple[cirq.Qid, cirq.Qid]] = {
        cast(Tuple[cirq.Qid, cirq.Qid], op.qubits)
        for op in moment
        if _gates_are_close(op.gate, cirq.CNOT)
    }
    hadamards = cirq.Moment(cirq.H(pair[1]) for pair in cnot_control_target_pairs)
    ops_besides_cnot = [op for op in moment if not _gates_are_close(op.gate, cirq.CNOT)]
    czs = [cirq.CZ(*pair) for pair in cnot_control_target_pairs]
    moment_with_czs_instead = cirq.Moment(ops_besides_cnot + czs)
    return hadamards, moment_with_czs_instead, hadamards


def _combine_1q_cliffords(*gates: cirq.Gate) -> cirq.Gate:
    """Multiply 1q Clifford unitaries and simplify to a human-friendly representation.

    Args:
        gates: 1q Clifford gates in sequential order of application (index 0 first)

    Returns:
        Single 1q Clifford gate equivalent to gates. We resolve to these gates if possible:
        I, H, X, Y, Z, X**0.5, Y**0.5, S, X**-0.5, Y**-0.5, S**-1
        Otherwise, we return a cirq.SingleQubitCliffordGate.
    """
    unitary = np.asarray(cirq.linalg.dot(*(cirq.unitary(gate) for gate in gates[::-1])))
    clifford = cirq.SingleQubitCliffordGate.from_unitary(unitary)

    clifford_to_friendly_rep: Dict[cirq.SingleQubitCliffordGate, cirq.Gate] = {
        cirq.SingleQubitCliffordGate.I: cirq.I,
        cirq.SingleQubitCliffordGate.H: cirq.H,
        cirq.SingleQubitCliffordGate.X: cirq.X,
        cirq.SingleQubitCliffordGate.Y: cirq.Y,
        cirq.SingleQubitCliffordGate.Z: cirq.Z,
        cirq.SingleQubitCliffordGate.X_sqrt: cirq.X**0.5,
        cirq.SingleQubitCliffordGate.Y_sqrt: cirq.Y**0.5,
        cirq.SingleQubitCliffordGate.Z_sqrt: cirq.S,
        cirq.SingleQubitCliffordGate.X_nsqrt: cirq.X**-0.5,
        cirq.SingleQubitCliffordGate.Y_nsqrt: cirq.Y**-0.5,
        cirq.SingleQubitCliffordGate.Z_nsqrt: cirq.S**-1,
    }
    return clifford_to_friendly_rep.get(clifford, clifford)


def convert_cnot_moments_to_cz_and_simplify_hadamards(
    circuit: cirq.Circuit,
) -> cirq.Circuit:
    """Create a new circuit with CNOT replaced with CZ and simplified H.

    We preserve the moment structure of the CNOTs and insert Hadamards as needed.
    """
    moment_on_deck = cirq.Moment()
    new_circuit = cirq.Circuit()

    for moment in circuit:
        if _has_2q_gates(moment):
            hadamard0, cz, hadamard1 = _break_up_cnots(moment)
            new_circuit += _merge_hadamards(moment_on_deck, hadamard0)
            new_circuit += cz
            moment_on_deck = hadamard1
        elif _is_exclusively_1q_gates(moment):
            merge_moments = _merge_hadamards(moment_on_deck, moment)
            if len(merge_moments) > 1:
                new_circuit += merge_moments[0]
            moment_on_deck = merge_moments[-1] or cirq.Moment()
        else:
            if moment_on_deck:
                new_circuit += moment_on_deck
                moment_on_deck = cirq.Moment()
            new_circuit += moment

    if moment_on_deck:
        new_circuit += moment_on_deck

    return new_circuit


def defer_single_qubit_gates(circuit: cirq.Circuit) -> cirq.Circuit:
    """Create a new circuit where some early single-qubit gates have been pushed later.

    This leaves qubits in the |0> state until right before their first entangling gates.

    This is intended for a moment structure where each moment is exclusively 1q or exclusively 2q
    gates. We do not add any additional moments, and we do not insert 1q gates into any moment
    that contains 2q gates. We stop deferring qubit gates once we encounter a gate on that qubit
    during a moment that contains multi-qubit gates.
    """
    new_circuit = copy.deepcopy(circuit)

    # We can move 1q gates in/out of these moments
    exclusively_1q_moments: List[int] = []
    deferrable_qubits = set(new_circuit.all_qubits())
    qubit_to_deferred_ops: Dict[cirq.Qid, List[cirq.Operation]] = defaultdict(list)

    def resolve_qubit(q: cirq.Qid) -> None:
        """Insert deferred gates into new_circuit and remove q from qubit_to_deferred_gates."""
        for idx, op in enumerate(reversed(qubit_to_deferred_ops[q])):
            insert_idx = (
                exclusively_1q_moments[-1 - idx] + 1
            )  # insert acts just before this index
            new_circuit.insert(insert_idx, op, strategy=cirq.InsertStrategy.INLINE)
        del qubit_to_deferred_ops[q]

    for idx, moment in enumerate(new_circuit):
        if _is_exclusively_1q_gates(moment):
            exclusively_1q_moments.append(idx)
            for op in moment:
                qubit = op.qubits[0]
                if qubit in deferrable_qubits:
                    qubit_to_deferred_ops[qubit].append(op)
            new_circuit[idx] = moment.without_operations_touching(deferrable_qubits)
        else:
            qubits_to_resolve_now = set(
                moment.qubits
            )  # Can't defer further if in this moment
            deferrable_qubits -= qubits_to_resolve_now
            for q in qubits_to_resolve_now:
                resolve_qubit(q)

    # Resolve remaining qubits
    for q in list(qubit_to_deferred_ops):
        resolve_qubit(q)

    # Remove empty moments (kept in thus far to preserve moment indices)
    new_circuit = cirq.Circuit(m for m in new_circuit.moments if m)

    return new_circuit


def propagate_inserted_paulis(
    circuit: cirq.Circuit, *, inserted_pauli_tag: str
) -> Tuple[cirq.Circuit, cirq.PauliString]:
    """Determines the measurements and output qubits flipped by inserted Pauli operations.

    A Pauli operation is inserted if it has the `inserted_tag` tag. The goal of this method is to
    adjust the circuit to undo the functional changes caused by any inserted Paulis by flipping
    certain measurement results.

    Args:
        circuit: The circuit to operate on. Should only contain stabilizer operations.
        inserted_pauli_tag: The tag indicating which operations have been inserted. Should only be
            applied to Pauli operations.

    Returns:
        A (modified_circuit, final_pauli_frame) tuple.
        modified_circuit: A copy of the given circuit where the `inserted_tag` tags have been
            removed and certain measurements have been inverted.
        final_pauli_frame: A Pauli string indicating which qubits at the end of the circuit are
            still bit flipped and/or phase flipped due to the inserted Paulis. Not important for
            circuits that measure all qubits at the end.
    """
    pauli_frame: cirq.PauliString[cirq.Qid] = cirq.PauliString()
    modified_circuit = cirq.Circuit()

    for old_moment in circuit:
        new_moment_ops = []
        for op in old_moment:
            new_op = op
            if inserted_pauli_tag in op.tags:
                # A new Pauli for our Pauli frame.
                pauli_frame *= op
                new_op = op.untagged
                if len(op.tags) > 1:
                    new_tags = (tag for tag in op.tags if tag != inserted_pauli_tag)
                    new_op = new_op.with_tags(*new_tags)

            elif isinstance(op.gate, cirq.MeasurementGate):
                # A measurement that we may need to invert.
                gate = op.gate
                inverted_qubits = {
                    q for q in op.qubits if pauli_frame.get(q) in (cirq.X, cirq.Y)
                }
                if inverted_qubits:
                    new_gate = gate.with_bits_flipped(
                        *(i for i, q in enumerate(op.qubits) if q in inverted_qubits)
                    )
                    new_op = new_gate.on(*op.qubits)

            elif isinstance(op.gate, cirq.ResetChannel):
                # Reset drops part of the Pauli frame.
                pauli_frame *= pauli_frame.get(op.qubits[0], cirq.I).on(op.qubits[0])

            else:
                # This should be a Clifford operation that can transform the Pauli frame.
                try:
                    pauli_frame = pauli_frame.pass_operations_over(
                        [op], after_to_before=True
                    )
                except TypeError as ex:
                    raise NotImplementedError(
                        f"Don't know how to handle {op!r}."
                    ) from ex

            new_moment_ops.append(new_op)

        modified_circuit.append(cirq.Moment(new_moment_ops))

    return modified_circuit, pauli_frame / pauli_frame.coefficient


def insert_echos_on_idle_qubits(
    circuit: cirq.Circuit, echo: cirq.Pauli = cirq.X, resolve_to_hadamard: bool = False
) -> cirq.Circuit:
    """Create a new circuit where we add "echo" gates to qubits during idle moments.

    - We only add "echo" gates once a qubit has done something
        - This is useful if the qubits begin the circuit in |0⟩
    - We only insert "echo" gates into moments that are exclusively 1q gates
    - We resolve any Pauli frame changes in the final exclusively-1q moment
        - If possible, we use standard gates like cirq.H, but in general may use
            cirq.SingleQubitCliffordGate
    - We stop making changes once we reach a moment with any measurements

    Args:
        circuit: Must be a Clifford circuit (to efficiently compute Pauli frame)
        echo: Single-qubit Pauli gate to be inserted
        resolve_to_hadamard: If False, leave potentially arbitrary
            cirq.SingleQubitCliffordGate's in the final exclusively-1q moment.
            If True, resolve this final moment into up to three moments, one with XY
            rotations (converted to H when possible) potentially with Z rotations
            before and after. This has the same experimental runtime (one microwave layer)
            but avoids less-intuitive SingleQubitCliffordGate transformations.

    Returns:
        New optimized circuit
    """
    if not isinstance(echo, cirq.Pauli):
        raise ValueError(f"{echo=} invalid: not a Pauli gate")

    # Figure out how deep to go into circuit
    # (up to final exclusively-1q moment before first measurement)
    new_circuit = copy.deepcopy(circuit)
    final_1q_moment: Optional[int] = None
    for idx, moment in enumerate(new_circuit):
        if _is_exclusively_1q_gates(moment):
            final_1q_moment = idx
        elif _has_measurement(moment):
            break
    if final_1q_moment is None:  # We only want to change exclusively-1q moments
        return new_circuit

    # Insert tagged echo gates into a copy of circuit
    tag = "insert_echos_on_idle_qubits"
    new_circuit = copy.deepcopy(circuit)
    idle_qubits: Set[cirq.Qid] = set()

    for idx, moment in enumerate(new_circuit[:final_1q_moment]):
        if _is_exclusively_1q_gates(moment):  # Add echos
            for q in idle_qubits - moment.qubits:
                new_circuit.insert(
                    idx + 1, echo(q).with_tags(tag), strategy=cirq.InsertStrategy.INLINE
                )
        idle_qubits |= moment.qubits  # Eligible for subsequent echos

    # Figure out the overall Pauli frame heading into the final 1q gate moment
    untagged_partial_circuit, pauli_frame = propagate_inserted_paulis(
        new_circuit[:final_1q_moment], inserted_pauli_tag=tag
    )
    new_circuit[:final_1q_moment] = untagged_partial_circuit

    # Combine pauli_frame corrections into the final 1q gate moment
    uncorrected_moment = new_circuit[final_1q_moment]
    combined_moment = cirq.Moment()
    relevant_qubits = uncorrected_moment.qubits | set(pauli_frame.qubits)
    for q in relevant_qubits:
        pauli = pauli_frame.get(q, cirq.I)
        op = uncorrected_moment.operation_at(q) or cirq.I(q)
        combined_moment += _combine_1q_cliffords(pauli, op.gate).on(q)

    if resolve_to_hadamard:  # Split into up to 3 moments ~ -Z--H--Z-
        resolved_combined_circuit = resolve_moment_to_hadamard(combined_moment)
        before = new_circuit[:final_1q_moment]
        after = new_circuit[final_1q_moment + 1 :]
        new_circuit = before + resolved_combined_circuit + after
    else:
        new_circuit[final_1q_moment] = combined_moment

    return new_circuit


def resolve_moment_to_hadamard(moment: cirq.Moment) -> cirq.Circuit:
    """Resolve a moment of SingleQubitClifford into Hadamards where possible.

    If the gate has a pi/2 XY rotation, we convert that to H surrounded by the necessary
    Z rotations.

    Args:
        moment: Moment of single-qubit Clifford gates

    Returns:
        Circuit with up to three moments. Only one moment will contain XY rotations
        (including H), and it may have a moment of Z rotations before and/or after.
    """
    # idx 0: Z, idx 1: XY rotations (including H), idx 2: Z
    new_ops: List[List[cirq.Operation]] = [[], [], []]

    for op in moment:
        replacements = resolve_gate_to_hadamard(op.gate)
        for idx, gate in enumerate(replacements):
            if gate != cirq.I:
                new_ops[idx].append(gate.on(op.qubits[0]))

    return cirq.Circuit(cirq.Moment(*ops) for ops in new_ops if ops)


def resolve_gate_to_hadamard(gate: cirq.Gate) -> Tuple[cirq.Gate, cirq.Gate, cirq.Gate]:
    """Resolve a single-qubit Clifford gate into Z rotations around an H, if possible.

    We always return three gates, with the middle gate (index 1) being the only one
    that can have an XY rotation. The outer gates (index 0, 2) are Z rotations or I.

    If the gate has a pi/2 XY rotation, we convert that to H surrounded by the necessary
    Z rotations. Otherwise, we simply return (I, gate, I).
    """
    xz_gate = cirq.PhasedXZGate.from_matrix(cirq.unitary(gate))

    # Only handle pi/2 XY rotations; return others unchanged
    if not np.isclose(abs(xz_gate.x_exponent), 0.5):
        return cirq.I, gate, cirq.I

    # Massage into canonical Hadamard decomposition ──H── = ── Z^-1/2 ── X^-1/2 ── Z^1/2 ── Z ──
    # Transform so x_exponent = -0.5
    if np.isclose(float(xz_gate.x_exponent), 0.5):
        xz_gate = cirq.PhasedXZGate(
            x_exponent=-0.5,
            z_exponent=xz_gate.z_exponent,
            axis_phase_exponent=xz_gate.axis_phase_exponent + 1,
        )

    # Calculate Z rotations to align with Hadamard
    # Choose z_exp_before and z_exp_after to make these decompositions equal
    #
    # 1. Gate's canonical PhasedXZ decomposition:
    # ──gate── = ── Z^-a ── X^-1/2 ── Z^a ── Z^z
    #
    # 2. Gate's decomposition into ZHZ
    # ──gate── = ── Z^z_exp_before ── H ── Z^z_exp_after ──
    #          = ── Z^z_exp_before ── Z^-1/2 ── X^-1/2 ── Z^1/2 ── Z ── Z^z_exp_after ──
    #
    # Before: -a = z_exp_before - 0.5
    # After: a + z = 1.5 + z_exp_after
    z_exp_before = 0.5 - xz_gate.axis_phase_exponent
    z_exp_after = xz_gate.axis_phase_exponent + xz_gate.z_exponent - 1.5
    gate_before = _combine_1q_cliffords(cirq.Z**z_exp_before)
    gate_after = _combine_1q_cliffords(cirq.Z**z_exp_after)

    return gate_before, cirq.H, gate_after
