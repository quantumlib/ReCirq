import pytest
import cirq

import recirq.quantum_chess.mcpe_utils as mcpe
from recirq.quantum_chess.swap_updater import SwapUpdater, generate_decomposed_swap
import recirq.quantum_chess.quantum_moves as qm

# Logical qubits q0 - q5.
q = list(cirq.NamedQubit(f'q{i}') for i in range(6))

# Qubits corresponding to figure 9 in
# https://ieeexplore.ieee.org/abstract/document/8976109.
# Coupling graph looks something like:
#   Q0 = Q1 = Q2
#   ||   ||   ||
#   Q3 = Q4 = Q5
FIGURE_9A_PHYSICAL_QUBITS = cirq.GridQubit.rect(2, 3)
# Circuit from figure 9a in
# https://ieeexplore.ieee.org/abstract/document/8976109.
FIGURE_9A_CIRCUIT = cirq.Circuit(cirq.CNOT(q[0], q[2]), cirq.CNOT(q[5], q[2]),
                                 cirq.CNOT(q[0], q[5]), cirq.CNOT(q[4], q[0]),
                                 cirq.CNOT(q[0], q[3]), cirq.CNOT(q[5], q[0]),
                                 cirq.CNOT(q[3], q[1]))


def test_example_9_candidate_swaps():
    Q = FIGURE_9A_PHYSICAL_QUBITS
    initial_mapping = dict(zip(q, Q))
    updater = SwapUpdater(FIGURE_9A_CIRCUIT, Q, initial_mapping)

    gate = cirq.CNOT(Q[0], Q[2])
    candidates = list(updater.generate_candidate_swaps([gate]))
    # The swaps connected to either q0 or q2 are:
    #   (Q0, Q1), (Q0, Q3), (Q2, Q1), (Q2, Q5)
    # But swapping (Q0, Q3) or (Q2, Q5) would negatively impact the distance
    # between q0 and q2, so those swaps are discarded.
    assert candidates == [(Q[0], Q[1]), (Q[2], Q[1])]


def test_example_9_iterations():
    Q = FIGURE_9A_PHYSICAL_QUBITS
    initial_mapping = dict(zip(q, Q))
    updater = SwapUpdater(FIGURE_9A_CIRCUIT, Q, initial_mapping,
                          lambda q1, q2: [cirq.SWAP(q1, q2)])

    # First iteration adds a swap between Q0 and Q1.
    assert list(updater.update_iteration()) == [cirq.SWAP(Q[0], Q[1])]
    # Next two iterations add the active gates as-is.
    assert list(updater.update_iteration()) == [cirq.CNOT(Q[1], Q[2])]
    assert list(updater.update_iteration()) == [cirq.CNOT(Q[5], Q[2])]
    # Next iteration adds a swap between Q1 and Q4.
    assert list(updater.update_iteration()) == [cirq.SWAP(Q[1], Q[4])]
    # Remaining gates are added as-is.
    assert list(updater.update_iteration()) == [cirq.CNOT(Q[4], Q[5])]
    assert list(updater.update_iteration()) == [cirq.CNOT(Q[1], Q[4])]
    assert list(updater.update_iteration()) == [cirq.CNOT(Q[4], Q[3])]
    # The final two gates are added in the same iteration, since they operate on
    # mutually exclusive qubits and are both simultaneously active.
    assert set(updater.update_iteration()) == {
        cirq.CNOT(Q[5], Q[4]), cirq.CNOT(Q[3], Q[0])
    }


def test_example_9():
    Q = FIGURE_9A_PHYSICAL_QUBITS
    initial_mapping = dict(zip(q, Q))
    updater = SwapUpdater(FIGURE_9A_CIRCUIT, Q, initial_mapping,
                          lambda q1, q2: [cirq.SWAP(q1, q2)])
    updated_circuit = cirq.Circuit(
        SwapUpdater(FIGURE_9A_CIRCUIT, Q, initial_mapping,
                    lambda q1, q2: [cirq.SWAP(q1, q2)]).add_swaps())
    assert updated_circuit == cirq.Circuit(cirq.SWAP(Q[0], Q[1]),
                                           cirq.CNOT(Q[1], Q[2]),
                                           cirq.CNOT(Q[5], Q[2]),
                                           cirq.SWAP(Q[1], Q[4]),
                                           cirq.CNOT(Q[4], Q[5]),
                                           cirq.CNOT(Q[1], Q[4]),
                                           cirq.CNOT(Q[4], Q[3]),
                                           cirq.CNOT(Q[5], Q[4]),
                                           cirq.CNOT(Q[3], Q[0]))


def test_pentagonal_split_and_merge():
    grid_2x3 = cirq.GridQubit.rect(2, 3, 4, 4)
    logical_qubits = list(
        cirq.NamedQubit(f'{x}{i}') for x in ('a', 'b') for i in range(3))
    initial_mapping = dict(zip(logical_qubits, grid_2x3))
    a1, a2, a3, b1, b2, b3 = logical_qubits
    logical_circuit = cirq.Circuit(qm.normal_move(a1, b1),
                                   qm.normal_move(a2, a3),
                                   qm.merge_move(a3, b1, b3))

    updater = SwapUpdater(logical_circuit, grid_2x3, initial_mapping)
    updated_circuit = cirq.Circuit(updater.add_swaps())

    # Whereas the original circuit's initial mapping was not valid due to
    # adjacency constraints, the updated circuit is valid.
    device = cirq.google.Sycamore
    with pytest.raises(ValueError):
        device.validate_circuit(
            logical_circuit.transform_qubits(lambda q: initial_mapping.get(q)))
    device.validate_circuit(updated_circuit)


def test_already_optimal():
    grid_2x3 = cirq.GridQubit.rect(2, 3)
    logical_qubits = list(
        cirq.NamedQubit(f'{x}{i}') for x in ('a', 'b') for i in range(3))
    initial_mapping = dict(zip(logical_qubits, grid_2x3))
    a1, a2, a3, b1, b2, b3 = logical_qubits
    # Circuit has gates that already operate only on adjacent qubits.
    circuit = cirq.Circuit(
        cirq.ISWAP(a1, b1),
        cirq.ISWAP(a2, a3),
        cirq.ISWAP(b1, b2),
        cirq.ISWAP(a3, b3),
        cirq.ISWAP(b2, b3),
    )
    updated_circuit = cirq.Circuit(
        SwapUpdater(circuit, grid_2x3, initial_mapping).add_swaps())
    # The circuit was already optimal, so we didn't need to add any extra
    # operations, just map the logical qubits to physical qubits.
    assert circuit.transform_qubits(
        lambda q: initial_mapping.get(q)) == updated_circuit


def test_decomposed_swaps():
    Q = FIGURE_9A_PHYSICAL_QUBITS
    initial_mapping = dict(zip(q, Q))
    updater = SwapUpdater(FIGURE_9A_CIRCUIT, Q, initial_mapping,
                          generate_decomposed_swap)
    # First iteration adds a swap between Q0 and Q1.
    # generate_decomposed_swap decomposes that into sqrt-iswap operations.
    assert list(updater.update_iteration()) == list(
        generate_decomposed_swap(Q[0], Q[1]))

    # Whatever the decomposed operations are, they'd better be equivalent to a
    # SWAP.
    swap_unitary = cirq.unitary(cirq.Circuit(cirq.SWAP(Q[0], Q[1])))
    generated_unitary = cirq.unitary(
        cirq.Circuit(generate_decomposed_swap(Q[0], Q[1])))
    cirq.testing.assert_allclose_up_to_global_phase(swap_unitary,
                                                    generated_unitary,
                                                    atol=1e-8)
