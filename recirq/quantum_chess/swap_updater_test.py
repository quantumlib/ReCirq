import pytest
import cirq
import cirq_google as cg

from recirq.quantum_chess.swap_updater import SwapUpdater, generate_decomposed_swap
import recirq.quantum_chess.quantum_moves as qm

# Logical qubits q0 - q5.
q = list(cirq.NamedQubit(f"q{i}") for i in range(6))

# Qubits corresponding to figure 9 in
# https://ieeexplore.ieee.org/abstract/document/8976109.
# Coupling graph looks something like:
#   Q0 = Q1 = Q2
#   ||   ||   ||
#   Q3 = Q4 = Q5
FIGURE_9A_PHYSICAL_QUBITS = cirq.GridQubit.rect(2, 3)
# Circuit from figure 9a in
# https://ieeexplore.ieee.org/abstract/document/8976109.
FIGURE_9A_CIRCUIT = cirq.Circuit(
    cirq.CNOT(q[0], q[2]),
    cirq.CNOT(q[5], q[2]),
    cirq.CNOT(q[0], q[5]),
    cirq.CNOT(q[4], q[0]),
    cirq.CNOT(q[0], q[3]),
    cirq.CNOT(q[5], q[0]),
    cirq.CNOT(q[3], q[1]),
)


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
    updater = SwapUpdater(
        FIGURE_9A_CIRCUIT, Q, initial_mapping, lambda q1, q2: [cirq.SWAP(q1, q2)]
    )

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
        cirq.CNOT(Q[5], Q[4]),
        cirq.CNOT(Q[3], Q[0]),
    }


def test_example_9():
    Q = FIGURE_9A_PHYSICAL_QUBITS
    initial_mapping = dict(zip(q, Q))
    updated_circuit = cirq.Circuit(
        SwapUpdater(
            FIGURE_9A_CIRCUIT, Q, initial_mapping, lambda q1, q2: [cirq.SWAP(q1, q2)]
        ).add_swaps()
    )
    assert updated_circuit == cirq.Circuit(
        cirq.SWAP(Q[0], Q[1]),
        cirq.CNOT(Q[1], Q[2]),
        cirq.CNOT(Q[5], Q[2]),
        cirq.SWAP(Q[1], Q[4]),
        cirq.CNOT(Q[4], Q[5]),
        cirq.CNOT(Q[1], Q[4]),
        cirq.CNOT(Q[4], Q[3]),
        cirq.CNOT(Q[5], Q[4]),
        cirq.CNOT(Q[3], Q[0]),
    )


def test_pentagonal_split_and_merge():
    grid_2x3 = cirq.GridQubit.rect(2, 3, 4, 4)
    logical_qubits = list(
        cirq.NamedQubit(f"{x}{i}") for x in ("a", "b") for i in range(3)
    )
    initial_mapping = dict(zip(logical_qubits, grid_2x3))
    a1, a2, a3, b1, b2, b3 = logical_qubits
    logical_circuit = cirq.Circuit(
        qm.normal_move(a1, b1), qm.normal_move(a2, a3), qm.merge_move(a3, b1, b3)
    )

    updater = SwapUpdater(logical_circuit, grid_2x3, initial_mapping)
    updated_circuit = cirq.Circuit(updater.add_swaps())

    # Whereas the original circuit's initial mapping was not valid due to
    # adjacency constraints, the updated circuit is valid.
    device = cg.Sycamore
    with pytest.raises(ValueError):
        device.validate_circuit(
            logical_circuit.transform_qubits(lambda q: initial_mapping.get(q))
        )
    device.validate_circuit(updated_circuit)


def test_already_optimal():
    grid_2x3 = cirq.GridQubit.rect(2, 3)
    logical_qubits = list(
        cirq.NamedQubit(f"{x}{i}") for x in ("a", "b") for i in range(3)
    )
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
        SwapUpdater(circuit, grid_2x3, initial_mapping).add_swaps()
    )
    # The circuit was already optimal, so we didn't need to add any extra
    # operations, just map the logical qubits to physical qubits.
    assert circuit.transform_qubits(lambda q: initial_mapping.get(q)) == updated_circuit


def test_decomposed_swaps():
    Q = FIGURE_9A_PHYSICAL_QUBITS
    initial_mapping = dict(zip(q, Q))
    updater = SwapUpdater(
        FIGURE_9A_CIRCUIT, Q, initial_mapping, generate_decomposed_swap
    )
    # First iteration adds a swap between Q0 and Q1.
    # generate_decomposed_swap decomposes that into sqrt-iswap operations.
    assert list(updater.update_iteration()) == list(
        generate_decomposed_swap(Q[0], Q[1])
    )

    # Whatever the decomposed operations are, they'd better be equivalent to a
    # SWAP.
    swap_unitary = cirq.unitary(cirq.Circuit(cirq.SWAP(Q[0], Q[1])))
    generated_unitary = cirq.unitary(cirq.Circuit(generate_decomposed_swap(Q[0], Q[1])))
    cirq.testing.assert_allclose_up_to_global_phase(
        swap_unitary, generated_unitary, atol=1e-8
    )


def test_holes_in_device_graph():
    grid_2x3 = cirq.GridQubit.rect(2, 3, 4, 4)
    logical_qubits = list(
        cirq.NamedQubit(f"{x}{i}") for x in ("a", "b") for i in range(3)
    )
    initial_mapping = dict(zip(logical_qubits, grid_2x3))
    a1, a2, a3, b1, b2, b3 = logical_qubits
    # Circuit has a gate that operate on (b1, b3) qubits.
    # The best way to add swaps would be between (b1, b2) and (b2, b3), but
    # we aren't allowed to use b2, so we're forced to route swaps around it.
    circuit = cirq.Circuit(cirq.ISWAP(b1, b3) ** 0.5)
    allowed_qubits = set(grid_2x3) - {initial_mapping[b2]}

    updated_circuit = cirq.Circuit(
        SwapUpdater(circuit, allowed_qubits, initial_mapping).add_swaps()
    )
    cg.Sycamore.validate_circuit(updated_circuit)
    for op in updated_circuit.all_operations():
        assert initial_mapping[b2] not in op.qubits


def test_bad_initial_mapping():
    grid_2x3 = cirq.GridQubit.rect(2, 3)
    logical_qubits = list(
        cirq.NamedQubit(f"{x}{i}") for x in ("a", "b") for i in range(3)
    )
    initial_mapping = dict(zip(logical_qubits, grid_2x3))
    a1, a2, a3, b1, b2, b3 = logical_qubits
    # Initial mapping puts a1 on a qubit that we're not allowed to use, but the
    # circuit uses a1.
    # That should cause the swap updater to throw.
    circuit = cirq.Circuit(cirq.ISWAP(a1, a3))
    allowed_qubits = set(grid_2x3) - {initial_mapping[a1]}
    with pytest.raises(KeyError):
        print(
            cirq.Circuit(
                SwapUpdater(circuit, allowed_qubits, initial_mapping).add_swaps()
            )
        )


def test_disconnected_components():
    grid_2x3 = cirq.GridQubit.rect(2, 3)
    logical_qubits = list(
        cirq.NamedQubit(f"{x}{i}") for x in ("a", "b") for i in range(3)
    )
    initial_mapping = dict(zip(logical_qubits, grid_2x3))
    a1, a2, a3, b1, b2, b3 = logical_qubits
    # a2 and b2 are disallowed. That splits the device graph into 2 disconnected
    # components and makes a3 unreachable from a1.
    circuit = cirq.Circuit(cirq.ISWAP(a1, a3))
    allowed_qubits = set(grid_2x3) - {initial_mapping[a2], initial_mapping[b2]}
    with pytest.raises(KeyError):
        print(
            cirq.Circuit(
                SwapUpdater(circuit, allowed_qubits, initial_mapping).add_swaps()
            )
        )


def test_termination_in_local_minimum():
    grid_2x5 = cirq.GridQubit.rect(2, 5)
    q = list(cirq.NamedQubit(f"q{i}") for i in range(6))
    # The initial mapping looks like:
    #   _|_0_|_1_|_2_|_3_|_4_|
    #   0|q0 |   |   |   |q4 |
    #   1|q1 |q2 |   |q3 |q5 |
    initial_mapping = {
        q[0]: cirq.GridQubit(0, 0),
        q[1]: cirq.GridQubit(1, 0),
        q[2]: cirq.GridQubit(1, 1),
        q[3]: cirq.GridQubit(1, 3),
        q[4]: cirq.GridQubit(0, 4),
        q[5]: cirq.GridQubit(1, 4),
    }
    # Here's the idea:
    #   * there are two "clumps" of qubits: (q0,q1,q2) and (q3,q4,q5)
    #   * the active gate(s) span the two clumps
    #   * there are also gates on qubits within clumps
    #   * intra-clump gate cost contribution outweighs inter-clump gate cost
    # In that case, we need to swap qubits away from their respective clumps in
    # order to progress beyond any of the active gates. But we never will
    # because doing so would increase the overall cost due to the intra-clump
    # contributions. In fact, no greedy algorithm would be able to make progress
    # in this case.
    circuit = cirq.Circuit()
    # Cross-clump active gates
    circuit.append(
        [cirq.CNOT(q[0], q[3]), cirq.CNOT(q[1], q[4]), cirq.CNOT(q[2], q[5])]
    )
    # Intra-clump q0,q1,q2
    circuit.append(
        [cirq.CNOT(q[0], q[1]), cirq.CNOT(q[0], q[2]), cirq.CNOT(q[1], q[2])]
    )
    # Intra-clump q3,q4,q5
    circuit.append(
        [cirq.CNOT(q[3], q[4]), cirq.CNOT(q[3], q[5]), cirq.CNOT(q[4], q[5])]
    )

    updater = SwapUpdater(
        circuit, grid_2x5, initial_mapping, lambda q1, q2: [cirq.SWAP(q1, q2)]
    )
    # Iterate until the SwapUpdater is finished or an assertion fails, keeping
    # track of the ops generated by the previous iteration.
    prev_it = list(updater.update_iteration())
    while not updater.dlists.all_empty():
        cur_it = list(updater.update_iteration())

        # If the current iteration adds a SWAP, it better not be the same SWAP
        # as the previous iteration...
        # If we pick the same SWAP twice in a row, then we're going in a loop
        # forever without making any progress!
        def _is_swap(ops):
            return len(ops) == 1 and ops[0] == cirq.SWAP(*ops[0].qubits)

        if _is_swap(prev_it) and _is_swap(cur_it):
            assert set(prev_it[0].qubits) != set(cur_it[0].qubits)
        prev_it = cur_it
