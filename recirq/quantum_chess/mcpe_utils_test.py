import pytest
import cirq
import numpy as np

import recirq.quantum_chess.mcpe_utils as mcpe


def test_manhattan_distance():
    assert mcpe.manhattan_dist(cirq.GridQubit(0, 0), cirq.GridQubit(0, 0)) == 0
    assert mcpe.manhattan_dist(cirq.GridQubit(1, 2), cirq.GridQubit(1, 2)) == 0
    assert mcpe.manhattan_dist(cirq.GridQubit(1, 2), cirq.GridQubit(3, 4)) == 4
    assert mcpe.manhattan_dist(cirq.GridQubit(3, 4), cirq.GridQubit(1, 2)) == 4
    assert mcpe.manhattan_dist(cirq.GridQubit(-1, 2), cirq.GridQubit(3, -4)) == 10


def test_swap_map_fn():
    x, y, z = (cirq.NamedQubit(f"q{i}") for i in range(3))
    swap = mcpe.swap_map_fn(x, y)
    assert swap(x) == y
    assert swap(y) == x
    assert swap(z) == z
    assert cirq.Circuit(
        cirq.ISWAP(x, z), cirq.ISWAP(y, z), cirq.ISWAP(x, y)
    ).transform_qubits(swap) == cirq.Circuit(
        cirq.ISWAP(y, z), cirq.ISWAP(x, z), cirq.ISWAP(y, x)
    )


def test_effect_of_swap():
    a1, a2, a3, b1, b2, b3 = cirq.GridQubit.rect(2, 3)
    # If there's a gate operating on (a1, a3), then swapping a1 and a2 will
    # bring the gate's qubits closer together by 1.
    assert mcpe.effect_of_swap((a1, a2), (a1, a3)) == 1
    # In reverse, a gate operating on (a2, a3) will get worse by 1 when swapping
    # (a1, a2).
    assert mcpe.effect_of_swap((a1, a2), (a2, a3)) == -1
    # If the qubits to be swapped are completely independent of the gate's
    # qubits, then there's no effect on the gate.
    assert mcpe.effect_of_swap((a1, a2), (b1, b2)) == 0
    # We can also measure the effect of swapping non-adjacent qubits (although
    # we would never be able to do this with a real SWAP gate).
    assert mcpe.effect_of_swap((a1, a3), (a1, b3)) == 2


def test_distance_fn():
    a1, a2, a3, b1, b2, b3 = cirq.GridQubit.rect(2, 3)

    # A gate operating on (a1, a3) will be improved by swapping a1 and a2, but
    # by how much depends on the distance function used.
    assert mcpe.effect_of_swap((a1, a2), (a1, a3), mcpe.manhattan_dist) == 1
    double_manhattan = lambda q1, q2: 2 * mcpe.manhattan_dist(q1, q2)
    assert mcpe.effect_of_swap((a1, a2), (a1, a3), double_manhattan) == 2

    def euclidean_dist(q1, q2):
        return ((q1.row - q2.row) ** 2 + (q1.col - q2.col) ** 2) ** 0.5

    # Before, the gate qubits (a1, b2) are sqrt(2) units apart from each other
    # by euclidean distance.
    # After swapping a1 and a2, they'd only be 1 unit apart, so things improve
    # by sqrt(2) - 1.
    effect = mcpe.effect_of_swap((a1, a2), (a1, b2), euclidean_dist)
    np.testing.assert_allclose(effect, 2**0.5 - 1)


def test_peek():
    x, y, z = (cirq.NamedQubit(f"q{i}") for i in range(3))
    g = [cirq.ISWAP(x, y), cirq.ISWAP(x, z), cirq.ISWAP(y, z)]
    dlists = mcpe.DependencyLists(cirq.Circuit(g))
    assert dlists.peek_front(x) == g[0]
    assert dlists.peek_front(y) == g[0]
    assert dlists.peek_front(z) == g[1]


def test_pop():
    x, y, z = (cirq.NamedQubit(f"q{i}") for i in range(3))
    g = [cirq.ISWAP(x, y), cirq.ISWAP(x, z), cirq.ISWAP(y, z)]
    dlists = mcpe.DependencyLists(cirq.Circuit(g))

    assert dlists.peek_front(x) == g[0]
    assert dlists.peek_front(y) == g[0]
    assert dlists.peek_front(z) == g[1]
    dlists.pop_active(g[0])
    assert dlists.peek_front(x) == g[1]
    assert dlists.peek_front(y) == g[2]
    assert dlists.peek_front(z) == g[1]


def test_pop_non_active():
    x, y, z = (cirq.NamedQubit(f"q{i}") for i in range(3))
    g = [cirq.ISWAP(x, y), cirq.ISWAP(x, z), cirq.ISWAP(y, z)]
    dlists = mcpe.DependencyLists(cirq.Circuit(g))

    with pytest.raises(KeyError):
        # Gate is in the dependency lists, but isn't currently active.
        dlists.pop_active(g[-1])
    with pytest.raises(KeyError):
        # Gate is not even in the dependency lists.
        dlists.pop_active(cirq.CNOT(x, y))


def test_empty():
    x, y, z = (cirq.NamedQubit(f"q{i}") for i in range(3))
    dlists = mcpe.DependencyLists(
        cirq.Circuit(cirq.ISWAP(x, y), cirq.ISWAP(x, z), cirq.ISWAP(y, z))
    )

    assert not dlists.empty(x)
    dlists.pop_active(dlists.peek_front(x))
    assert not dlists.empty(x)
    dlists.pop_active(dlists.peek_front(x))
    assert dlists.empty(x)

    assert not dlists.all_empty()
    dlists.pop_active(dlists.peek_front(y))
    assert dlists.all_empty()


def test_active_gates():
    w, x, y, z = (cirq.NamedQubit(f"q{i}") for i in range(4))
    dlists = mcpe.DependencyLists(
        cirq.Circuit(cirq.ISWAP(x, y), cirq.ISWAP(y, z), cirq.X(w))
    )

    assert dlists.active_gates == {cirq.ISWAP(x, y), cirq.X(w)}


def test_physical_mapping():
    q = list(cirq.NamedQubit(f"q{i}") for i in range(6))
    Q = list(cirq.GridQubit(row, col) for row in range(2) for col in range(3))
    mapping = mcpe.QubitMapping(dict(zip(q, Q)))
    assert list(map(mapping.physical, q)) == Q
    assert cirq.ISWAP(q[1], q[5]).transform_qubits(mapping.physical) == cirq.ISWAP(
        Q[1], Q[5]
    )


def test_swap():
    q = list(cirq.NamedQubit(f"q{i}") for i in range(6))
    Q = list(cirq.GridQubit(row, col) for row in range(2) for col in range(3))
    mapping = mcpe.QubitMapping(dict(zip(q, Q)))

    mapping.swap_physical(Q[0], Q[1])
    g = cirq.CNOT(q[0], q[2])
    assert g.transform_qubits(mapping.physical) == cirq.CNOT(Q[1], Q[2])

    mapping.swap_physical(Q[2], Q[3])
    mapping.swap_physical(Q[1], Q[4])
    assert g.transform_qubits(mapping.physical) == cirq.CNOT(Q[4], Q[3])


def test_mcpe_example_8():
    # This test is example 8 from the circuit in figure 9 of
    # https://ieeexplore.ieee.org/abstract/document/8976109.
    q = list(cirq.NamedQubit(f"q{i}") for i in range(6))
    Q = list(cirq.GridQubit(row, col) for row in range(2) for col in range(3))
    mapping = mcpe.QubitMapping(dict(zip(q, Q)))
    dlists = mcpe.DependencyLists(
        cirq.Circuit(
            cirq.CNOT(q[0], q[2]),
            cirq.CNOT(q[5], q[2]),
            cirq.CNOT(q[0], q[5]),
            cirq.CNOT(q[4], q[0]),
            cirq.CNOT(q[0], q[3]),
            cirq.CNOT(q[5], q[0]),
            cirq.CNOT(q[3], q[1]),
        )
    )

    assert dlists.maximum_consecutive_positive_effect(Q[0], Q[1], mapping) == 4


def test_mcpe_example_9():
    # This test is example 9 from the circuit in figures 9 and 10 of
    # https://ieeexplore.ieee.org/abstract/document/8976109.
    q = list(cirq.NamedQubit(f"q{i}") for i in range(6))
    Q = list(cirq.GridQubit(row, col) for row in range(2) for col in range(3))
    mapping = mcpe.QubitMapping(dict(zip(q, Q)))
    dlists = mcpe.DependencyLists(
        cirq.Circuit(
            cirq.CNOT(q[0], q[2]),
            cirq.CNOT(q[5], q[2]),
            cirq.CNOT(q[0], q[5]),
            cirq.CNOT(q[4], q[0]),
            cirq.CNOT(q[0], q[3]),
            cirq.CNOT(q[5], q[0]),
            cirq.CNOT(q[3], q[1]),
        )
    )

    # At first CNOT(q0, q2) is the active gate.
    assert dlists.active_gates == {cirq.CNOT(q[0], q[2])}
    # The swaps connected to either q0 or q2 to consider are:
    #   (Q0, Q1), (Q0, Q3), (Q1, Q2), (Q2, Q5)
    # Of these, (Q0, Q3) and (Q2, Q5) can be discarded because they would
    # negatively impact the active CNOT(q0, q2) gate.
    assert mcpe.effect_of_swap((Q[0], Q[3]), (Q[0], Q[2])) < 0
    assert mcpe.effect_of_swap((Q[2], Q[5]), (Q[0], Q[2])) < 0
    # The remaining candidate swaps are: (Q0, Q1) and (Q1, Q2)
    # (Q0, Q1) has a higher MCPE, so it looks better to apply that one.
    assert dlists.maximum_consecutive_positive_effect(Q[0], Q[1], mapping) == 4
    assert dlists.maximum_consecutive_positive_effect(Q[1], Q[2], mapping) == 1
    mapping.swap_physical(Q[0], Q[1])

    # The swap-update algorithm would now advance beyond the front-most gates that
    # now satisfy adjacency constraints after the swap -- the CNOT(q0, q2) and
    # CNOT(q5, q2)
    assert dlists.active_gates == {cirq.CNOT(q[0], q[2])}
    dlists.pop_active(dlists.peek_front(q[0]))
    assert dlists.active_gates == {cirq.CNOT(q[5], q[2])}
    dlists.pop_active(dlists.peek_front(q[5]))

    # Now the active gate is g2 (which is CNOT(q0, q5))
    assert dlists.active_gates == {cirq.CNOT(q[0], q[5])}
    # For this active gate, the swaps to consider are:
    #   (Q0, Q1), (Q1, Q2), (Q1, Q4), (Q2, Q5), (Q4, Q5)
    # (Q0, Q1) can be discarded because it negatively impacts the active gate.
    assert mcpe.effect_of_swap((Q[0], Q[1]), (Q[1], Q[5])) < 0
    # Of the remaining candidate swaps, (Q0, Q4) has the highest MCPE.
    assert dlists.maximum_consecutive_positive_effect(Q[1], Q[2], mapping) == 1
    assert dlists.maximum_consecutive_positive_effect(Q[1], Q[4], mapping) == 3
    assert dlists.maximum_consecutive_positive_effect(Q[2], Q[5], mapping) == 2
    assert dlists.maximum_consecutive_positive_effect(Q[4], Q[5], mapping) == 2
