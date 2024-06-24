from typing import Dict, Tuple

import cirq

from recirq.qcqmc.fermion_mode import FermionicMode


def get_qubits_a_b(*, n_orb: int) -> Tuple[cirq.GridQubit, ...]:
    """Get grid alpha/beta grid qubits in ascending order.

    Args:
        n_orb: The number of spatial orbitals.

    This ordering creates qubits to facilitate a Jordan Wigner string
    threading through a row of alpha orbitals in ascending order followed by a
    row of beta orbitals in ascending order.
    """
    return tuple(
        [cirq.GridQubit(0, i) for i in range(n_orb)]
        + [cirq.GridQubit(1, i) for i in range(n_orb)]
    )


def get_qubits_a_b_reversed(*, n_orb: int) -> Tuple[cirq.GridQubit, ...]:
    """Get grid quibts with correct spin ordering.

    This ordering creates qubits to facilitate operations that need a linearly
    connected array of qubits with the order threading through a row of alpha
    orbitals in ascending order followed by a row of beta orbitals in
    descending order.

    Args:
        n_orb: The number of spatial orbitals.
    """
    return tuple(
        [cirq.GridQubit(0, i) for i in range(n_orb)]
        + [cirq.GridQubit(1, i) for i in reversed(range(n_orb))]
    )


def get_4_qubit_fermion_qubit_map() -> Dict[int, cirq.GridQubit]:
    """A helper function that provides the fermion qubit map for 4 qubits.

    We map the fermionic orbitals to grid qubits like so:
    3 1
    2 0
    """
    fermion_index_to_qubit_map = {
        2: cirq.GridQubit(0, 0),
        3: cirq.GridQubit(1, 0),
        0: cirq.GridQubit(0, 1),
        1: cirq.GridQubit(1, 1),
    }

    return fermion_index_to_qubit_map


def get_8_qubit_fermion_qubit_map():
    """A helper function that provides the fermion qubit map for 8 qubits.

    We map the fermionic orbitals to grid qubits like so:
    3 5 1 7
    2 4 0 6
    """

    # Linear connectivity is fine.
    # This ordering is dictated by the way we specify perfect pairing (i)
    # Elsewhere we generate the perfect pairing parameters using a specific
    # convention for how we index the FermionOperators (which is itself)
    # partly dictated by the OpenFermion conventions. Here we choose a
    # mapping between the indices of these FermionOperators and the qubits in our
    # grid that allows for the perfect pairing pairs to be in squares of four.
    fermion_index_to_qubit_map = {
        2: cirq.GridQubit(0, 0),
        3: cirq.GridQubit(1, 0),
        4: cirq.GridQubit(0, 1),
        5: cirq.GridQubit(1, 1),
        0: cirq.GridQubit(0, 2),
        1: cirq.GridQubit(1, 2),
        6: cirq.GridQubit(0, 3),
        7: cirq.GridQubit(1, 3),
    }

    return fermion_index_to_qubit_map


def get_12_qubit_fermion_qubit_map():
    """A helper function that provides the fermion qubit map for 12 qubits.

    We map the fermionic orbitals to grid qubits like so:
    5 7 3 9 1 11
    4 6 2 8 0 10
    """

    # Linear connectivity is fine.
    # This ordering is dictated by the way we specify perfect pairing (i)
    # Elsewhere we generate the perfect pairing parameters using a specific
    # convention for how we index the FermionOperators (which is itself)
    # partly dictated by the OpenFermion conventions. Here we choose a
    # mapping between the indices of these FermionOperators and the qubits in our
    # grid that allows for the perfect pairing pairs to be in squares of four.
    fermion_index_to_qubit_map = {
        4: cirq.GridQubit(0, 0),
        5: cirq.GridQubit(1, 0),
        6: cirq.GridQubit(0, 1),
        7: cirq.GridQubit(1, 1),
        2: cirq.GridQubit(0, 2),
        3: cirq.GridQubit(1, 2),
        8: cirq.GridQubit(0, 3),
        9: cirq.GridQubit(1, 3),
        0: cirq.GridQubit(0, 4),
        1: cirq.GridQubit(1, 4),
        10: cirq.GridQubit(0, 5),
        11: cirq.GridQubit(1, 5),
    }

    return fermion_index_to_qubit_map


def get_16_qubit_fermion_qubit_map():
    """A helper function that provides the fermion qubit map for 16 qubits.

    We map the fermionic orbitals to grid qubits like so:
    7 9 5 11 3 13 1 15
    6 8 4 10 2 12 0 14
    """

    # Linear connectivity is fine.
    # This ordering is dictated by the way we specify perfect pairing (i)
    # Elsewhere we generate the perfect pairing parameters using a specific
    # convention for how we index the FermionOperators (which is itself)
    # partly dictated by the OpenFermion conventions. Here we choose a
    # mapping between the indices of these FermionOperators and the qubits in our
    # grid that allows for the perfect pairing pairs to be in squares of four.
    fermion_index_to_qubit_map = {
        6: cirq.GridQubit(0, 0),
        7: cirq.GridQubit(1, 0),
        8: cirq.GridQubit(0, 1),
        9: cirq.GridQubit(1, 1),
        4: cirq.GridQubit(0, 2),
        5: cirq.GridQubit(1, 2),
        10: cirq.GridQubit(0, 3),
        11: cirq.GridQubit(1, 3),
        2: cirq.GridQubit(0, 4),
        3: cirq.GridQubit(1, 4),
        12: cirq.GridQubit(0, 5),
        13: cirq.GridQubit(1, 5),
        0: cirq.GridQubit(0, 6),
        1: cirq.GridQubit(1, 6),
        14: cirq.GridQubit(0, 7),
        15: cirq.GridQubit(1, 7),
    }

    return fermion_index_to_qubit_map


def get_fermion_qubit_map_pp_plus(*, n_qubits: int) -> Dict[int, cirq.GridQubit]:
    """Dispatcher for qubit mappings."""
    if n_qubits == 4:
        return get_4_qubit_fermion_qubit_map()
    elif n_qubits == 8:
        return get_8_qubit_fermion_qubit_map()
    elif n_qubits == 12:
        return get_12_qubit_fermion_qubit_map()
    elif n_qubits == 16:
        return get_16_qubit_fermion_qubit_map()
    else:
        raise NotImplementedError()


def get_mode_qubit_map_pp_plus(*, n_qubits: int) -> Dict[FermionicMode, cirq.GridQubit]:
    """A map from Fermionic modes to qubits for our particular circuits.

    This function dispatches to _get_fermion_qubit_map_pp_plus, and ultimately
    to get_X_qubit_fermion_qubit_map for specific values of X but it translates
    this logic to a new system that uses the FermionicMode class rather than
    opaque combinations of integers and strings.

    Args:
        n_qubits: The number of qubits.
    """
    old_fermion_qubit_map = get_fermion_qubit_map_pp_plus(n_qubits=n_qubits)

    n_orb = n_qubits // 2

    mode_qubit_map = {}

    for i in range(n_orb):
        mode_qubit_map[FermionicMode(i, "a")] = old_fermion_qubit_map[2 * i]
        mode_qubit_map[FermionicMode(i, "b")] = old_fermion_qubit_map[2 * i + 1]

    return mode_qubit_map
