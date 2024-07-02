"""Contains edited versions of certain FQE conversion utilities."""

from typing import Dict, Optional, Mapping, Sequence, Callable

import numpy as np
import cirq
import fqe
import fqe.openfermion_utils as fqe_of
import fqe.wavefunction as fqe_wfn
import openfermion as of

import recirq.qcqmc.fermion_mode as fm


def fill_in_wfn_from_cirq(
    wfn: fqe_wfn.Wavefunction,
    state: np.ndarray,
    fermion_ind_to_qubit_ind: Optional[Dict[int, int]] = None,
) -> None:
    """Find the projection onto the cirq wavefunction and set the coefficients to the proper value.

    Does this for each FQE wavefunction.

    Args:
        wfn: an Fqe Wavefunction to fill from the cirq wavefunction
        state: a cirq state to convert into an Fqe wavefunction
        fermion_ind_to_qubit_ind: A dictionary that describes the assignment of
            the fermionic modes to the qubit modes.

    Returns:
        nothing - mutates the wfn in place
    """
    nqubits = int(np.log2(state.size))
    for key in wfn.sectors():
        usevec = state.copy()
        fqe_wfn = cirq_to_fqe_single(
            usevec, key[0], key[1], nqubits, fermion_ind_to_qubit_ind
        )
        wfndata = fqe_of.fermion_opstring_to_bitstring(fqe_wfn)
        for val in wfndata:
            wfn[(val[0], val[1])] = val[2]


def cirq_to_fqe_single(
    cirq_wfn: np.ndarray,
    nele: int,
    m_s: int,
    qubin: int,
    fermion_ind_to_qubit_ind: Optional[Dict[int, int]] = None,
) -> of.FermionOperator:
    """Create a FermionOperator string which will create the same state as the cirq wavefunction.
    
    Created in the basis of Fermionic modes such that

    .. math::

        |\\Psi\\rangle &= \\mathrm{(qubit\\ operators)}|0 0\\cdots\\rangle
        = \\mathrm{(Fermion\\ Operators)}|\\mathrm{vac}\\rangle \\\\
        |\\Psi\\rangle &= \\sum_iC_i \\mathrm{ops}_{i}|\\mathrm{vac}>

    where the c_{i} are the projection of the wavefunction onto a FCI space.

    Args:
        cirq_wfn: - coeffcients in the qubit basis.
        nele: - the number of electrons
        m_s: - the s_z spin angular momentum
        qubin: The number of qubits in the wavefunction.

        fermion_ind_to_qubit_ind - A dictionary that describes the assignment of
            the fermionic modes to the qubit modes.

    Returns:
        FermionOperator
    """
    if nele == 0:
        return of.FermionOperator("", cirq_wfn[0] * 1.0)

    if qubin:
        nqubits = qubin
    else:
        nqubits = int(np.log2(cirq_wfn.size))

    if nele > nqubits:
        raise ValueError("particle number > number of orbitals")

    norb = nqubits // 2

    jw_ops = fci_qubit_representation(norb, nele, m_s, fermion_ind_to_qubit_ind)

    qubits = cirq.LineQubit.range(nqubits)
    proj_coeff = np.zeros(len(jw_ops.terms), dtype=np.complex128)
    fqe.cirq_utils.qubit_projection(jw_ops, qubits, cirq_wfn, proj_coeff)
    proj_coeff /= 2.0**nele

    fqe_of.update_operator_coeff(jw_ops, proj_coeff)
    return convert_qubit_wfn_to_fqe_syntax(jw_ops, fermion_ind_to_qubit_ind)


def fci_qubit_representation(
    norb: int,
    nele: int,
    m_s: int,
    fermion_ind_to_qubit_ind: Optional[Dict[int, int]] = None,
) -> "of.QubitOperator":
    """Create the qubit representation of Full CI according to the parameters passed.

    Args:
        norb: number of spatial orbitals
        nele: number of electrons
        m_s: spin projection onto sz
        fermion_ind_to_qubit_ind: a dictionary that describes the assignment of
            the fermionic modes to the qubit modes.

    Returns:
        ops
    """
    fermion_op = fqe_of.fci_fermion_operator_representation(norb, nele, m_s)

    # We might need to reorder the terms.
    if fermion_ind_to_qubit_ind is not None:
        num_modes = len(fermion_ind_to_qubit_ind.items())

        def mapper(idx, num_modes):
            return fermion_ind_to_qubit_ind[idx]

        fermion_op = of.reorder(fermion_op, mapper, num_modes=num_modes)

    return of.jordan_wigner(fermion_op)


def get_reorder_func(
    *,
    mode_qubit_map: Mapping[fm.FermionicMode, cirq.Qid],
    ordered_qubits: Sequence[cirq.Qid],
) -> Callable[[int, int], int]:
    """This is a helper function that allows us to reorder fermionic modes.

    Under the Jordan-Wigner transform, each fermionic mode is assigned to a
    qubit. If we are provided an openfermion FermionOperator with the modes
    assigned to qubits as described by mode_qubit_map this function gives us a
    reorder_func that we can use to reorder the modes (with
    openfermion.reorder(...)) so that they match the order of the qubits in
    ordered_qubits. This is necessary to make a correspondence between
    fermionic operators / wavefunctions and their qubit counterparts.

    Args:
        mode_qubit_map: A dict that shows how each FermionicMode is mapped to a qubit.
        ordered_qubits: An ordered sequence of qubits.
    """
    qubits = list(mode_qubit_map.values())
    assert len(qubits) == len(ordered_qubits)

    # We sort the key: value pairs by the order of the values (qubits) in
    # ordered_qubits.
    sorted_mapping = list(mode_qubit_map.items())
    sorted_mapping.sort(key=lambda x: ordered_qubits.index(x[1]))

    remapping_map = {}
    for i, (mode, _) in enumerate(sorted_mapping):
        openfermion_index = 2 * mode.orb_ind + (0 if mode.spin == "a" else 1)
        remapping_map[openfermion_index] = i

    def remapper(index: int, _: int) -> int:
        """A function that maps from the old index to the new one.

        The _ argument is because it's expected by openfermion.reorder"""
        return remapping_map[index]

    return remapper


def convert_qubit_wfn_to_fqe_syntax(
    ops: of.QubitOperator, fermion_ind_to_qubit_ind: Optional[Dict[int, int]] = None
) -> of.FermionOperator:
    """Converts qubit wavefunction to FQE syntax.

    This takes a qubit wavefunction in the form of a string of qubit operators
    and returns a string of FermionOperators with the proper formatting for easy
    digestion by FQE.

    Args:
        ops (QubitOperator) - a string of qubit operators

        fermion_ind_to_qubit_ind - A dictionary that describes the assignment of
            the fermionic modes to the qubit modes.

    Returns:
        out (FermionOperator) - a string of fermion operators
    """

    # We might need to reorder the terms.
    ferm_str = of.reverse_jordan_wigner(ops)

    if fermion_ind_to_qubit_ind is not None:
        num_modes = len(fermion_ind_to_qubit_ind.items())

        def mapper(idx, num_modes):
            return fermion_ind_to_qubit_ind[idx]

        ferm_str = of.reorder(ferm_str, mapper, num_modes=num_modes, reverse=True)

    out = of.FermionOperator()
    for term in ferm_str.terms:
        out += fqe_of.ascending_index_order(term, ferm_str.terms[term])

    return out


def convert_fqe_wf_to_cirq(
    fqe_wf: fqe_wfn.Wavefunction,
    mode_qubit_map: Mapping[fm.FermionicMode, cirq.Qid],
    ordered_qubits: Sequence[cirq.Qid],
) -> np.ndarray:
    """Converts an FQE wavefunction to one on qubits with a particular ordering.

    Args:
        fqe_wf: The FQE wavefunction.
        mode_qubit_map: A mapping from fermion modes to cirq qubits.
        ordered_qubits:
    """
    n_qubits = len(mode_qubit_map)
    fermion_op = fqe.openfermion_utils.fqe_to_fermion_operator(fqe_wf)

    reorder_func = get_reorder_func(
        mode_qubit_map=mode_qubit_map, ordered_qubits=ordered_qubits
    )
    fermion_op = of.reorder(fermion_op, reorder_func, num_modes=n_qubits)

    qubit_op = of.jordan_wigner(fermion_op)

    return fqe.qubit_wavefunction_from_vacuum(
        qubit_op, list(cirq.LineQubit.range(n_qubits))
    )
