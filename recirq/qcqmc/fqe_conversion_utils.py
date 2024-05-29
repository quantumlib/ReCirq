"""Contains edited versions of certain FQE conversion utilities."""

from typing import Dict, Optional, TYPE_CHECKING

import numpy
from cirq import LineQubit
from fqe.cirq_utils import qubit_projection
from fqe.openfermion_utils import (
    ascending_index_order,
    fci_fermion_operator_representation,
    fermion_opstring_to_bitstring,
    update_operator_coeff,
)
from fqe.wavefunction import Wavefunction as FqeWavefunction
from openfermion import FermionOperator, jordan_wigner, reorder, reverse_jordan_wigner

if TYPE_CHECKING:
    import fqe
    import openfermion as of


def fill_in_wfn_from_cirq(
    wfn: FqeWavefunction,
    state: numpy.ndarray,
    fermion_ind_to_qubit_ind: Optional[Dict[int, int]] = None,
) -> None:
    """For each available FqeData structure, find the projection onto the cirq
    wavefunction and set the coefficients to the proper value.

    Args:
        wfn - an Fqe Wavefunction to fill from the cirq wavefunction

        state - a cirq state to convert into an Fqe wavefunction

        fermion_ind_to_qubit_ind - A dictionary that describes the assignment of
            the fermionic modes to the qubit modes.

    Returns:
        nothing - mutates the wfn in place
    """
    nqubits = int(numpy.log2(state.size))
    for key in wfn.sectors():
        usevec = state.copy()
        fqe_wfn = cirq_to_fqe_single(usevec, key[0], key[1], nqubits, fermion_ind_to_qubit_ind)
        wfndata = fermion_opstring_to_bitstring(fqe_wfn)
        for val in wfndata:
            wfn[(val[0], val[1])] = val[2]


def cirq_to_fqe_single(
    cirq_wfn: numpy.ndarray,
    nele: int,
    m_s: int,
    qubin: int,
    fermion_ind_to_qubit_ind: Optional[Dict[int, int]] = None,
) -> FermionOperator:
    """Given a wavefunction from cirq, create a FermionOperator string which
    will create the same state in the basis of Fermionic modes such that

    .. math::

        |\\Psi\\rangle &= \\mathrm{(qubit\\ operators)}|0 0\\cdots\\rangle
        = \\mathrm{(Fermion\\ Operators)}|\\mathrm{vac}\\rangle \\\\
        |\\Psi\\rangle &= \\sum_iC_i \\mathrm{ops}_{i}|\\mathrm{vac}>

    where the c_{i} are the projection of the wavefunction onto a FCI space.

    Args:
        cirq_wfn (numpy.array(ndim=1, numpy.dtype=complex64)) - coeffcients in \
            the qubit basis.

        nele (int) - the number of electrons

        m_s (int) - the s_z spin angular momentum

        qubin (int) - The number of qubits in the wavefunction.

        fermion_ind_to_qubit_ind - A dictionary that describes the assignment of
            the fermionic modes to the qubit modes.

    Returns:
        FermionOperator
    """
    if nele == 0:
        return FermionOperator('', cirq_wfn[0] * 1.0)

    if qubin:
        nqubits = qubin
    else:
        nqubits = int(numpy.log2(cirq_wfn.size))

    if nele > nqubits:
        raise ValueError('particle number > number of orbitals')

    norb = nqubits // 2

    jw_ops = fci_qubit_representation(norb, nele, m_s, fermion_ind_to_qubit_ind)

    qubits = LineQubit.range(nqubits)
    proj_coeff = numpy.zeros(len(jw_ops.terms), dtype=numpy.complex128)
    qubit_projection(jw_ops, qubits, cirq_wfn, proj_coeff)
    proj_coeff /= 2.0**nele

    update_operator_coeff(jw_ops, proj_coeff)
    return convert_qubit_wfn_to_fqe_syntax(jw_ops, fermion_ind_to_qubit_ind)


def fci_qubit_representation(
    norb: int, nele: int, m_s: int, fermion_ind_to_qubit_ind: Optional[Dict[int, int]] = None
) -> 'of.QubitOperator':
    """Create the qubit representation of Full CI according to the parameters
    passed

    Args:
        norb (int) - number of spatial orbitals

        nele (int) - number of electrons

        m_s (int) - spin projection onto sz

        fermion_ind_to_qubit_ind - A dictionary that describes the assignment of
            the fermionic modes to the qubit modes.

    Returns:
        ops (QubitOperator)
    """
    fermion_op = fci_fermion_operator_representation(norb, nele, m_s)

    # We might need to reorder the terms.
    if fermion_ind_to_qubit_ind is not None:
        num_modes = len(fermion_ind_to_qubit_ind.items())

        def mapper(idx, num_modes):
            return fermion_ind_to_qubit_ind[idx]

        fermion_op = reorder(fermion_op, mapper, num_modes=num_modes)

    return jordan_wigner(fermion_op)


def convert_qubit_wfn_to_fqe_syntax(
    ops: 'of.QubitOperator', fermion_ind_to_qubit_ind: Optional[Dict[int, int]] = None
) -> 'FermionOperator':
    """This takes a qubit wavefunction in the form of a string of qubit
    operators and returns a string of FermionOperators with the proper
    formatting for easy digestion by FQE

    Args:
        ops (QubitOperator) - a string of qubit operators

        fermion_ind_to_qubit_ind - A dictionary that describes the assignment of
            the fermionic modes to the qubit modes.

    Returns:
        out (FermionOperator) - a string of fermion operators
    """

    # We might need to reorder the terms.
    ferm_str = reverse_jordan_wigner(ops)

    if fermion_ind_to_qubit_ind is not None:
        num_modes = len(fermion_ind_to_qubit_ind.items())

        def mapper(idx, num_modes):
            return fermion_ind_to_qubit_ind[idx]

        ferm_str = reorder(ferm_str, mapper, num_modes=num_modes, reverse=True)

    out = FermionOperator()
    for term in ferm_str.terms:
        out += ascending_index_order(term, ferm_str.terms[term])

    return out
