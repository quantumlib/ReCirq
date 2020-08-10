from typing import List, Optional, Union
from copy import deepcopy
from itertools import product

import cirq
import numpy as np
from scipy.linalg import expm

from openfermion import slater_determinant_preparation_circuit

from openfermioncirq.experiments.hfvqe import util
# pylint: disable=C


def rhf_params_to_matrix(parameters: np.ndarray, num_qubits: int,
                         occ: Optional[Union[None, List[int]]]=None,
                         virt: Optional[Union[None, List[int]]]=None):
    """
    For restricted Hartree-Fock we have nocc * nvirt parameters.  These are provided
    as a list that is ordered by (virtuals) \times (occupied) where
    occupied is a set of indices corresponding to the occupied oribitals w.r.t the
    Lowdin basis and virtuals is a set of indices of the virutal orbitals w.r.t the
    Lowdin basis.  For example, for H4 we have 2 orbitals occupied and 2 virtuals

    occupied = [0, 1]  virtuals = [2, 3]

    parameters = [(v_{0}, o_{0}), (v_{0}, o_{1}), (v_{1}, o_{0}), (v_{1}, o_{1})]
               = [(2, 0), (2, 1), (3, 0), (3, 1)]

    You can think of the tuples of elements of the upper right triangle of the
    antihermitian matrix that specifies the c_{b, i} coefficients.

    coefficient matrix
    [[ c_{0, 0}, -c_{1, 0}, -c_{2, 0}, -c_{3, 0}],
     [ c_{1, 0},  c_{1, 1}, -c_{2, 1}, -c_{3, 1}],
     [ c_{2, 0},  c_{2, 1},  c_{2, 2}, -c_{3, 2}],
     [ c_{3, 0},  c_{3, 1},  c_{3, 2},  c_{3, 3}]]

    Since we are working with only non-redundant operators we know c_{i, i} = 0
    and any c_{i, j} where i and j are both in occupied or both in virtual = 0.
    """
    if occ is None:
        occ = range(num_qubits//2)
    if virt is None:
        virt = range(num_qubits // 2, num_qubits)

    # check that parameters are a real array
    if not np.allclose(parameters.imag, 0):
        raise ValueError("parameters input must be real valued")

    kappa = np.zeros((len(occ) + len(virt), len(occ) + len(virt)))
    for idx, (v, o) in enumerate(product(virt, occ)):
        kappa[v, o] = parameters[idx].real
        kappa[o, v] = -parameters[idx].real
    return kappa


def generate_circuits_from_params_or_u(qubits: List[cirq.Qid],
                                       parameters: np.ndarray,
                                       nocc: int,
                                       return_unitaries: Optional[bool] = False,
                                       occ: Optional[Union[None, List[int]]] = None,
                                       virt: Optional[Union[None, List[int]]] = None,
                                       clean_ryxxy: Optional[bool] = False):  # testpragma: no cover
    """
    Make the circuits required for the estimation of the 1-RDM

    :param qubits: define the qubits in the memory
    :param parameters: parameters of the kappa matrix
    :param nocc: number of occupied orbitals
    :param return_unitaries:  Check if the user wants unitaries returned
    :param occ: List of occupied indices
    :param virt: List of virtual orbitals
    :param clean_ryxxy: Determine the type of Givens rotation synthesis to use
                        Options are 1, 2, 3, 4.
    :return: List[cirq.OP_TREEE]
    """

    num_qubits = len(qubits)
    # determine if parameters is a unitary
    if len(parameters.shape) == 2:
        if parameters.shape[0] == parameters.shape[1]:
            unitary = parameters
    else:
        generator = rhf_params_to_matrix(parameters,
                                         num_qubits,
                                         occ=occ,
                                         virt=virt)
        unitary = expm(generator)

    circuits = []
    unitaries = []
    for swap_depth in range(0, num_qubits, 2):
        fswap_pairs = util.generate_fswap_pairs(swap_depth, num_qubits)
        swap_unitaries = util.generate_fswap_unitaries(fswap_pairs, num_qubits)
        shifted_unitary = unitary.copy()
        for uu in swap_unitaries:
            shifted_unitary = uu @ shifted_unitary
        unitaries.append(shifted_unitary)
        matrix = shifted_unitary.T[:nocc, :]

        permuted_circuit = cirq.Circuit()
        permuted_circuit += prepare_slater_determinant(qubits,
                                                       matrix.real.copy(),
                                                       clean_ryxxy=clean_ryxxy)
        circuits.append(permuted_circuit)

    if return_unitaries:
        return circuits, unitaries

    return circuits


def xxyy_basis_rotation(pairs, clean_xxyy = False):
    """Generate the measurement circuits"""
    all_ops = []

    for a, b in pairs:
        if clean_xxyy:
            all_ops += [cirq.rz(-np.pi*0.25).on(a),
                        cirq.rz(np.pi * 0.25).on(b),
                        cirq.ISWAP.on(a, b) ** 0.5]
        else:
            all_ops += [cirq.rz(-np.pi*0.25).on(a),
                        cirq.rz(np.pi * 0.25).on(b),
                        cirq.FSimGate(-np.pi/4, np.pi/24).on(a, b)]
    return all_ops


def circuits_with_measurements(qubits, circuits, clean_xxyy = False):  # testpragma: no cover
    """Append the appropriate measurements to each of the permutation circuits"""
    num_qubits = len(qubits)
    even_pairs = [
        qubits[idx:idx + 2] for idx in np.arange(0, num_qubits - 1, 2)
    ]
    odd_pairs = [qubits[idx:idx + 2] for idx in np.arange(1, num_qubits - 1, 2)]
    measure_labels = ['z', 'xy_even', 'xy_odd']
    all_circuits_with_measurements = {label: {} for label in measure_labels}
    for circuit_index in range(len(circuits)):
        for _, label in enumerate(measure_labels):
            circuit = deepcopy(circuits[circuit_index])
            if label == 'xy_even':
                circuit.append(xxyy_basis_rotation(even_pairs,
                                                   clean_xxyy=clean_xxyy),
                               strategy=cirq.InsertStrategy.EARLIEST)
            if label == 'xy_odd':
                circuit.append(xxyy_basis_rotation(odd_pairs,
                                                   clean_xxyy=clean_xxyy),
                               strategy=cirq.InsertStrategy.EARLIEST)
            circuit.append(cirq.Moment([cirq.measure(q) for q in qubits]))
            all_circuits_with_measurements[label][circuit_index] = circuit
    return all_circuits_with_measurements


def prepare_slater_determinant(qubits: List[cirq.Qid],
                               slater_determinant_matrix: np.ndarray,
                               clean_ryxxy: Optional[Union[bool, int]] = True):
    """
    High level interface to the real basis rotation circuit generator

    :param qubits: List of cirq.Qids denoting logical qubits
    :param slater_determinant_matrix: basis rotation matrix
    :param clean_ryxxy: Optional[True, 1, 2, 3, 4] for indicating an error
                        model of the Givens rotation.
    :return: generator for circuit
    """
    circuit_description = slater_determinant_preparation_circuit(slater_determinant_matrix)
    yield (cirq.X(qubits[j]) for j in range(slater_determinant_matrix.shape[0]))
    for parallel_ops in circuit_description:
        for op in parallel_ops:
            i, j, theta, phi = op
            if not np.isclose(phi, 0):  # testpragma: no cover
                raise ValueError("unitary must be real valued only")
            if clean_ryxxy is True or clean_ryxxy == 1:
                yield ryxxy(qubits[i], qubits[j], theta)
            elif clean_ryxxy == 2:
                yield ryxxy2(qubits[i], qubits[j], theta)
            elif clean_ryxxy == 3:
                yield ryxxy3(qubits[i], qubits[j], theta)
            elif clean_ryxxy == 4:
                yield ryxxy3(qubits[i], qubits[j], theta)
            else:
                raise ValueError("Invalide clean_ryxxy value")


def ryxxy(a, b, theta):
    """Implements the givens rotation with sqrt(iswap).
    The inverse(sqrt(iswap)) is made with z before and after"""
    yield cirq.ISWAP.on(a, b) ** 0.5
    yield cirq.rz(-theta + np.pi).on(a)
    yield cirq.rz(theta).on(b)
    yield cirq.ISWAP.on(a, b) ** 0.5
    yield cirq.rz(np.pi).on(a)


def ryxxy2(a, b, theta):
    """
    Implement realistic Givens rotation considering the always on parasitic
    cphase
    """
    yield cirq.FSimGate(-np.pi/4, np.pi/24).on(a, b)
    yield cirq.rz(-theta + np.pi).on(a)
    yield cirq.rz(theta).on(b)
    yield cirq.FSimGate(-np.pi/4, np.pi/24).on(a, b)
    yield cirq.rz(np.pi).on(a)


def ryxxy3(a, b, theta):
    """
    Implement realistic Givens rotation considering the always on parasitic
    cphase and attempt to reduce the error by 1/3
    """
    yield cirq.FSimGate(-np.pi/4, np.pi/24).on(a, b)
    yield cirq.rz(-theta + np.pi + np.pi/48).on(a)
    yield cirq.rz(theta + np.pi/48).on(b)
    yield cirq.FSimGate(-np.pi/4, np.pi/24).on(a, b)
    yield cirq.rz(np.pi + np.pi/48).on(a)
    yield cirq.rz(+np.pi/48).on(b)


def ryxxy4(a, b, theta):
    """
    Implement realistic Givens rotation considering the always on parasitic
    cphase and attempt to reduce the error by 1/3 for running on hardware
    """
    yield cirq.FSimGate(-np.pi/4, 0).on(a, b)
    yield cirq.rz(-theta + np.pi + np.pi/48).on(a)
    yield cirq.rz(theta + np.pi/48).on(b)
    yield cirq.FSimGate(-np.pi/4, 0).on(a, b)
    yield cirq.rz(np.pi + np.pi/48).on(a)
    yield cirq.rz(+np.pi/48).on(b)
