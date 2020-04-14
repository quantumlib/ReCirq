# Copyright 2020 Google
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

"""
This is a collection of useful functions that help simulate QAOA efficiently
Has been used on a standard laptop up to system size N=24

Based on arXiv:1812.01041 and https://github.com/leologist/GenQAOA
"""
from typing import Tuple, Optional, Dict, List

import itertools
import multiprocessing

import networkx as nx
import numpy as np

import cirq

SIGMA_X_IND, SIGMA_Y_IND, SIGMA_Z_IND = (1, 2, 3)


def minimizable_qaoa_fun(graph: nx.Graph, flag_z2_sym=True,
                         node_to_index_map=None, dtype=np.complex128):
    r"""Given a graph, return a function suitable for scipy minimization.

    Specifically, return f(param) that outputs a tuple (F, Fgrad), where
    F = <C> is the expectation value of objective function with respect
    to the QAOA wavefunction, and Fgrad = the analytically calculated gradient
    of F with respect to the QAOA parameters

    This can then be passed to scipy.optimize.minimize, with jac=True

    By convention, C = \sum_{<ij> \in graph} w_{ij} Z_i Z_j
    """
    HamC = create_ZZ_HamC(graph=graph, flag_z2_sym=flag_z2_sym,
                          node_to_index_map=node_to_index_map, dtype=dtype)
    n_nodes = graph.number_of_nodes()

    def f(param):
        return ising_qaoa_grad(n_nodes, HamC, param, flag_z2_sym, dtype=dtype)

    return f


def create_ZZ_HamC(graph: nx.Graph, flag_z2_sym=True, node_to_index_map=None, dtype=np.complex128):
    r"""Generate a vector corresponding to the diagonal of the C Hamiltonian

    Setting flag_z2_sym to True means we're only working in the X^\otimes N = +1
    symmetric sector which reduces the Hilbert space dimension by a factor of 2,
    saving memory.
    """
    n_nodes = graph.number_of_nodes()
    HamC = np.zeros((2 ** n_nodes, 1), dtype=dtype)

    sigma_z = np.asarray([[1], [-1]], dtype=dtype)
    if node_to_index_map is None:
        node_to_index_map = {q: i for i, q in enumerate(graph.nodes)}

    for a, b in graph.edges:
        HamC += graph[a][b]['weight'] * ham_two_local_term(
            sigma_z, sigma_z, node_to_index_map[a], node_to_index_map[b], n_nodes, dtype=dtype)

    if flag_z2_sym:
        # restrict to first half of Hilbert space
        return HamC[range(2 ** (n_nodes - 1)), 0]
    else:
        return HamC[:, 0]


def ham_two_local_term(op1, op2, ind1, ind2, N, dtype=np.complex128):
    r"""Utility function for conveniently creating a 2-Local term op1 \otimes op2 among N spins"""
    if ind1 > ind2:
        return ham_two_local_term(op2, op1, ind2, ind1, N, dtype=dtype)

    if op1.shape != op2.shape or op1.ndim != 2:
        raise ValueError('ham_two_local_term: invalid operator input')

    if ind1 < 0 or ind2 > N - 1:
        raise ValueError('ham_two_local_term: invalid input indices')

    if op1.shape[0] == 1 or op1.shape[1] == 1:
        myeye = lambda n: np.ones(np.asarray(op1.shape) ** n, dtype=dtype)
    else:
        myeye = lambda n: np.eye(np.asarray(op1.shape) ** n, dtype=dtype)

    return np.kron(myeye(ind1),
                   np.kron(op1, np.kron(myeye(ind2 - ind1 - 1),
                                        np.kron(op2, myeye(N - ind2 - 1)))))


def multiply_single_spin(psi, N: int, i: int, pauli_ind: int, dtype=np.complex128):
    """ Multiply a single pauli operator on the i-th spin of the input wavefunction

        Args:
            psi: input wavefunction (as numpy.ndarray)
            N: number of spins
            i: zero-based index of spin location to apply pauli
            pauli_ind: one of (1,2,3) for (X, Y, Z)
    """

    IndL = 2 ** i
    IndR = 2 ** (N - i - 1)

    out = psi.reshape([IndL, 2, IndR], order='F').copy()

    if pauli_ind == SIGMA_X_IND:  # sigma_X
        out = np.flip(out, 1)
    elif pauli_ind == SIGMA_Y_IND:  # sigma_Y
        out = np.flip(out, 1).astype(dtype, copy=False)
        out[:, 0, :] = -1j * out[:, 0, :]
        out[:, 1, :] = 1j * out[:, 1, :]
    elif pauli_ind == SIGMA_Z_IND:  # sigma_Z
        out[:, 1, :] = -out[:, 1, :]

    return out.reshape(2 ** N, order='F')


def evolve_by_HamB(N, beta, psi_in, flag_z2_sym=False, copy=True, dtype=np.complex128):
    r"""Use reshape to efficiently implement evolution under B=\sum_i X_i"""
    if copy:
        psi = psi_in.copy()
    else:
        psi = psi_in

    if not flag_z2_sym:
        for i in range(N):
            psi = (np.cos(beta) * psi
                   - 1j * np.sin(beta) * multiply_single_spin(psi, N, i, SIGMA_X_IND, dtype))
    else:
        for i in range(N - 1):
            psi = (np.cos(beta) * psi
                   - 1j * np.sin(beta) * multiply_single_spin(psi, N - 1, i, SIGMA_X_IND, dtype))
        psi = np.cos(beta) * psi - 1j * np.sin(beta) * np.flipud(psi)

    return psi


def ising_qaoa_grad(N, HamC, param, flag_z2_sym=False, dtype=np.complex128):
    """For QAOA on Ising problems, calculate the objective function F and its
        gradient exactly

    Args:
        N: number of spins
        HamC: a vector of diagonal of objective Hamiltonian in Z basis
        param: parameters of QAOA. Should be 2*p in length
        flag_z2_sym: if True, we're only working in the Z2-symmetric sector
            (saves Hilbert space dimension by a factor of 2)
            default set to False because we can take more general HamC
            that is not Z2-symmetric.

    Returns:
        F: <HamC> for minimization
        Fgrad: gradient of F with respect to param
    """
    p = len(param) // 2
    gammas = param[:p]
    betas = param[p:]

    def evolve_by_ham_b_local(beta, psi):
        return evolve_by_HamB(N, beta, psi, flag_z2_sym, copy=False, dtype=dtype)

    # pre-allocate space for storing 2p+2 copies of wavefunction
    # This is necessary for efficient computation of analytic gradient
    if flag_z2_sym:
        psi_p = np.zeros([2 ** (N - 1), 2 * p + 2], dtype=dtype)
        psi_p[:, 0] = 1 / 2 ** ((N - 1) / 2)
    else:
        psi_p = np.zeros([2 ** N, 2 * p + 2], dtype=dtype)
        psi_p[:, 0] = 1 / 2 ** (N / 2)

    # evolving forward
    for q in range(p):
        psi_p[:, q + 1] = evolve_by_ham_b_local(
            betas[q], np.exp(-1j * gammas[q] * HamC) * psi_p[:, q])

    # multiply by HamC
    psi_p[:, p + 1] = HamC * psi_p[:, p]

    # evolving backwards
    for q in range(p):
        psi_p[:, p + 2 + q] = (np.exp(1j * gammas[p - 1 - q] * HamC)
                               * evolve_by_ham_b_local(-betas[p - 1 - q], psi_p[:, p + 1 + q]))

    # evaluating objective function
    F = np.real(np.vdot(psi_p[:, p], psi_p[:, p + 1]))

    # evaluating gradient analytically
    Fgrad = np.zeros(2 * p)
    for q in range(p):
        Fgrad[q] = -2 * np.imag(np.vdot(psi_p[:, q], HamC * psi_p[:, 2 * p + 1 - q]))

        if not flag_z2_sym:
            psi_temp = np.zeros(2 ** N, dtype=dtype)
            for i in range(N):
                psi_temp += multiply_single_spin(psi_p[:, 2 * p - q], N, i, 1, dtype=dtype)
        else:
            psi_temp = np.zeros(2 ** (N - 1), dtype=dtype)
            for i in range(N - 1):
                psi_temp += multiply_single_spin(psi_p[:, 2 * p - q], N - 1, i, 1, dtype=dtype)
            psi_temp += np.flipud(psi_p[:, 2 * p - q])

        Fgrad[p + q] = -2 * np.imag(np.vdot(psi_p[:, q + 1], psi_temp))

    return F, Fgrad


def ising_qaoa_expectation_and_variance(N, HamC, param, flag_z2_sym=False, dtype=np.complex128):
    """For QAOA on Ising problems, calculate the expectation, variance, and wavefunction.

    Args:
        N: number of spins
        HamC: a vector of diagonal of objective Hamiltonian in Z basis
        param: parameters of QAOA, should be 2*p in length
        flag_z2_sym: if True, we're only working in the Z2-symmetric sector
            (saves Hilbert space dimension by a factor of 2)
            default set to False because we can take more general HamC
            that is not Z2-symmetric

    Returns: expectation, variance, wavefunction
    """
    p = len(param) // 2
    gammas = param[:p]
    betas = param[p:]

    def evolve_by_HamB_local(beta, psi):
        return evolve_by_HamB(N, beta, psi, flag_z2_sym, False, dtype=dtype)

    if flag_z2_sym:
        psi = np.empty(2 ** (N - 1), dtype=dtype)
        psi[:] = 1 / 2 ** ((N - 1) / 2)
    else:
        psi = np.empty(2 ** N, dtype=dtype)
        psi[:] = 1 / 2 ** (N / 2)

    for q in range(p):
        psi = evolve_by_HamB_local(betas[q], np.exp(-1j * gammas[q] * HamC) * psi)

    expectation = np.real(np.vdot(psi, HamC * psi))
    variance = np.real(np.vdot(psi, HamC ** 2 * psi) - expectation ** 2)

    return expectation, variance, psi


def qaoa_expectation_and_variance_fun(
        graph: nx.Graph,
        flag_z2_sym=True,
        node_to_index_map=None,
        dtype=np.complex128):
    r""" Return a function f(param) that outputs the expectation, variance,
    and evolved wavefunction for the QAOA evaluated at the parameters.

    By convention, C = \sum_{<ij> \in graph} w_{ij} Z_i Z_j
    """
    HamC = create_ZZ_HamC(graph, flag_z2_sym, node_to_index_map, dtype=dtype)
    N = graph.number_of_nodes()

    def f(param):
        return ising_qaoa_expectation_and_variance(
            N=N, HamC=HamC, param=param, flag_z2_sym=flag_z2_sym, dtype=dtype)

    return f


def _ising_qaoa_expectation(N, HamC, param, flag_z2_sym, dtype):
    return ising_qaoa_expectation_and_variance(
        N, HamC, param, flag_z2_sym, dtype)[0]


def exact_qaoa_values_on_grid(
        graph: nx.Graph,
        xlim: Tuple[float, float] = (0, np.pi / 2),
        ylim: Tuple[float, float] = (-np.pi / 4, np.pi / 4),
        x_grid_num: int = 20,
        y_grid_num: int = 20,
        num_processors: int = 1,
        dtype=np.complex128):
    """Compute exact p=1 QAOA values on a grid.

    Args:
        graph: The graph representing the Hamiltonian.
        xlim: The range of values for gamma.
        ylim: The range of values for beta.
        num: The number of points in a single dimension of the grid.
            The total number of points evaluated will be num^2.
    Returns:
        A 2-dimensional Numpy array containing the QAOA values.
        The rows index the betas and the columns index the gammas.
    """
    a, b = xlim
    c, d = ylim
    gammas = np.linspace(a, b, x_grid_num)
    betas = np.linspace(c, d, y_grid_num)

    HamC = create_ZZ_HamC(graph, dtype=dtype)
    N = graph.number_of_nodes()
    with multiprocessing.Pool(num_processors) as pool:
        vals = pool.starmap(_ising_qaoa_expectation,
                            [(N, HamC, x, True, dtype)
                             for x in itertools.product(gammas, betas)])
    return np.reshape(np.array(vals), (x_grid_num, y_grid_num)).T


def hamiltonian_objective(
        bitstring: np.ndarray,
        graph: nx.Graph,
        node_to_index_map: Optional[Dict[cirq.Qid, int]] = None
) -> float:
    if node_to_index_map is None:
        node_to_index_map = {q: i for i, q in enumerate(sorted(graph.nodes))}
    return sum(graph[a][b]['weight'] * (-1) ** (
            bool(bitstring[node_to_index_map[a]])
            + bool(bitstring[node_to_index_map[b]]))
               for a, b in graph.edges)


def hamiltonian_objectives(
        bitstrings: np.ndarray,
        graph: nx.Graph,
        nodelist: Optional[List[int]] = None,
        readout_calibration: Optional[cirq.experiments.SingleQubitReadoutCalibrationResult] = None,
        qubit_map: Optional[Dict[int, cirq.Qid]] = None
) -> np.ndarray:
    if nodelist is None:
        nodelist = sorted(graph.nodes)
    mat = nx.adjacency_matrix(graph, nodelist=nodelist)
    if readout_calibration:
        if qubit_map is None:
            qubit_map = {q: q for q in nodelist}
        correction_matrix = np.empty(mat.shape)
        for i, j in itertools.product(range(len(graph)), repeat=2):
            p0_i = 1 - readout_calibration.zero_state_errors[qubit_map[nodelist[i]]]
            p1_i = 1 - readout_calibration.one_state_errors[qubit_map[nodelist[i]]]
            p0_j = 1 - readout_calibration.zero_state_errors[qubit_map[nodelist[j]]]
            p1_j = 1 - readout_calibration.one_state_errors[qubit_map[nodelist[j]]]
            correction_matrix[i, j] = (1 / ((p0_i + p1_i - 1) * (p0_j + p1_j - 1)))
        mat = mat.toarray() * correction_matrix
    vecs = (-1) ** bitstrings
    return 0.5 * np.sum(vecs * (mat @ vecs.T).T, axis=-1)


def hamiltonian_objective_avg_and_err(
        bitstrings: np.ndarray,
        graph: nx.Graph,
        nodelist: Optional[List[int]] = None,
) -> Tuple[float, float]:
    if nodelist is None:
        nodelist = sorted(graph.nodes)

    assert len(nodelist) == bitstrings.shape[1]
    node_to_i = {n: i for i, n in enumerate(nodelist)}

    vecs = (-1) ** bitstrings
    coeffs = []
    vars = []
    for n1, n2, w in graph.edges.data('weight'):
        coeffs += [w]
        i1 = node_to_i[n1]
        i2 = node_to_i[n2]
        vars += [vecs[:, i1] * vecs[:, i2]]

    vars = np.asarray(vars)
    coeffs = np.asarray(coeffs)[np.newaxis, :]
    f = coeffs @ np.mean(vars, axis=1)
    f = f.item()  # to normal float
    var = coeffs @ np.atleast_2d(np.cov(vars)) @ coeffs.T
    var = var.item()  # to normal float

    std_err = np.sqrt(var / len(bitstrings))
    return f, std_err


def lowest_and_highest_energy(graph: nx.Graph):
    hamiltonian = create_ZZ_HamC(graph, dtype=np.float64)
    return np.min(hamiltonian), np.max(hamiltonian)
