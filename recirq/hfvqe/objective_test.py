"""
Test the Hartree-Fock Objective function

1. Hamiltonian expectation
2. The gradient values (local)
3. The Hessian values (local)

The local gradient is the gradient assuming $kappa = 0$.  This is the case
in most electronic structure codes because the Hamiltonian (or orbitals)
are always rotated to the $kappa=0$ point after updating the parameters in
$kappa$.
"""
from itertools import product

import cirq

import numpy as np

import scipy as sp

import openfermion as of

from openfermioncirq.experiments.hfvqe.circuits import \
    prepare_slater_determinant
from openfermioncirq.experiments.hfvqe.objective import get_matrix_of_eigs
from openfermioncirq.experiments.hfvqe.circuits import rhf_params_to_matrix
from openfermioncirq.experiments.hfvqe.molecular_example import make_h6_1_3




def get_opdm(wf, num_orbitals, transform=of.jordan_wigner):
    opdm_hw = np.zeros((num_orbitals, num_orbitals),
                       dtype=np.complex128)
    creation_ops = [
        of.get_sparse_operator(transform(of.FermionOperator(((p, 1)))),
                               n_qubits=num_orbitals)
        for p in range(num_orbitals)
    ]
    # not using display style objects
    for p, q in product(range(num_orbitals), repeat=2):
        operator = creation_ops[p] @ creation_ops[q].conj().transpose()
        opdm_hw[p, q] = wf.conj().T @ operator @ wf

    return opdm_hw


def test_global_gradient_h4():
    """
    Test the gradient at the solution given by psi4
    """
    # first get molecule
    rhf_objective, molecule, params, _, _ = make_h6_1_3()
    # molecule = h4_linear_molecule(1.0)
    nocc = molecule.n_electrons // 2
    occ = list(range(nocc))
    virt = list(range(nocc, molecule.n_orbitals))

    qubits = cirq.LineQubit.range(molecule.n_orbitals)
    u = sp.linalg.expm(rhf_params_to_matrix(params,
                                            len(qubits),
                                            occ=occ,
                                            virt=virt))
    circuit = cirq.Circuit(prepare_slater_determinant(qubits, u[:, :nocc].T))

    simulator = cirq.Simulator(dtype=np.complex128)
    wf = simulator.simulate(circuit).final_state.reshape((-1, 1))
    opdm_alpha = get_opdm(wf, molecule.n_orbitals)
    opdm = np.zeros((molecule.n_qubits, molecule.n_qubits), dtype=np.complex128)
    opdm[::2, ::2] = opdm_alpha
    opdm[1::2, 1::2] = opdm_alpha

    grad = rhf_objective.global_gradient_opdm(params, opdm_alpha)

    # get finite difference gradient
    finite_diff_grad = np.zeros(9)
    epsilon = 0.0001
    for i in range(9):
        params_epsilon = params.copy()
        params_epsilon[i] += epsilon
        u = sp.linalg.expm(rhf_params_to_matrix(params_epsilon, len(qubits),
                                                occ=occ,
                                                virt=virt))
        circuit = cirq.Circuit(
            prepare_slater_determinant(qubits, u[:, :nocc].T))
        wf = simulator.simulate(circuit).final_state.reshape((-1, 1))
        opdm_pepsilon = get_opdm(wf, molecule.n_orbitals)
        energy_plus_epsilon = rhf_objective.energy_from_opdm(opdm_pepsilon)

        params_epsilon[i] -= 2 * epsilon
        u = sp.linalg.expm(rhf_params_to_matrix(params_epsilon,
                                                len(qubits),
                                                occ=occ,
                                                virt=virt))
        circuit = cirq.Circuit(
            prepare_slater_determinant(qubits, u[:, :nocc].T))
        wf = simulator.simulate(circuit).final_state.reshape((-1, 1))
        opdm_pepsilon = get_opdm(wf, molecule.n_orbitals)
        energy_minus_epsilon = rhf_objective.energy_from_opdm(opdm_pepsilon)

        finite_diff_grad[i] = (energy_plus_epsilon -
                               energy_minus_epsilon) / (2 * epsilon)

    assert np.allclose(finite_diff_grad, grad, atol=epsilon)

    # random parameters now
    params = np.random.randn(9)
    u = sp.linalg.expm(rhf_params_to_matrix(params,
                                            len(qubits),
                                            occ=occ,
                                            virt=virt))
    circuit = cirq.Circuit(prepare_slater_determinant(qubits, u[:, :nocc].T))

    simulator = cirq.Simulator(dtype=np.complex128)
    wf = simulator.simulate(circuit).final_state.reshape((-1, 1))
    opdm_alpha = get_opdm(wf, molecule.n_orbitals)
    opdm = np.zeros((molecule.n_qubits, molecule.n_qubits), dtype=np.complex128)
    opdm[::2, ::2] = opdm_alpha
    opdm[1::2, 1::2] = opdm_alpha
    grad = rhf_objective.global_gradient_opdm(params, opdm_alpha)

    # get finite difference gradient
    finite_diff_grad = np.zeros(9)
    epsilon = 0.0001
    for i in range(9):
        params_epsilon = params.copy()
        params_epsilon[i] += epsilon
        u = sp.linalg.expm(rhf_params_to_matrix(params_epsilon, len(qubits),
                                                occ=occ,
                                                virt=virt))
        circuit = cirq.Circuit(
            prepare_slater_determinant(qubits, u[:, :nocc].T))
        wf = simulator.simulate(circuit).final_state.reshape((-1, 1))
        opdm_pepsilon = get_opdm(wf, molecule.n_orbitals)
        energy_plus_epsilon = rhf_objective.energy_from_opdm(opdm_pepsilon)

        params_epsilon[i] -= 2 * epsilon
        u = sp.linalg.expm(rhf_params_to_matrix(params_epsilon,
                                                len(qubits),
                                                occ=occ,
                                                virt=virt))
        circuit = cirq.Circuit(
            prepare_slater_determinant(qubits, u[:, :nocc].T))
        wf = simulator.simulate(circuit).final_state.reshape((-1, 1))
        opdm_pepsilon = get_opdm(wf, molecule.n_orbitals)
        energy_minus_epsilon = rhf_objective.energy_from_opdm(opdm_pepsilon)

        finite_diff_grad[i] = (energy_plus_epsilon -
                               energy_minus_epsilon) / (2 * epsilon)

    assert np.allclose(finite_diff_grad, grad, atol=epsilon)


def test_get_matrix_of_eigs():
    """
    Generate the matrix of [exp(i (li - lj)) - 1] / (i(li - lj)
    :return:
    """
    lam_vals = np.random.randn(4) + 1j * np.random.randn(4)
    mat_eigs = np.zeros((lam_vals.shape[0],
                         lam_vals.shape[0]),
                         dtype=np.complex128)
    for i, j in product(range(lam_vals.shape[0]), repeat=2):
        if np.isclose(abs(lam_vals[i] - lam_vals[j]), 0):
            mat_eigs[i, j] = 1
        else:
            mat_eigs[i, j] = (np.exp(1j * (lam_vals[i] - lam_vals[j])) - 1) / (
                        1j * (lam_vals[i] - lam_vals[j]))

    test_mat_eigs = get_matrix_of_eigs(lam_vals)
    assert np.allclose(test_mat_eigs, mat_eigs)
