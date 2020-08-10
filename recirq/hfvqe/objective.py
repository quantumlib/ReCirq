from typing import Optional
from itertools import product
import numpy as np
from openfermion.ops import InteractionOperator, InteractionRDM
from openfermion.utils import wedge
from openfermion.transforms import get_fermion_operator
from openfermioncirq.experiments.hfvqe.circuits import rhf_params_to_matrix
# pylint: disable=C


def get_matrix_of_eigs(w: np.ndarray) -> np.ndarray:
    """
    Transform the eigenvalues for getting the gradient

    .. math:
        f(w) \rightarrow \frac{e^{i (\lambda_{i} - \lambda_{j})}{i (\lambda_{i} - \lambda_{j})}

    :param w: eigenvalues of C-matrix
    :return: new array of transformed eigenvalues
    """
    transform_eigs = np.zeros((w.shape[0], w.shape[0]),
                                 dtype=np.complex128)
    for i, j in product(range(w.shape[0]), repeat=2):
        if np.isclose(abs(w[i] - w[j]), 0):
            transform_eigs[i, j] = 1
        else:
            transform_eigs[i, j] = (np.exp(1j * (w[i] - w[j])) - 1) / (
                        1j * (w[i] - w[j]))
    return transform_eigs


class RestrictedHartreeFockObjective():
    """
    Implementation of the objective function code for Restricted Hartree-Fock

    The object transforms a variety of input types into the appropriate output.
    It does this by analyzing the type and size of the input based on its
    knowledge of each type.
    """
    def __init__(self, hamiltonian: InteractionOperator, num_electrons: int):
        self.hamiltonian = hamiltonian
        self.fermion_hamiltonian = get_fermion_operator(self.hamiltonian)
        self.num_qubits = hamiltonian.one_body_tensor.shape[0]
        self.num_orbitals = self.num_qubits // 2
        self.num_electrons = num_electrons
        self.nocc = self.num_electrons // 2
        self.nvirt = self.num_orbitals - self.nocc
        self.occ = list(range(self.nocc))
        self.virt = list(range(self.nocc, self.nocc + self.nvirt))

    def rdms_from_opdm_aa(self, opdm_aa):
        """
        Generate InteractionRDM for the problem from opdm_aa

        :param opdm_aa:
        :return:
        """
        opdm = np.zeros((self.num_qubits, self.num_qubits),
                           dtype=complex)
        opdm[::2, ::2] = opdm_aa
        opdm[1::2, 1::2] = opdm_aa
        tpdm = wedge(opdm, opdm, (1, 1), (1, 1))
        rdms = InteractionRDM(opdm, 2 * tpdm)
        return rdms

    def energy_from_opdm(self, opdm_aa: np.ndarray) -> float:
        """
        Return the energy

        :param opdm:
        :return:
        """
        rdms = self.rdms_from_opdm_aa(opdm_aa)
        return rdms.expectation(self.hamiltonian).real

    def global_gradient_opdm(self, params: np.ndarray,
                              alpha_opdm: np.ndarray):
        opdm = np.zeros((self.num_qubits, self.num_qubits),
                           dtype=np.complex128)
        opdm[::2, ::2] = alpha_opdm
        opdm[1::2, 1::2] = alpha_opdm
        tpdm = 2 * wedge(opdm, opdm, (1, 1), (1, 1))

        # now go through and generate all the necessary Z, Y, Y_kl matrices
        kappa_matrix = rhf_params_to_matrix(params,
                                            len(self.occ) + len(self.virt),
                                            self.occ, self.virt)
        kappa_matrix_full = np.kron(kappa_matrix, np.eye(2))
        w_full, v_full = np.linalg.eigh(-1j * kappa_matrix_full)  # so that kappa = i U lambda U^
        eigs_scaled_full = get_matrix_of_eigs(w_full)

        grad = np.zeros(self.nocc * self.nvirt, dtype=np.complex128)
        kdelta = np.eye(self.num_qubits)

        # NOW GENERATE ALL TERMS ASSOCIATED WITH THE GRADIENT!!!!!!
        for p in range(self.nocc * self.nvirt):
            grad_params = np.zeros_like(params)
            grad_params[p] = 1
            Y = rhf_params_to_matrix(grad_params,
                                     len(self.occ) + len(self.virt),
                                     self.occ, self.virt)
            Y_full = np.kron(Y, np.eye(2))

            # Now rotate Y int othe basis that diagonalizes Z
            Y_kl_full = v_full.conj().T @ Y_full @ v_full
            # now rotate Y_{kl} * (exp(i(l_{k} - l_{l})) - 1) / (i(l_{k} - l_{l}))
            # into the original basis
            pre_matrix_full = v_full @ (eigs_scaled_full * Y_kl_full) @ v_full.conj().T

            grad_expectation = -1.0 * np.einsum('ab,pq,aq,pb', self.hamiltonian.one_body_tensor, pre_matrix_full,
                                                kdelta, opdm, optimize='optimal').real
            grad_expectation += 1.0 * np.einsum('ab,pq,bp,aq', self.hamiltonian.one_body_tensor, pre_matrix_full,
                                                kdelta, opdm, optimize='optimal').real
            grad_expectation += 1.0 * np.einsum('ijkl,pq,iq,jpkl', self.hamiltonian.two_body_tensor, pre_matrix_full,
                                                kdelta, tpdm, optimize='optimal').real
            grad_expectation +=-1.0 * np.einsum('ijkl,pq,jq,ipkl', self.hamiltonian.two_body_tensor, pre_matrix_full,
                                                kdelta, tpdm, optimize='optimal').real
            grad_expectation +=-1.0 * np.einsum('ijkl,pq,kp,ijlq', self.hamiltonian.two_body_tensor, pre_matrix_full,
                                                kdelta, tpdm, optimize='optimal').real
            grad_expectation += 1.0 * np.einsum('ijkl,pq,lp,ijkq', self.hamiltonian.two_body_tensor, pre_matrix_full,
                                                kdelta, tpdm, optimize='optimal').real
            grad[p] = grad_expectation

        return grad


def generate_hamiltonian(one_body_integrals: np.ndarray,
                         two_body_integrals: np.ndarray,
                         constant: float,
                         EQ_TOLERANCE: Optional[float]=1.0E-12):
    n_qubits = 2 * one_body_integrals.shape[0]
    # Initialize Hamiltonian coefficients.
    one_body_coefficients = np.zeros((n_qubits, n_qubits))
    two_body_coefficients = np.zeros((n_qubits, n_qubits,
                                      n_qubits, n_qubits))
    # Loop through integrals.
    for p in range(n_qubits // 2):
        for q in range(n_qubits // 2):

            # Populate 1-body coefficients. Require p and q have same spin.
            one_body_coefficients[2 * p, 2 * q] = one_body_integrals[
                p, q]
            one_body_coefficients[2 * p + 1, 2 *
                                  q + 1] = one_body_integrals[p, q]
            # Continue looping to prepare 2-body coefficients.
            for r in range(n_qubits // 2):
                for s in range(n_qubits // 2):
                    # Mixed spin
                    two_body_coefficients[2 * p, 2 * q + 1,
                                          2 * r + 1, 2 * s] = (
                            two_body_integrals[p, q, r, s] / 2.)
                    two_body_coefficients[2 * p + 1, 2 * q,
                                          2 * r, 2 * s + 1] = (
                            two_body_integrals[p, q, r, s] / 2.)

                    # Same spin
                    two_body_coefficients[2 * p, 2 * q,
                                          2 * r, 2 * s] = (
                            two_body_integrals[p, q, r, s] / 2.)
                    two_body_coefficients[2 * p + 1, 2 * q + 1,
                                          2 * r + 1, 2 * s + 1] = (
                            two_body_integrals[p, q, r, s] / 2.)

    # Truncate.
    one_body_coefficients[
        np.absolute(one_body_coefficients) < EQ_TOLERANCE] = 0.
    two_body_coefficients[
        np.absolute(two_body_coefficients) < EQ_TOLERANCE] = 0.

    # Cast to InteractionOperator class and return.
    molecular_hamiltonian = InteractionOperator(
        constant, one_body_coefficients, two_body_coefficients)
    return molecular_hamiltonian