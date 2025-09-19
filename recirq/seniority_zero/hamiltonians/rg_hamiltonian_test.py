# Copyright 2023 Google
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

import os

os.environ['MKL_NUM_THREADS'] = "6"

# pylint: disable=wrong-import-position
import fqe
import numpy as np
import openfermion as of
from scipy.sparse import coo_matrix

from recirq.seniority_zero.hamiltonians.rg_hamiltonian import RGDOCI


def get_doci_projector(n_qubits, n_target):
    """Get the projector on to the doubly occupied space

    :param int n_qubits: number of qubits (or fermionic modes)
    :param int n_target: number of electrons total
    :return: coo_matrix that is 1 along diagonal elements that correspond
             to doci terms.
    """
    n_row_idx = []
    for ii in range(2**n_qubits):
        ket = np.binary_repr(ii, width=n_qubits)
        keta = ket[::2]
        ketb = ket[1::2]
        res = int(keta, 2) ^ int(ketb, 2)
        if np.isclose(res, 0) and ket.count('1') == n_target:
            n_row_idx.append(ii)
    n_projector = coo_matrix(
        ([1] * len(n_row_idx), (n_row_idx, n_row_idx)), shape=(2**n_qubits, 2**n_qubits)
    )
    return n_projector, n_row_idx


def num_op_projector(n_qubits, n_target):
    n_row_idx = []
    for ii in range(2**n_qubits):
        ket = np.binary_repr(ii, width=n_qubits)
        if ket.count('1') == n_target:
            n_row_idx.append(ii)
    n_projector = coo_matrix(
        ([1] * len(n_row_idx), (n_row_idx, n_row_idx)), shape=(2**n_qubits, 2**n_qubits)
    )
    return n_projector, n_row_idx


def test_rg_fermion_qubit_spectrum():
    rgham_instance = RGDOCI(0.5, 4)
    nmo = rgham_instance.norbs
    fermion_ham = rgham_instance.get_fermion_ham()
    dense_rg_fham = of.get_sparse_operator(fermion_ham).toarray().real
    s0_proj, s0_ci_basis = get_doci_projector(2 * rgham_instance.norbs, rgham_instance.norbs)
    # take just the seniority zero basis
    doci_ham_test = dense_rg_fham[:, s0_ci_basis]
    doci_ham_test = doci_ham_test[s0_ci_basis, :]
    # solve in the S0 determinant space
    # we do this for numerical stability
    proj_doci_eigs, proj_doci_vecs = np.linalg.eigh(doci_ham_test)

    # put wavefuction on S0 space back into full space
    full_space_wf = np.zeros((4**nmo), dtype=np.complex128)
    for idx in range(len(s0_ci_basis)):
        full_space_wf[s0_ci_basis[idx]] = proj_doci_vecs[idx, 0]
    # print wf
    fqe_doci = fqe.from_cirq(full_space_wf.flatten(), thresh=1.0e-12)
    fqe_doci.print_wfn()
    fqe_fham = fqe.get_hamiltonian_from_openfermion(fermion_ham)
    assert np.isclose(fqe_doci.expectationValue(fqe_fham), proj_doci_eigs[0])

    fqe_civec = fqe_doci.sector((nmo, 0)).coeff.diagonal()
    graph = fqe_doci.sector((nmo, 0)).get_fcigraph()
    test_qubit_civec = np.zeros((2**nmo, 1), dtype=np.complex128)
    for ket_id, xpos in graph.index_beta_all().items():
        ket_val = int(np.binary_repr(ket_id, width=nmo)[::-1], 2)
        test_qubit_civec[ket_val] = fqe_civec[xpos]

    qubit_ham, constant = rgham_instance.get_qubit_hamiltonian()
    dense_rg_qham = of.get_sparse_operator(qubit_ham).toarray()
    assert np.isclose(
        (test_qubit_civec.T @ dense_rg_qham @ test_qubit_civec)[0, 0] + constant, proj_doci_eigs[0]
    )

    # need to check in the number operator basis
    n_proj, n_proj_basis = num_op_projector(nmo, nmo // 2)
    dense_rg_qham = dense_rg_qham[:, n_proj_basis]
    dense_rg_qham = dense_rg_qham[n_proj_basis, :]

    w, v = np.linalg.eigh(dense_rg_qham)
    v_full = np.zeros((2**nmo, 1), dtype=np.complex128)
    for idx, proj_basis_idx in enumerate(n_proj_basis):
        v_full[proj_basis_idx, 0] = v[idx, 0]
    assert np.allclose(proj_doci_eigs[:10], w[:10] + constant)

    full_spin_orb_cirq_wf = np.zeros((4**nmo, 1), dtype=np.complex128)
    for ii in range(2**nmo):
        ket_a = np.binary_repr(ii, width=nmo)
        ket_b = np.binary_repr(ii, width=nmo)
        ket = [kk for jj in zip(ket_a, ket_b) for kk in jj]
        fidx = int("".join(ket), 2)
        full_spin_orb_cirq_wf[fidx, 0] = v_full[ii, 0]

    fqe_doci = fqe.from_cirq(full_spin_orb_cirq_wf.flatten(), thresh=1.0e-12)
    fqe_doci.print_wfn()


def test_rg_qubit_ov_spectrum():
    rgham_instance = RGDOCI(0.5, 4)
    nmo = rgham_instance.norbs

    qubit_ham, constant = rgham_instance.get_qubit_hamiltonian()
    dense_rg_qham = of.get_sparse_operator(qubit_ham).toarray()
    n_proj, n_proj_basis = num_op_projector(nmo, nmo // 2)
    dense_rg_qham = dense_rg_qham[:, n_proj_basis]
    dense_rg_qham = dense_rg_qham[n_proj_basis, :]
    w, v = np.linalg.eigh(dense_rg_qham)

    v_full = np.zeros((2**nmo, 1), dtype=np.complex128)
    for idx, proj_basis_idx in enumerate(n_proj_basis):
        v_full[proj_basis_idx, 0] = v[idx, 0]
    full_spin_orb_cirq_wf = np.zeros((4**nmo, 1), dtype=np.complex128)
    for ii in range(2**nmo):
        ket_a = np.binary_repr(ii, width=nmo)
        ket_b = np.binary_repr(ii, width=nmo)
        ket = [kk for jj in zip(ket_a, ket_b) for kk in jj]
        fidx = int("".join(ket), 2)
        full_spin_orb_cirq_wf[fidx, 0] = v_full[ii, 0]
    fqe_doci = fqe.from_cirq(full_spin_orb_cirq_wf.flatten(), thresh=1.0e-12)
    fqe_doci_opdm, _ = fqe_doci.sector((nmo, 0)).get_openfermion_rdms()

    qubit_ham_ov, constant = rgham_instance.get_qubit_hamiltonian_ov_basis()
    dense_rg_qham = of.get_sparse_operator(qubit_ham_ov).toarray()
    dense_rg_qham = dense_rg_qham[:, n_proj_basis]
    dense_rg_qham = dense_rg_qham[n_proj_basis, :]
    ww, vv = np.linalg.eigh(dense_rg_qham)
    assert np.allclose(ww, w)

    v_full_ov = np.zeros((2**nmo, 1), dtype=np.complex128)
    for idx, proj_basis_idx in enumerate(n_proj_basis):
        v_full_ov[proj_basis_idx, 0] = vv[idx, 0]
    full_spin_orb_cirq_wf_ov = np.zeros((4**nmo, 1), dtype=np.complex128)
    for ii in range(2**nmo):
        ket_a = np.binary_repr(ii, width=nmo)
        ket_b = np.binary_repr(ii, width=nmo)
        ket = [kk for jj in zip(ket_a, ket_b) for kk in jj]
        fidx = int("".join(ket), 2)
        full_spin_orb_cirq_wf_ov[fidx, 0] = v_full_ov[ii, 0]
    fqe_doci_ov = fqe.from_cirq(full_spin_orb_cirq_wf_ov.flatten(), thresh=1.0e-12)
    fqe_doci_opdm_ov, _ = fqe_doci_ov.sector((nmo, 0)).get_openfermion_rdms()
    occs = np.diagonal(fqe_doci_opdm)[::2]
    occs_ov = np.diagonal(fqe_doci_opdm_ov)[::2]
    assert np.allclose(occs, np.hstack((occs_ov[::2], occs_ov[1::2])))


def test_antisym_integrals():
    rgham_instance = RGDOCI(0.5, 6)
    fham = rgham_instance.get_fermion_ham()
    h1, ah2 = rgham_instance.get_cc_spinorbs()
    ncr_antisymm_fham = of.InteractionOperator(0, h1.astype(float), ah2.astype(float))
    test1 = of.normal_ordered(of.get_fermion_operator(ncr_antisymm_fham))
    test2 = of.normal_ordered(fham)
    assert test1 == test2  # check if equivalent to the original HamiltonianA
