# Copyright 2024 Google
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

import pathlib
import numpy as np
import h5py
import math

import fqe.wavefunction as fqe_wfn
import fqe.bitstring as fqe_bs

from recirq.qcqmc import analysis, convert_to_ipie


def test_rotate_wavefunction_explicitly():
    n_orb = 4
    n_elec = 4
    n_qubits = 2 * n_orb
    fqe_wf = fqe_wfn.Wavefunction([[n_elec, 0, n_orb]])
    fqe_wf.set_wfn(strategy="hartree-fock")
    one_body_basis_change_mat = np.eye(n_qubits)
    rotated_fqe_wf = convert_to_ipie.rotate_wavefunction_explicitly(
        fqe_wf, one_body_basis_change_mat, n_qubits
    )
    assert np.allclose(
        fqe_wf.get_coeff((n_elec, 0)), rotated_fqe_wf.get_coeff((n_elec, 0))
    )

    one_body_basis_change_mat = np.random.rand(n_orb, n_orb)
    _, one_body_basis_change_mat = np.linalg.eigh(
        one_body_basis_change_mat + one_body_basis_change_mat.T
    )
    rotated_fqe_wf = convert_to_ipie.rotate_wavefunction_explicitly(
        fqe_wf, one_body_basis_change_mat, n_qubits
    )
    assert not np.allclose(
        fqe_wf.get_coeff((n_elec, 0)), rotated_fqe_wf.get_coeff((n_elec, 0))
    )


def test_get_trial_wf_export_data(fixture_8_qubit_ham_and_trial_wf):
    _, trial_wf_data = fixture_8_qubit_ham_and_trial_wf
    export_data = convert_to_ipie._get_trial_wf_export_data(trial_wf_data)
    assert export_data.path == pathlib.Path(trial_wf_data.params.base_path).with_suffix(
        ".h5"
    )
    assert export_data.variational_energy == trial_wf_data.ansatz_energy
    assert export_data.norm == 1.0


def test_get_overlap_export_data(fixture_4_qubit_ham_trial_wf_and_overlap_analysis):
    hamiltonian_data, trial_wf_data, overlap_analysis_data = (
        fixture_4_qubit_ham_trial_wf_and_overlap_analysis
    )
    export_data = convert_to_ipie._get_overlap_export_data(
        overlap_analysis_data, trial_wf_data, hamiltonian_data, k="1"
    )
    assert export_data.path == pathlib.Path(
        overlap_analysis_data.params.base_path
    ).with_suffix(".h5")
    assert (
        export_data.variational_energy
        == overlap_analysis_data.get_variational_energy(
            trial_wf_data=trial_wf_data, hamiltonian_data=hamiltonian_data, k="1"
        )
    )


def test_get_occa_occb_coeff():
    """Test that the occa and occb coefficients are extracted correctly."""
    n_orb = 4
    n_elec = 4
    fqe_wf = fqe_wfn.Wavefunction([[n_elec, 0, n_orb]])
    fqe_wf.set_wfn(strategy="hartree-fock")
    occ_coef = convert_to_ipie.get_occa_occb_coeff(fqe_wf)
    assert occ_coef.occa.shape == (1, n_elec // 2)
    assert occ_coef.occb.shape == (1, n_elec // 2)
    assert occ_coef.coeffs.shape == (1,)

    # Test with a random wavefunction
    fqe_wf = fqe_wfn.Wavefunction([[n_elec, 0, n_orb]])
    n_det_a = math.comb(n_orb, n_elec // 2)
    n_det_b = math.comb(n_orb, n_elec // 2)
    data = {
        (n_elec, 0): np.random.random(size=(n_det_a, n_det_b))
        + 1j * np.random.random(size=(n_det_a, n_det_b))
    }
    fqe_wf.set_wfn(strategy="from_data", raw_data=data)
    occ_coeff = convert_to_ipie.get_occa_occb_coeff(fqe_wf)
    assert len(occ_coeff.occa) == n_det_a * n_det_b
    assert len(occ_coeff.occb) == n_det_a * n_det_b
    assert len(occ_coeff.coeffs) == n_det_a * n_det_b
    sector = fqe_wf.sector((n_elec, 0))
    fqe_graph = sector.get_fcigraph()
    # Round trip: Check that converting back to FQE wavefunction from the ipie
    # occ/coeff representation yields the same FQE wavefunction.
    for coeff, occa, occb in zip(occ_coeff.coeffs, occ_coeff.occa, occ_coeff.occb):
        alpha_str = fqe_bs.reverse_integer_index(occa)
        beta_str = fqe_bs.reverse_integer_index(occb)
        inda = fqe_graph.index_alpha(alpha_str)
        indb = fqe_graph.index_alpha(beta_str)
        assert np.isclose(coeff, fqe_wf.sector((n_elec, 0)).coeff[inda, indb])

    fqe_wf_from_coeff = convert_to_ipie.get_fqe_wf_from_occ_coeff(
        occ_coeff, n_elec, 0, n_orb
    )
    assert np.sum(np.abs(fqe_wf_from_coeff.get_coeff((n_elec, 0)))) > 0
    sector_from_coeff = fqe_wf_from_coeff.sector((n_elec, 0))
    assert np.allclose(sector_from_coeff.coeff, sector.coeff)


def test_save_wavefunction_for_ipie(
    fixture_4_qubit_ham_and_trial_wf,
):
    """Test that the wavefunction is saved correctly to an h5 file."""
    hamiltonian_data, trial_wf_data = fixture_4_qubit_ham_and_trial_wf
    export_data = convert_to_ipie.save_wavefunction_for_ipie(
        hamiltonian_data, trial_wf_data, k="1"
    )
    with h5py.File(export_data.path, "r") as f:
        assert "fci_energy" in f
        assert "ideal_ansatz_energy" in f
        assert "variational energy" in f
        assert "reconstructed_wf_norm" in f
        assert "occa_unrotated" in f
        assert "occb_unrotated" in f
        assert "coeffs_unrotated" in f
        assert "occa_rotated" in f
        assert "occb_rotated" in f
        assert "coeffs_rotated" in f
        assert "basis_change" in f
        assert "k" in f
