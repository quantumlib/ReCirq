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
from typing import List, Optional

import attrs
import fqe.algorithm.low_rank as fqe_alg
import fqe.bitstring as fqe_bs
import fqe.wavefunction as fqe_wfn
import h5py
import numpy as np

from recirq.qcqmc import (
    afqmc_generators,
    analysis,
    config,
    fqe_conversion,
    hamiltonian,
    optimize_wf,
    trial_wf,
)


@attrs.frozen
class IPieExportData:
    """Helper dataclass used in `save_wavefunction_for_ipie`.

    Args:
        rotated_fqe_wf: The FQE wavefunction rotated to some other single particle baiss.
        unrotated_fqe_wf: The FQE wavefunction in the original basis before orbital rotation.
        variataional_energy: The variational energy of the wavefunction.
        norm: The normalization of the wavefunction.
        path: The path to save the wavefunction to.
    """

    rotated_fqe_wf: fqe_wfn.Wavefunction
    unrotated_fqe_wf: fqe_wfn.Wavefunction
    variational_energy: float
    norm: float
    path: pathlib.Path


def rotate_wavefunction_explicitly(
    fqe_wf: fqe_wfn.Wavefunction, one_body_basis_change_mat: np.ndarray, n_qubits: int
) -> fqe_wfn.Wavefunction:
    """Rotates fqe_wf to a different single particle basis.

    Args:
        fqe_wf: The wavefunction to rotate.
        one_body_basis_change_mat: a unitary matrix that describes the restricted
            or unrestricted change of basis.
        n_qubits: The number of qubits.

    Returns:
        The rotated wavefunction.
    """

    if one_body_basis_change_mat.shape[0] == n_qubits:
        # unrestricted wf means we need to reorder the tensor indices.
        alpha_indices = np.asarray([2 * i for i in range(n_qubits // 2)])
        beta_indices = np.asarray([2 * i + 1 for i in range(n_qubits // 2)])

        alpha_mat = one_body_basis_change_mat[:, alpha_indices]
        alpha_mat = alpha_mat[alpha_indices, :]

        beta_mat = one_body_basis_change_mat[:, beta_indices]
        beta_mat = beta_mat[beta_indices, :]

        evolved_wf = fqe_alg.evolve_fqe_givens_sector(fqe_wf, alpha_mat, sector="alpha")
        evolved_wf = fqe_alg.evolve_fqe_givens_sector(
            evolved_wf, beta_mat, sector="beta"
        )

    else:
        evolved_wf = fqe_alg.evolve_fqe_givens(fqe_wf, one_body_basis_change_mat)

    return evolved_wf


def _get_trial_wf_export_data(
    trial_wf_data: trial_wf.TrialWavefunctionData,
) -> IPieExportData:
    """Get the data to export to ipie."""
    params = trial_wf_data.params
    initial_wf = fqe_wfn.Wavefunction([[params.n_elec, 0, params.n_orb]])
    initial_wf.set_wfn(strategy="hartree-fock")
    rotated_fqe_wf, unrotated_fqe_wf = optimize_wf.get_evolved_wf(
        one_body_params=trial_wf_data.one_body_params,
        two_body_params=trial_wf_data.two_body_params,
        wf=initial_wf,
        gate_generators=afqmc_generators.get_pp_plus_gate_generators(
            n_elec=params.n_elec,
            heuristic_layers=params.heuristic_layers,
            do_pp=params.do_pp,
        ),
        n_orb=params.n_orb,
        restricted=params.restricted,
        initial_orbital_rotation=params.initial_orbital_rotation,
    )
    return IPieExportData(
        rotated_fqe_wf=rotated_fqe_wf,
        unrotated_fqe_wf=unrotated_fqe_wf,
        variational_energy=trial_wf_data.ansatz_energy,
        norm=1.0,
        path=trial_wf_data.params.base_path.with_suffix(".h5"),
    )


def _get_overlap_export_data(
    overlap_analysis_data: analysis.OverlapAnalysisData,
    trial_wf_data: trial_wf.TrialWavefunctionData,
    hamiltonian_data: hamiltonian.HamiltonianData,
    threshold: float = 1e-5,
    k: str = "1",
) -> IPieExportData:
    """Get the data to export to ipie from the shadow tomography analysis."""
    sorted_mapping = list(trial_wf_data.params.mode_qubit_map.items())
    sorted_mapping.sort(key=lambda x: x[1])

    integer_fermion_qubit_map = {}
    for i, (old_index, _) in enumerate(sorted_mapping):
        integer_fermion_qubit_map[old_index.openfermion_standard_index] = i

    reconstructed_wf = np.copy(overlap_analysis_data.reconstructed_wf_for_k[k])
    reconstructed_wf[np.abs(reconstructed_wf) < threshold] = 0.0
    norm = float(np.linalg.norm(reconstructed_wf))
    reconstructed_wf /= norm

    unrotated_fqe_wf = fqe_wfn.Wavefunction(
        [[trial_wf_data.params.n_elec, 0, trial_wf_data.params.n_orb]]
    )
    assert isinstance(unrotated_fqe_wf, fqe_wfn.Wavefunction)

    fqe_conversion.fill_in_wfn_from_cirq(
        unrotated_fqe_wf, reconstructed_wf, integer_fermion_qubit_map
    )

    rotated_fqe_wf = rotate_wavefunction_explicitly(
        unrotated_fqe_wf,
        trial_wf_data.one_body_basis_change_mat,
        trial_wf_data.params.n_qubits,
    )

    unrotated_fqe_wf.scale(norm)
    rotated_fqe_wf.scale(norm)

    return IPieExportData(
        rotated_fqe_wf=rotated_fqe_wf,
        unrotated_fqe_wf=unrotated_fqe_wf,
        variational_energy=overlap_analysis_data.get_variational_energy(
            trial_wf_data=trial_wf_data,
            hamiltonian_data=hamiltonian_data,
            k=k,
        ),
        norm=norm,
        path=overlap_analysis_data.params.base_path.with_suffix(".h5"),
    )


@attrs.frozen(eq=False)
class OccCoef:
    occa: np.ndarray
    occb: np.ndarray
    coeffs: np.ndarray
    """A container for the wavefunction bitstrings and coefficients.

    Args:
        occa: The spin-alpha (up) determinants. Each entry of occa is an array
            of integers representing which orbitals are occupied in that
            determinant.
        occb: The spin-beta (down) determinants. Each entry of occa is an array
            of integers representing which orbitals are occupied in that
            determinant.
        coeffs: The coefficient for each determinant. 
    """

    def to_h5_dict(self, suffix: str = ""):
        return {
            f"occa{suffix}": self.occa,
            f"occb{suffix}": self.occb,
            f"coeffs{suffix}": self.coeffs,
        }


def get_occa_occb_coeff(
    fqe_wf: fqe_wfn.Wavefunction, threshold: float = 1e-5
) -> OccCoef:
    """A helper function to organize data for an AFQMC code to ingest.

    Args:
        fqe_wf: The FQE wavefunction.
        threshold: An optional threshold below which to set coefficients to zero.

    Returns:
        The AFQMC wavefunction
    """

    def _get_sector_data(sector, threshold, occa_list, occb_list, coeffs):
        for inda in range(sector._core.lena()):
            alpha_str = sector._core.string_alpha(inda)
            for indb in range(sector._core.lenb()):
                if np.abs(sector.coeff[inda, indb]) > threshold:
                    alpha_str = sector._core.string_alpha(inda)
                    beta_str = sector._core.string_beta(indb)
                    coeff = sector.coeff[inda, indb]

                    occa_list.append(fqe_bs.integer_index(alpha_str))
                    occb_list.append(fqe_bs.integer_index(beta_str))
                    coeffs.append(coeff)

    occa_list: List[np.ndarray] = []
    occb_list: List[np.ndarray] = []
    coeffs: List[np.ndarray] = []

    for sector_key in fqe_wf.sectors():
        sector = fqe_wf.sector(sector_key)
        _get_sector_data(sector, threshold, occa_list, occb_list, coeffs)

    return OccCoef(
        occa=np.asarray(occa_list),
        occb=np.asarray(occb_list),
        coeffs=np.asarray(coeffs),
    )


def get_fqe_wf_from_occ_coeff(
    occ_coeff: OccCoef, n_elec: int, ms: int, n_orb: int, threshold: float = 1e-5
) -> fqe_wfn.Wavefunction:
    """A helper function to map an AFQMC wavefunction to FQE.

    Args:
        occ_coeff: OccCoef object storing list of coefficients and occupation strings.
        n_elec: Number of electrons.
        ms: spin polarization.
        n_orb: number of orbitals.
        threshold: ci coefficient threshold. A coefficient whose absolute value
            below this value is considered zero.
    """

    def _set_sector_data(sector, threshold, occa_list, occb_list, coeffs):
        fqe_graph = sector.get_fcigraph()
        for idet, (occa, occb) in enumerate(zip(occa_list, occb_list)):
            alpha_str = fqe_bs.reverse_integer_index(occa)
            beta_str = fqe_bs.reverse_integer_index(occb)
            inda = fqe_graph.index_alpha(alpha_str)
            indb = fqe_graph.index_alpha(beta_str)
            if np.abs(coeffs[idet]) > threshold:
                sector.coeff[inda, indb] = coeffs[idet]

    fqe_wf = fqe_wfn.Wavefunction([[n_elec, ms, n_orb]])

    for sector_key in fqe_wf.sectors():
        sector = fqe_wf.sector(sector_key)
        _set_sector_data(
            sector, threshold, occ_coeff.occa, occ_coeff.occb, occ_coeff.coeffs
        )

    return fqe_wf


def _print_wfn_export_data(
    xd: IPieExportData, *, fci_energy: float, ansatz_energy: float
):
    """Print some information about the wavefunction."""
    print("Rotated Wavefunction")
    xd.rotated_fqe_wf.print_wfn()

    print("Unrotated Wavefunction")
    xd.unrotated_fqe_wf.print_wfn()

    if xd.variational_energy is not None and fci_energy is not None:
        fci_diff = xd.variational_energy - fci_energy
    if xd.variational_energy is not None and ansatz_energy is not None:
        ansatz_diff = xd.variational_energy - ansatz_energy

    print(
        f"FCI Energy: {fci_energy} \n"
        f"Ansatz Energy: {ansatz_energy} \n"
        f"Variational Energy: {xd.variational_energy} \n"
        f"Error (vs fci): {fci_diff} \n"
        f"Error (vs ansatz): {ansatz_diff} \n"
        f"Norm {xd.norm}"
    )


def save_wavefunction_for_ipie(
    hamiltonian_data: hamiltonian.HamiltonianData,
    trial_wf_data: trial_wf.TrialWavefunctionData,
    overlap_analysis_data: Optional[analysis.OverlapAnalysisData] = None,
    threshold: float = 1e-5,
    k: str = "1",
    do_print: bool = config.VERBOSE_EXECUTION,
):
    """A utility function that saves data for iPie.

    Args:
        hamiltonian_data: The hamiltonian data.
        trial_wf_data: The trial wavefunction data.
        overlap_analysis_data: The overlap analysis data.
        threshold: Threshold for zeroing out small amplitude ci coeffiencts.
        k: k value indexing median of means for shadow tomography.
        do_print: Print some information about the wavefunction.

    Returns:
        The AFQMC wavefunction
    """

    xd: IPieExportData
    if overlap_analysis_data is not None:
        xd = _get_overlap_export_data(
            overlap_analysis_data,
            trial_wf_data,
            hamiltonian_data,
            threshold=threshold,
            k=k,
        )
    else:
        xd = _get_trial_wf_export_data(trial_wf_data)

    occ_unrot = get_occa_occb_coeff(xd.unrotated_fqe_wf, threshold)
    occ_rot = get_occa_occb_coeff(xd.rotated_fqe_wf, threshold)
    fci_energy = trial_wf_data.fci_energy
    ansatz_energy = trial_wf_data.ansatz_energy

    if do_print:
        _print_wfn_export_data(xd, fci_energy=fci_energy, ansatz_energy=ansatz_energy)

    if not xd.path.parent.is_dir():
        xd.path.parent.mkdir(parents=True)

    with h5py.File(xd.path, "w") as f:
        if fci_energy is not None:
            f["fci_energy"] = fci_energy
        if ansatz_energy is not None:
            f["ideal_ansatz_energy"] = ansatz_energy
        if xd.variational_energy is not None:
            f["variational energy"] = xd.variational_energy
        f["reconstructed_wf_norm"] = xd.norm
        f.update(occ_unrot.to_h5_dict(suffix="_unrotated"))
        f.update(occ_rot.to_h5_dict(suffix="_rotated"))
        f["basis_change"] = trial_wf_data.one_body_basis_change_mat
        f["k"] = int(k)

    return xd
