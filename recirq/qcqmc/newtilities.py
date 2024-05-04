import dataclasses
from os import PathLike
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import cirq
import fqe
import h5py
import numpy as np
from filelock import BaseFileLock, FileLock
from fqe.algorithm.low_rank import evolve_fqe_givens, evolve_fqe_givens_sector
from fqe.bitstring import integer_index
from fqe.wavefunction import Wavefunction as FqeWavefunction

from recirq.qcqmc.analysis import (
    build_analysis,
    get_variational_energy,
    OverlapAnalysisData,
    OverlapAnalysisParams,
)
from recirq.qcqmc.blueprint import (
    BlueprintParamsRobustShadow,
    BlueprintParamsTrialWf,
    build_blueprint,
)
from recirq.qcqmc.experiment import build_experiment, SimulatedExperimentParams
from recirq.qcqmc.fqe_conversion_utils import fill_in_wfn_from_cirq
from recirq.qcqmc.hamiltonian import (
    build_hamiltonian_from_file,
    build_hamiltonian_from_pyscf,
    HamiltonianData,
    LoadFromFileHamiltonianParams,
    PyscfHamiltonianParams,
)
from recirq.qcqmc.trial_wf import (
    _get_fqe_wavefunctions,
    build_pp_plus_trial_wavefunction,
    PerfectPairingPlusTrialWavefunctionParams,
    TrialWavefunctionData,
)
from recirq.qcqmc.utilities import Data, Params

VERBOSE_EXECUTION = True


def nested_params_iter(params: Params, yield_self: bool = True) -> Iterator[Params]:
    for thing in tuple(
        getattr(params, field.name) for field in dataclasses.fields(params)
    ):
        if isinstance(thing, Params):
            yield from nested_params_iter(thing)

    if yield_self:
        yield params


def load_dependencies(
    params: Params,
    *,
    run_if_file_not_found: bool = False,
    save_if_fall_back_to_run: bool = False,
):
    dependencies: Dict[Params, Data] = {}
    for dependency in nested_params_iter(params, yield_self=False):
        dependencies[dependency] = load_data(
            dependency,
            run_if_file_not_found=run_if_file_not_found,
            save_if_fall_back_to_run=save_if_fall_back_to_run,
        )

    return dependencies


def run(
    params: Params,
    *,
    save: bool = True,
    load_if_exists: bool = True,
    run_dependencies_if_necessary: bool = False,
    do_print: bool = VERBOSE_EXECUTION,
    lock: Optional[BaseFileLock] = None,
) -> Data:
    if load_if_exists and params.base_path.with_suffix(".gzip").exists():
        return load_data(params)

    dependencies = load_dependencies(
        params,
        run_if_file_not_found=run_dependencies_if_necessary,
        save_if_fall_back_to_run=save,
    )

    if lock is None:
        params.base_path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(params.base_path.with_suffix(".lock"))

    with lock:
        if do_print:
            print("\nRunning : ")
            print(params)
            print(f"File: {params.base_path}")

        if isinstance(params, LoadFromFileHamiltonianParams):
            data: Data = build_hamiltonian_from_file(params=params)

        elif isinstance(params, PyscfHamiltonianParams):
            data = build_hamiltonian_from_pyscf(params=params)

        elif isinstance(params, PerfectPairingPlusTrialWavefunctionParams):
            data = build_pp_plus_trial_wavefunction(
                params=params, dependencies=dependencies, do_print=do_print
            )

        elif isinstance(params, (BlueprintParamsTrialWf, BlueprintParamsRobustShadow)):
            data = build_blueprint(params, dependencies=dependencies)

        elif isinstance(params, SimulatedExperimentParams):
            data = build_experiment(params, dependencies=dependencies)

        elif isinstance(params, OverlapAnalysisParams):
            data = build_analysis(params, dependencies=dependencies)

        else:
            raise NotImplementedError(
                f"Generating data for {params} is not implemented."
            )

        if save:
            save_data(data, lock=lock)
        else:
            print("-" * 80)

    return data


def save_data(
    data: Data,
    *,
    lock: Optional[BaseFileLock] = None,
    do_print: bool = VERBOSE_EXECUTION,
) -> None:
    """A utility function that calls a more specific save method."""
    params = data.params
    assert isinstance(
        data, Data
    ), f"data must inherit from of type Data but is {type(data)}"
    assert isinstance(params, Params), f"data has invalid params of type {type(params)}"

    if lock is None:
        lock = FileLock(params.base_path.with_suffix(".lock"))
    with lock:
        print("\nSaving : ")
        print(params)
        file_path = params.base_path.with_suffix(".gzip")
        print(f"File: {file_path}")

        # TODO: clean up this special wavefunction saving junk (#66).
        if isinstance(data, TrialWavefunctionData):
            hamiltonian_data = load_data(data.params.hamiltonian_params)
            assert isinstance(hamiltonian_data, HamiltonianData)
            save_wavefunction_for_ipie(
                trial_wf_data=data, hamiltonian_data=hamiltonian_data
            )
        elif isinstance(data, OverlapAnalysisData):
            save_overlap_analysis_wavefunctions_for_ipie(data)

        cirq.to_json_gzip(data, file_or_fn=file_path)

        print("-" * 80)
        return None


def load_data(
    params: Params,
    *,
    run_if_file_not_found: bool = False,
    save_if_fall_back_to_run: bool = False,
    do_print: bool = VERBOSE_EXECUTION,
) -> Data:
    """A utility function that calls a more specific load method."""
    base_path = params.base_path

    lock_path = base_path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(lock_path)
    results = None

    with lock:
        if base_path.with_suffix(".gzip").exists():
            if do_print:
                print("\nLoading : ")
                print(params)
                print(f"File: {base_path}")

            results = cirq.read_json_gzip(base_path.with_suffix(".gzip"))

        elif run_if_file_not_found:
            results = run(
                params,
                save=save_if_fall_back_to_run,
                run_dependencies_if_necessary=True,
                lock=lock,
            )

        else:
            raise ValueError(f"Gzip file not found for {params} at {base_path}.")

        if params != results.params:
            print("params: \n")
            print(params)

            print("results.params: \n")
            print(results.params)

            raise ValueError(
                f"Loading invariant failure {params.hash_key} {results.params.hash_key}"
            )

        assert isinstance(results, Data)

    return results


def load_data_from_gzip_file(
    path: Union[str, PathLike], *, do_print: bool = VERBOSE_EXECUTION
) -> Data:
    """A utility function that calls a more specific load method."""
    path = Path(path)

    if path.suffix != ".gzip":
        raise ValueError(f"{path} should be a .gzip file")
    if not path.exists():
        raise FileNotFoundError(f"No file found at {path}")

    lock_path = path.with_suffix(".lock")
    lock = FileLock(lock_path)

    with lock:
        if do_print:
            print("\nLoading : ")
            print(f"File: {path}")

        results = cirq.read_json_gzip(path)
        assert isinstance(results, Data)

        params = results.params

        if do_print:
            print("Data loaded for params:\n")
            print(params)

    return results


def try_delete_data_file(params: Params) -> None:
    base_path = params.base_path

    for suffix in [".h5", ".chk", ".gzip", ".lock"]:
        path = base_path.with_suffix(suffix)

        if path.exists():
            path.unlink()
            print(f"Deleted {path}")


def save_overlap_analysis_wavefunctions_for_ipie(data: OverlapAnalysisData):
    """A utility function that saves that data for an experiment."""
    assert isinstance(data, OverlapAnalysisData)

    blueprint_params = data.params.experiment_params.blueprint_params
    assert isinstance(
        blueprint_params, BlueprintParamsTrialWf
    ), f"blueprint_params must be of type BlueprintParamsTrialWf but are {type(blueprint_params)}"

    trial_wf_data = load_data(blueprint_params.trial_wf_params)
    assert isinstance(trial_wf_data, TrialWavefunctionData)

    hamiltonian_data = load_data(blueprint_params.trial_wf_params.hamiltonian_params)
    assert isinstance(hamiltonian_data, HamiltonianData)

    save_wavefunction_for_ipie(
        hamiltonian_data=hamiltonian_data,
        trial_wf_data=trial_wf_data,
        overlap_analysis_data=data,
    )


@dataclasses.dataclass
class _IPieExportData:
    """Helper dataclass used in `save_wavefunction_for_ipie`."""

    rotated_fqe_wf: FqeWavefunction
    unrotated_fqe_wf: FqeWavefunction
    variational_energy: float
    norm: float
    path: Path


def _get_trial_wf_export_data(trial_wf_data: TrialWavefunctionData) -> _IPieExportData:
    rotated_fqe_wf, unrotated_fqe_wf = _get_fqe_wavefunctions(
        one_body_params=trial_wf_data.one_body_params,
        two_body_params=trial_wf_data.two_body_params,
        n_orb=trial_wf_data.params.n_orb,
        n_elec=trial_wf_data.params.n_elec,
        heuristic_layers=trial_wf_data.params.heuristic_layers,
        do_pp=trial_wf_data.params.do_pp,
        restricted=trial_wf_data.params.restricted,
        initial_orbital_rotation=trial_wf_data.params.initial_orbital_rotation,
    )
    return _IPieExportData(
        rotated_fqe_wf=rotated_fqe_wf,
        unrotated_fqe_wf=unrotated_fqe_wf,
        variational_energy=trial_wf_data.ansatz_energy,
        norm=1.0,
        path=trial_wf_data.params.base_path.with_suffix(".h5"),
    )


def _get_overlap_export_data(
    overlap_analysis_data: OverlapAnalysisData,
    trial_wf_data: TrialWavefunctionData,
    hamiltonian_data: HamiltonianData,
    threshold: float = 1e-5,
    k: str = "1",
) -> _IPieExportData:
    sorted_mapping = list(trial_wf_data.params.mode_qubit_map.items())
    sorted_mapping.sort(key=lambda x: x[1])

    integer_fermion_qubit_map = {}
    for i, (old_index, _) in enumerate(sorted_mapping):
        integer_fermion_qubit_map[old_index.openfermion_standard_index] = i

    reconstructed_wf = np.copy(overlap_analysis_data.reconstructed_wf_for_k[k])
    reconstructed_wf[np.abs(reconstructed_wf) < threshold] = 0.0
    norm = float(np.linalg.norm(reconstructed_wf))
    reconstructed_wf /= norm

    unrotated_fqe_wf = fqe.Wavefunction(
        [[trial_wf_data.params.n_elec, 0, trial_wf_data.params.n_orb]]
    )
    assert isinstance(unrotated_fqe_wf, FqeWavefunction)

    fill_in_wfn_from_cirq(unrotated_fqe_wf, reconstructed_wf, integer_fermion_qubit_map)

    rotated_fqe_wf = rotate_wavefunction_explicitly(
        unrotated_fqe_wf,
        trial_wf_data.one_body_basis_change_mat,
        trial_wf_data.params.n_qubits,
    )

    unrotated_fqe_wf.scale(norm)
    rotated_fqe_wf.scale(norm)

    return _IPieExportData(
        rotated_fqe_wf=rotated_fqe_wf,
        unrotated_fqe_wf=unrotated_fqe_wf,
        variational_energy=get_variational_energy(
            analysis_data=overlap_analysis_data,
            trial_wf_data=trial_wf_data,
            hamiltonian_data=hamiltonian_data,
            k=k,
        ),
        norm=norm,
        path=overlap_analysis_data.params.base_path.with_suffix(".h5"),
    )


def print_wfn_export_data(
    xd: _IPieExportData, *, fci_energy: float, ansatz_energy: float
):
    print("Rotated Wavefunction")
    xd.rotated_fqe_wf.print_wfn()

    print("Unrotated Wavefunction")
    xd.unrotated_fqe_wf.print_wfn()

    if xd.variational_energy is not None and fci_energy is not None:
        fci_diff = xd.variational_energy - fci_energy
    else:
        fci_diff = None
    if xd.variational_energy is not None and ansatz_energy is not None:
        ansatz_diff = xd.variational_energy - ansatz_energy
    else:
        ansatz_diff = None

    print(
        f"FCI Energy: {fci_energy} \n"
        f"Ansatz Energy: {ansatz_energy} \n"
        f"Variational Energy: {xd.variational_energy} \n"
        f"Error (vs fci): {fci_diff} \n"
        f"Error (vs ansatz): {ansatz_diff} \n"
        f"Norm {xd.norm}"
    )


def save_wavefunction_for_ipie(
    hamiltonian_data: HamiltonianData,
    trial_wf_data: TrialWavefunctionData,
    overlap_analysis_data: Optional[OverlapAnalysisData] = None,
    threshold: float = 1e-5,
    do_print: bool = VERBOSE_EXECUTION,
    k: str = "1",
):
    """A utility function that saves data for iPie."""

    xd: _IPieExportData
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
        print_wfn_export_data(xd, fci_energy=fci_energy, ansatz_energy=ansatz_energy)

    f = h5py.File(xd.path, "w")
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

    f.close()
    return xd


@dataclasses.dataclass
class OccCoef:
    occa: np.ndarray
    occb: np.ndarray
    coeffs: np.ndarray

    def to_h5_dict(self, suffix: str = ""):
        return {
            f"occa{suffix}": self.occa,
            f"occb{suffix}": self.occb,
            f"coeffs{suffix}": self.coeffs,
        }


def get_occa_occb_coeff(fqe_wf: FqeWavefunction, threshold: float = 1e-5) -> OccCoef:
    """A helper function to organize data for an AFQMC code to ingest."""

    def _get_sector_data(sector, threshold, occa_list, occb_list, coeffs):
        for inda in range(sector._core.lena()):
            alpha_str = sector._core.string_alpha(inda)
            for indb in range(sector._core.lenb()):
                if np.abs(sector.coeff[inda, indb]) > threshold:
                    alpha_str = sector._core.string_alpha(inda)
                    beta_str = sector._core.string_beta(indb)
                    coeff = sector.coeff[inda, indb]

                    occa_list.append(integer_index(alpha_str))
                    occb_list.append(integer_index(beta_str))
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
) -> FqeWavefunction:
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
            alpha_str = fqe.bitstring.reverse_integer_index(occa)
            beta_str = fqe.bitstring.reverse_integer_index(occb)
            inda = fqe_graph.index_alpha(alpha_str)
            indb = fqe_graph.index_alpha(beta_str)
            if np.abs(coeffs[idet]) > threshold:
                sector.coeff[inda, indb] = coeffs[idet]

    fqe_wf = fqe.Wavefunction([[n_elec, ms, n_orb]])

    for sector_key in fqe_wf.sectors():
        sector = fqe_wf.sector(sector_key)
        _set_sector_data(
            sector, threshold, occ_coeff.occa, occ_coeff.occb, occ_coeff.coeffs
        )

    return fqe_wf


def rotate_wavefunction_explicitly(
    fqe_wf: FqeWavefunction, one_body_basis_change_mat: np.ndarray, n_qubits: int
) -> FqeWavefunction:
    """Rotates fqe_wf to a different single particle basis.

    one_body_basis_change_mat is a unitary matrix that describes the restricted
    or unrestricted change of basis.
    """

    if one_body_basis_change_mat.shape[0] == n_qubits:
        restricted = False

        # unrestricted wf means we need to reorder the tensor indices.
        alpha_indices = np.asarray([2 * i for i in range(n_qubits // 2)])
        beta_indices = np.asarray([2 * i + 1 for i in range(n_qubits // 2)])

        alpha_mat = one_body_basis_change_mat[:, alpha_indices]
        alpha_mat = alpha_mat[alpha_indices, :]

        beta_mat = one_body_basis_change_mat[:, beta_indices]
        beta_mat = beta_mat[beta_indices, :]

        evolved_wf = evolve_fqe_givens_sector(fqe_wf, alpha_mat, sector="alpha")
        evolved_wf = evolve_fqe_givens_sector(evolved_wf, beta_mat, sector="beta")

    else:
        restricted = True

        evolved_wf = evolve_fqe_givens(fqe_wf, one_body_basis_change_mat)

    return evolved_wf
