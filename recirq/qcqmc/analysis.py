from dataclasses import dataclass, field, InitVar
from types import MappingProxyType
from typing import Dict, Mapping, Tuple, Union

import cirq
import numpy as np

from qc_afqmc.blueprint import BlueprintData, BlueprintParamsTrialWf
from qc_afqmc.experiment import ExperimentData, SimulatedExperimentParams
from qc_afqmc.hamiltonian import HamiltonianData
from qc_afqmc.shadow_tomography import reconstruct_wavefunctions_from_samples
from qc_afqmc.trial_wf import get_rotated_hamiltonians, TrialWavefunctionData
from qc_afqmc.utilities import Data, OUTDIRS, Params


@dataclass(frozen=True, repr=False)
class OverlapAnalysisParams(Params):
    """Class for storing the parameters that specify an OverlapAnalysisData.

    This stage of pipeline is where the data from a (real or simulated) experiment is
    analyzed and we reconstruct the trial wavefunction from the classical shadow.

    Args:
        name: A `Params` name for this experiment.
        experiment_params: Backreference to the `ExperimentParams` preceding this stage.
            We should have experiment_params.blueprint_params be a blueprint for an experiment
            that does shadow tomography on a trial wavefunction.
        k_to_calculate: For shadow tomography, we use a "median of means" approach to statistical
            robustness as explained in Huang et. al. (https://arxiv.org/abs/2002.08953). These
            k's serve as keys for the groups. TODO(wjhuggins): Why can't we just give a range?
            Why are they odd numbers sometimes? Why are we finding the LCM of them?
    """

    name: str
    experiment_params: SimulatedExperimentParams
    k_to_calculate: Tuple[int, ...]

    def __post_init__(self):
        """A little special sauce to make sure that this ends up as a tuple."""
        object.__setattr__(self, "k_to_calculate", tuple(self.k_to_calculate))

        assert isinstance(
            self.experiment_params.blueprint_params, BlueprintParamsTrialWf
        ), "Experiment must be build from a blueprint derived from a BlueprintParamsTrialWf."

    @property
    def path_string(self) -> str:
        return OUTDIRS.DEFAULT_ANALYSIS_DIRECTORY + self.name

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)


@dataclass(frozen=True, eq=False)
class OverlapAnalysisData(Data):
    params: OverlapAnalysisParams
    _reconstructed_wf_for_k: InitVar[Dict[str, np.ndarray]]
    reconstructed_wf_for_k: Mapping[str, np.ndarray] = field(init=False)

    def __post_init__(self, _reconstructed_wf_for_k: Dict[str, np.ndarray]):
        """We need to make some inputs into np.ndarrays if aren't provided that way."""
        reconstructed_wf_for_k = {}

        for k, v in _reconstructed_wf_for_k.items():
            array_v = np.asarray(v)
            array_v.setflags(write=False)
            reconstructed_wf_for_k[k] = np.asarray(array_v)

        reconstructed_wf_for_k = MappingProxyType(reconstructed_wf_for_k)

        object.__setattr__(self, "reconstructed_wf_for_k", reconstructed_wf_for_k)

    def _json_dict_(self):
        to_return = cirq.dataclass_json_dict(self)
        to_return["_reconstructed_wf_for_k"] = to_return["reconstructed_wf_for_k"]
        del to_return["reconstructed_wf_for_k"]

        return to_return


def build_analysis(
    params: OverlapAnalysisParams, *, dependencies: Dict[Params, Data]
) -> OverlapAnalysisData:
    """Builds a OverlapAnalysisData from OverlapAnalysisParams"""
    experiment_params = params.experiment_params
    experiment = dependencies[experiment_params]
    assert isinstance(experiment, ExperimentData)

    blueprint_params = experiment_params.blueprint_params
    blueprint = dependencies[blueprint_params]
    assert isinstance(
        blueprint_params, BlueprintParamsTrialWf
    )  # redundant with assert in __post_init__ but useful for type checking
    assert isinstance(blueprint, BlueprintData)

    trial_wf_params = blueprint_params.trial_wf_params
    trial_wf = dependencies[trial_wf_params]
    assert isinstance(trial_wf, TrialWavefunctionData)

    wavefunctions_for_various_k = reconstruct_wavefunctions_from_samples(
        raw_samples=experiment.raw_samples,
        factorized_cliffords=list(blueprint.resolved_clifford_circuits),
        qubit_partition=blueprint_params.qubit_partition,
        valid_configurations=list(trial_wf_params.bitstrings),
        k_to_calculate=params.k_to_calculate,
        qubits_jordan_wigner_ordered=trial_wf_params.qubits_jordan_wigner_ordered,
        qubits_linearly_connected=trial_wf_params.qubits_linearly_connected,
    )

    return OverlapAnalysisData(params=params, _reconstructed_wf_for_k=wavefunctions_for_various_k)


def get_variational_energy(
    *,
    analysis_data: OverlapAnalysisData,
    trial_wf_data: TrialWavefunctionData,
    hamiltonian_data: HamiltonianData,
    k: Union[str, int] = 1,
) -> float:
    """Gets the variational energy of the reconstructed wavefunction"""
    wf = analysis_data.reconstructed_wf_for_k[str(k)]

    _, _, qubit_ham = get_rotated_hamiltonians(
        hamiltonian_data=hamiltonian_data,
        one_body_basis_change_mat=trial_wf_data.one_body_basis_change_mat,
        mode_qubit_map=trial_wf_data.params.mode_qubit_map,
        ordered_qubits=trial_wf_data.params.qubits_jordan_wigner_ordered,
    )

    energy = (np.conj(wf) @ qubit_ham @ wf / np.linalg.norm(wf) ** 2).item()
    np.testing.assert_almost_equal(energy, energy.real)
    energy = energy.real

    return energy
