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

from typing import Dict, Mapping, Tuple, Union

import attrs
import numpy as np

from recirq.qcqmc import (
    blueprint,
    config,
    data,
    experiment,
    hamiltonian,
    shadow_tomography,
    trial_wf,
)


@attrs.frozen(repr=False)
class OverlapAnalysisParams(data.Params):
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
            k's serve as keys for the groups.
    """

    name: str
    experiment_params: experiment.SimulatedExperimentParams
    k_to_calculate: Tuple[int, ...] = attrs.field(converter=tuple)
    path_prefix: str = ""

    def __attrs_post_init__(self):
        if not isinstance(
            self.experiment_params.blueprint_params, blueprint.BlueprintParamsTrialWf
        ):
            raise ValueError(
                "Experiment must be build from a blueprint derived from a BlueprintParamsTrialWf."
            )

    @property
    def path_string(self) -> str:
        return self.path_prefix + config.OUTDIRS.DEFAULT_ANALYSIS_DIRECTORY + self.name

    def _json_dict_(self):
        return attrs.asdict(self)


@attrs.frozen(eq=False)
class OverlapAnalysisData(data.Data):
    """Container for storing the (shadow tomography-)reconstructed wavefunctions.

    See the factory method `build_analysis_from_dependencies` which will
    properly construct this object from the (real or simulated) experimental bitstrings.

    Args:
        params: The parameters for overlap construction.
        recontstructed_wf_for_k: A mapping from k to the reconstructed wavefunction.
    """

    params: OverlapAnalysisParams
    reconstructed_wf_for_k: Mapping[str, np.ndarray] = attrs.field(
        converter=lambda x: {k: np.asarray(v) for k, v in x.items()}
    )

    def _json_dict_(self):
        return attrs.asdict(self)

    @classmethod
    def build_analysis_from_dependencies(
        cls,
        params: OverlapAnalysisParams,
        *,
        dependencies: Dict[data.Params, data.Data],
    ) -> "OverlapAnalysisData":
        """Builds a OverlapAnalysisData from OverlapAnalysisParams.

        Given the sampled bitstrings from either a real experiment or a
        simulated experiment, this will construct the shadow wavefunction.

        Args:
            params: The parameters for overlap analysis.
            dependencies: The dependencies leading up to this point (i.e. the
                experiment data, blueprint, trial, ...)

        Returns:
            A constructed OverlapAnalysisData object.
        """
        experiment_params = params.experiment_params
        exp = dependencies[experiment_params]
        assert isinstance(exp, experiment.ExperimentData)

        blueprint_params = experiment_params.blueprint_params
        bp = dependencies[blueprint_params]
        assert isinstance(blueprint_params, blueprint.BlueprintParamsTrialWf)
        assert isinstance(bp, blueprint.BlueprintData)

        trial_wf_params = blueprint_params.trial_wf_params
        trial = dependencies[trial_wf_params]
        assert isinstance(trial, trial_wf.TrialWavefunctionData)

        wavefunctions_for_various_k = shadow_tomography.reconstruct_wavefunctions_from_samples(

        return OverlapAnalysisData(
            params=params, reconstructed_wf_for_k=wavefunctions_for_various_k
        )

    def get_variational_energy(
        self,
        *,
        trial_wf_data: trial_wf.TrialWavefunctionData,
        hamiltonian_data: hamiltonian.HamiltonianData,
        k: Union[str, int] = 1,
    ) -> float:
        """Computes the variational energy of the reconstructed wavefunction.

        Args:
            analysis_data: The overlap analysis data.
            trial_wf_data: The trial wavefunction data.
            hamiltonian_data: The hamiltonian data.
            k: k value indexing median of means for shadow tomography.
        """
        wf = self.reconstructed_wf_for_k[str(k)]

        _, _, qubit_ham = trial_wf.get_rotated_hamiltonians(
            hamiltonian_data=hamiltonian_data,
            one_body_basis_change_mat=trial_wf_data.one_body_basis_change_mat,
            mode_qubit_map=trial_wf_data.params.mode_qubit_map,
            ordered_qubits=trial_wf_data.params.qubits_jordan_wigner_ordered,
        )

        energy = (np.conj(wf) @ qubit_ham @ wf / np.linalg.norm(wf) ** 2).item()
        np.testing.assert_almost_equal(energy, energy.real)
        energy = energy.real

        return energy
