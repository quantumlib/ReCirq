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

from typing import Tuple

import cirq

from recirq.qcqmc.blueprint import BlueprintData
from recirq.qcqmc.experiment import SimulatedExperimentParams, get_experimental_metadata, ExperimentData
from recirq.qcqmc.hamiltonian import HamiltonianData
from recirq.qcqmc.trial_wf import TrialWavefunctionData


def test_small_experiment_raw_samples_shape(
    fixture_4_qubit_ham_trial_wf_and_blueprint: Tuple[
        HamiltonianData, TrialWavefunctionData, BlueprintData
    ]
):
    _, _, blueprint_data = fixture_4_qubit_ham_trial_wf_and_blueprint

    simulated_experiment_params = SimulatedExperimentParams(
        name="test_1",
        blueprint_params=blueprint_data.params,
        noise_model_name="None",
        noise_model_params=(0,),
        n_samples_per_clifford=31,
        seed=1,
    )

    experiment = ExperimentData.build_experiment_from_dependencies(
        params=simulated_experiment_params,
        dependencies={blueprint_data.params: blueprint_data},
    )

    raw_samples = experiment.raw_samples

    assert raw_samples.shape == (17, 31, 4)

    exp2 = cirq.read_json(json_text=cirq.to_json(experiment))
    assert exp2 == experiment

def test_get_experimental_metadata():
    md = get_experimental_metadata()
    assert md.get('PST_formatted_date_time') is not None
    assert md.get('iso_formatted_date_time') is not None