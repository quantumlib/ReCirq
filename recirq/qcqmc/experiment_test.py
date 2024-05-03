from typing import Tuple

from qc_afqmc.blueprint import BlueprintData
from qc_afqmc.experiment import build_experiment, SimulatedExperimentParams
from qc_afqmc.hamiltonian import HamiltonianData
from qc_afqmc.trial_wf import TrialWavefunctionData


def test_small_experiment_raw_samples_shape(
    fixture_4_qubit_ham_trial_wf_and_blueprint: Tuple[
        HamiltonianData, TrialWavefunctionData, BlueprintData
    ]
):
    _, _, blueprint_data = fixture_4_qubit_ham_trial_wf_and_blueprint

    simulated_experiment_params = SimulatedExperimentParams(
        name='test_1',
        blueprint_params=blueprint_data.params,
        noise_model_name="None",
        noise_model_params=(0,),
        n_samples_per_clifford=31,
        seed=1,
    )

    experiment = build_experiment(
        params=simulated_experiment_params, dependencies={blueprint_data.params: blueprint_data}
    )

    raw_samples = experiment.raw_samples

    assert raw_samples.shape == (17, 31, 4)
