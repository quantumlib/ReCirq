from typing import Dict, Tuple

import cirq
import numpy as np
import pytest

from recirq.qcqmc.analysis import (
    OverlapAnalysisParams,
    build_analysis,
    get_variational_energy,
)
from recirq.qcqmc.blueprint import BlueprintData, BlueprintParamsTrialWf
from recirq.qcqmc.data import Data, Params
from recirq.qcqmc.experiment import ExperimentData, SimulatedExperimentParams
from recirq.qcqmc.hamiltonian import HamiltonianData
from recirq.qcqmc.trial_wf import TrialWavefunctionData


@pytest.mark.slow
@pytest.mark.parametrize(
    "blueprint_seed, n_cliffords, n_samples_per_clifford, target_error",
    [(1, 10, 1, 1e-1), (1, 100, 10, 2e-2), (1, 200, 10, 1e-2)],
)
def test_small_experiment_partition_match_reversed_partition_same_seed(
    fixture_4_qubit_ham_and_trial_wf: Tuple[HamiltonianData, TrialWavefunctionData],
    blueprint_seed: int,
    n_cliffords: int,
    n_samples_per_clifford: int,
    target_error: float,
):
    hamiltonian_data, trial_wf = fixture_4_qubit_ham_and_trial_wf
    trial_wf_params = trial_wf.params
    dependencies: Dict[Params, Data] = {
        trial_wf_params: trial_wf,
        hamiltonian_data.params: hamiltonian_data,
    }

    blueprint_params_1 = BlueprintParamsTrialWf(
        name="blueprint_test_analysis",
        trial_wf_params=trial_wf_params,
        n_cliffords=n_cliffords,
        qubit_partition=(
            tuple(qubit for qubit in trial_wf_params.qubits_jordan_wigner_ordered),
        ),
        seed=blueprint_seed,
    )
    blueprint_params_2 = BlueprintParamsTrialWf(
        name="blueprint_test_analysis",
        trial_wf_params=trial_wf_params,
        n_cliffords=n_cliffords,
        qubit_partition=(
            tuple(
                qubit
                for qubit in reversed(trial_wf_params.qubits_jordan_wigner_ordered)
            ),
        ),
        seed=blueprint_seed,
    )

    blueprint_1 = BlueprintData.build_blueprint_from_dependencies(
        blueprint_params_1, dependencies=dependencies
    )
    blueprint_2 = BlueprintData.build_blueprint_from_dependencies(
        blueprint_params_2, dependencies=dependencies
    )
    dependencies[blueprint_params_1] = blueprint_1
    dependencies[blueprint_params_2] = blueprint_2

    experiment_params_1 = SimulatedExperimentParams(
        name="experiment_test_analysis",
        blueprint_params=blueprint_params_1,
        n_samples_per_clifford=n_samples_per_clifford,
        noise_model_name="None",
        noise_model_params=(0,),
        seed=1,
    )
    experiment_params_2 = SimulatedExperimentParams(
        name="experiment_test_analysis",
        blueprint_params=blueprint_params_2,
        n_samples_per_clifford=n_samples_per_clifford,
        noise_model_name="None",
        noise_model_params=(0,),
        seed=2,
    )

    experiment_1 = ExperimentData.build_experiment_from_dependencies(
        experiment_params_1, dependencies=dependencies
    )
    experiment_2 = ExperimentData.build_experiment_from_dependencies(
        experiment_params_2, dependencies=dependencies
    )
    dependencies[experiment_params_1] = experiment_1
    dependencies[experiment_params_2] = experiment_2

    analysis_params_1 = OverlapAnalysisParams(
        "test_analysis", experiment_params=experiment_params_1, k_to_calculate=(1,)
    )
    analysis_params_2 = OverlapAnalysisParams(
        "test_analysis", experiment_params=experiment_params_2, k_to_calculate=(1,)
    )

    analysis_1 = build_analysis(analysis_params_1, dependencies=dependencies)
    analysis_2 = build_analysis(analysis_params_2, dependencies=dependencies)

    energy_1 = get_variational_energy(
        analysis_data=analysis_1,
        trial_wf_data=trial_wf,
        hamiltonian_data=hamiltonian_data,
        k=1,
    )
    energy_2 = get_variational_energy(
        analysis_data=analysis_2,
        trial_wf_data=trial_wf,
        hamiltonian_data=hamiltonian_data,
        k=1,
    )

    print(energy_1)
    print(energy_2)
    print(trial_wf.ansatz_energy)
    print(trial_wf.ansatz_energy - energy_1)
    print(trial_wf.ansatz_energy - energy_2)

    print(analysis_1.reconstructed_wf_for_k["1"])
    print(analysis_2.reconstructed_wf_for_k["1"])

    assert np.abs(energy_1 - energy_2) < target_error


@pytest.mark.slow
@pytest.mark.parametrize(
    "qubit_partition",
    [
        # One part
        (
            (
                cirq.GridQubit(0, 0),
                cirq.GridQubit(0, 1),
                cirq.GridQubit(1, 1),
                cirq.GridQubit(1, 0),
            ),
        ),
        # One part (shuffled)
        (
            (
                cirq.GridQubit(1, 1),
                cirq.GridQubit(0, 1),
                cirq.GridQubit(0, 0),
                cirq.GridQubit(1, 0),
            ),
        ),
        # One part (shuffled again)
        (
            (
                cirq.GridQubit(0, 1),
                cirq.GridQubit(1, 0),
                cirq.GridQubit(1, 1),
                cirq.GridQubit(0, 0),
            ),
        ),
        # One part (shuffled to sorted order)
        (
            (
                cirq.GridQubit(0, 0),
                cirq.GridQubit(0, 1),
                cirq.GridQubit(1, 0),
                cirq.GridQubit(1, 1),
            ),
        ),
        # Two parts
        (
            (cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
            (cirq.GridQubit(1, 1), cirq.GridQubit(1, 0)),
        ),
        # Two parts (shuffled within pairs)
        (
            (cirq.GridQubit(0, 1), cirq.GridQubit(0, 0)),
            (cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)),
        ),
        # Two parts (shuffled)
        (
            (cirq.GridQubit(0, 0), cirq.GridQubit(1, 1)),
            (cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)),
        ),
        # Two parts (shuffled again)
        (
            (cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
            (cirq.GridQubit(1, 1), cirq.GridQubit(0, 1)),
        ),
        # Three parts
        (
            (cirq.GridQubit(0, 0),),
            (cirq.GridQubit(0, 1),),
            (cirq.GridQubit(1, 1), cirq.GridQubit(1, 0)),
        ),
        # Different Three parts
        (
            (cirq.GridQubit(1, 1),),
            (cirq.GridQubit(0, 0),),
            (cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)),
        ),
        # Four parts
        (
            (cirq.GridQubit(0, 0),),
            (cirq.GridQubit(1, 1),),
            (cirq.GridQubit(0, 1),),
            (cirq.GridQubit(1, 0),),
        ),
        # Four parts (shuffled back in order)
        (
            (cirq.GridQubit(0, 0),),
            (cirq.GridQubit(0, 1),),
            (cirq.GridQubit(1, 1),),
            (cirq.GridQubit(1, 0),),
        ),
        # Four parts (shuffled)
        (
            (cirq.GridQubit(0, 0),),
            (cirq.GridQubit(0, 1),),
            (cirq.GridQubit(1, 0),),
            (cirq.GridQubit(1, 1),),
        ),
    ],
)
def test_small_experiment_partitioned(
    qubit_partition: Tuple[Tuple[cirq.GridQubit]],
    fixture_4_qubit_ham_and_trial_wf: Tuple[HamiltonianData, TrialWavefunctionData],
):
    hamiltonian_data, trial_wf = fixture_4_qubit_ham_and_trial_wf
    trial_wf_params = trial_wf.params
    dependencies: Dict[Params, Data] = {trial_wf_params: trial_wf}

    blueprint_params = BlueprintParamsTrialWf(
        name="blueprint_test_analysis",
        trial_wf_params=trial_wf_params,
        n_cliffords=150,
        qubit_partition=qubit_partition,
        seed=18,
    )

    blueprint = BlueprintData.build_blueprint_from_dependencies(
        blueprint_params, dependencies=dependencies
    )
    dependencies[blueprint_params] = blueprint

    experiment_params = SimulatedExperimentParams(
        name="experiment_test_analysis",
        blueprint_params=blueprint_params,
        n_samples_per_clifford=10,
        noise_model_name="None",
        noise_model_params=(0,),
        seed=1,
    )

    experiment = ExperimentData.build_experiment_from_dependencies(
        experiment_params, dependencies=dependencies
    )
    dependencies[experiment_params] = experiment

    analysis_params = OverlapAnalysisParams(
        "test_analysis", experiment_params=experiment_params, k_to_calculate=(1,)
    )

    analysis = build_analysis(analysis_params, dependencies=dependencies)

    energy = get_variational_energy(
        analysis_data=analysis,
        trial_wf_data=trial_wf,
        hamiltonian_data=hamiltonian_data,
        k=1,
    )
    print(analysis.reconstructed_wf_for_k["1"])

    assert np.abs(trial_wf.ansatz_energy - energy) < 2.5e-2 * len(qubit_partition)
