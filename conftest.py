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

import numpy as np
import pytest

from recirq.qcqmc import blueprint, qubit_maps
from recirq.qcqmc.hamiltonian import (
    HamiltonianData,
    HamiltonianFileParams,
    build_hamiltonian_from_file,
)
from recirq.qcqmc.optimize_wf import build_pp_plus_trial_wavefunction
from recirq.qcqmc.trial_wf import (
    PerfectPairingPlusTrialWavefunctionParams,
    TrialWavefunctionData,
)


@pytest.fixture(scope="package")
def fixture_4_qubit_ham() -> HamiltonianData:
    params = HamiltonianFileParams(
        name="test hamiltonian 4 qubits", integral_key="fh_sto3g", n_orb=2, n_elec=2
    )

    hamiltonian_data = build_hamiltonian_from_file(params)

    return hamiltonian_data


@pytest.fixture(scope="package")
def fixture_8_qubit_ham() -> HamiltonianData:
    params = HamiltonianFileParams(
        name="test hamiltonian 8 qubits", integral_key="h4_sto3g", n_orb=4, n_elec=4
    )

    hamiltonian_data = build_hamiltonian_from_file(params)

    return hamiltonian_data


@pytest.fixture(scope="package")
def fixture_12_qubit_ham() -> HamiltonianData:
    params = HamiltonianFileParams(
        name="test hamiltonian 12 qubits",
        integral_key="diamond_dzvp/cas66",
        n_orb=6,
        n_elec=6,
        do_eri_restore=True,
    )

    hamiltonian_data = build_hamiltonian_from_file(params)

    return hamiltonian_data


@pytest.fixture(scope="package")
def fixture_4_qubit_ham_and_trial_wf(
    fixture_4_qubit_ham: HamiltonianData,
) -> Tuple[HamiltonianData, TrialWavefunctionData]:
    params = PerfectPairingPlusTrialWavefunctionParams(
        name="pp_test_wf_1",
        hamiltonian_params=fixture_4_qubit_ham.params,
        heuristic_layers=tuple(),
        do_pp=True,
        restricted=True,
    )

    trial_wf = build_pp_plus_trial_wavefunction(
        params, dependencies={fixture_4_qubit_ham.params: fixture_4_qubit_ham}
    )

    return fixture_4_qubit_ham, trial_wf


@pytest.fixture(scope="package")
def fixture_8_qubit_ham_and_trial_wf(
    fixture_8_qubit_ham: HamiltonianData,
) -> Tuple[HamiltonianData, TrialWavefunctionData]:
    params = PerfectPairingPlusTrialWavefunctionParams(
        name="pp_test_wf_qchem",
        hamiltonian_params=fixture_8_qubit_ham.params,
        heuristic_layers=tuple(),
        initial_orbital_rotation=None,
        initial_two_body_qchem_amplitudes=np.asarray([0.3, 0.4]),
        do_optimization=False,
    )

    trial_wf = build_pp_plus_trial_wavefunction(
        params, dependencies={fixture_8_qubit_ham.params: fixture_8_qubit_ham}
    )

    return fixture_8_qubit_ham, trial_wf


@pytest.fixture(scope="package")
def fixture_4_qubit_ham_trial_wf_and_blueprint(
    fixture_4_qubit_ham_and_trial_wf,
) -> Tuple[HamiltonianData, TrialWavefunctionData, blueprint.BlueprintData]:
    ham_data, trial_wf_data = fixture_4_qubit_ham_and_trial_wf
    trial_wf_params = trial_wf_data.params

    blueprint_params = blueprint.BlueprintParamsTrialWf(
        name="blueprint_test",
        trial_wf_params=trial_wf_params,
        n_cliffords=17,
        qubit_partition=(
            tuple(qubit_maps.get_qubits_a_b_reversed(n_orb=trial_wf_params.n_orb)),
        ),
        seed=1,
    )

    bp = blueprint.BlueprintData.build_blueprint_from_dependencies(
        blueprint_params, dependencies={trial_wf_params: trial_wf_data}
    )

    return ham_data, trial_wf_data, bp


def pytest_addoption(parser):
    parser.addoption("--skipslow", action="store_true", help="skips slow tests")


def pytest_runtest_setup(item):
    if "slow" in item.keywords and item.config.getvalue("skipslow"):
        pytest.skip("skipped because of --skipslow option")
