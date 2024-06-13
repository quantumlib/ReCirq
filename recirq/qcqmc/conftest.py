from typing import Tuple

import numpy as np
import pytest

from recirq.qcqmc.hamiltonian import (
    HamiltonianData,
    LoadFromFileHamiltonianParams,
    build_hamiltonian_from_file,
)
from recirq.qcqmc.trial_wf import (
    PerfectPairingPlusTrialWavefunctionParams,
    TrialWavefunctionData,
    _get_qubits_a_b_reversed,
    build_pp_plus_trial_wavefunction,
)


@pytest.fixture(scope="package")
def fixture_4_qubit_ham() -> HamiltonianData:
    params = LoadFromFileHamiltonianParams(
        name="test hamiltonian 4 qubits", integral_key="fh_sto3g", n_orb=2, n_elec=2
    )

    hamiltonian_data = build_hamiltonian_from_file(params)

    return hamiltonian_data


@pytest.fixture(scope="package")
def fixture_8_qubit_ham() -> HamiltonianData:
    params = LoadFromFileHamiltonianParams(
        name="test hamiltonian 8 qubits", integral_key="h4_sto3g", n_orb=4, n_elec=4
    )

    hamiltonian_data = build_hamiltonian_from_file(params)

    return hamiltonian_data


@pytest.fixture(scope="package")
def fixture_12_qubit_ham() -> HamiltonianData:
    params = LoadFromFileHamiltonianParams(
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
