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

import math

import cirq
import numpy as np
import pytest

from recirq.qcqmc import hamiltonian, qubit_maps, trial_wf


def test_trial_wavefunction_params():
    integral_key, n_orb, n_elec, do_eri_restore = "fh_sto3g", 2, 2, False
    params = hamiltonian.HamiltonianFileParams(
        name="test hamiltonian",
        integral_key=integral_key,
        n_orb=n_orb,
        n_elec=n_elec,
        do_eri_restore=do_eri_restore,
    )
    with pytest.raises(NotImplementedError, match="should be subclassed"):
        _ = trial_wf.TrialWavefunctionParams(
            name="test",
            hamiltonian_params=params,
        ).bitstrings()
    with pytest.raises(NotImplementedError, match="should be subclassed"):
        _ = trial_wf.TrialWavefunctionParams(
            name="test",
            hamiltonian_params=params,
        ).qubits_jordan_wigner_ordered()
    with pytest.raises(NotImplementedError, match="should be subclassed"):
        _ = trial_wf.TrialWavefunctionParams(
            name="test",
            hamiltonian_params=params,
        ).qubits_linearly_connected()


def test_perfect_pairing_plus_trial_wavefunction_params():
    ham_params = hamiltonian.HamiltonianFileParams(
        name="test hamiltonian 4 qubits", integral_key="fh_sto3g", n_orb=2, n_elec=2
    )
    trial_params = trial_wf.PerfectPairingPlusTrialWavefunctionParams(
        name="pp_test_wf_1",
        hamiltonian_params=ham_params,
        heuristic_layers=tuple(),
        do_pp=True,
        restricted=True,
        path_prefix="/tmp",
    )
    assert trial_params.n_orb == ham_params.n_orb
    assert trial_params.n_elec == ham_params.n_elec
    assert trial_params.n_qubits == 2 * ham_params.n_orb
    assert trial_params.n_pairs == ham_params.n_elec // 2
    assert trial_params.path_string == "/tmp/data/trial_wfs/pp_test_wf_1"
    n_orb = trial_params.n_orb
    n_el_each_spin = trial_params.n_elec // 2
    assert len(list(trial_params.bitstrings)) == math.comb(n_orb, n_el_each_spin) ** 2
    assert trial_params.qubits_jordan_wigner_ordered == qubit_maps.get_qubits_a_b(
        n_orb=trial_params.n_orb
    )
    assert trial_params.qubits_linearly_connected == qubit_maps.get_qubits_a_b_reversed(
        n_orb=trial_params.n_orb
    )
    assert trial_params.mode_qubit_map == qubit_maps.get_mode_qubit_map_pp_plus(
        n_qubits=trial_params.n_qubits
    )
    params2 = cirq.read_json(json_text=cirq.to_json(trial_params))
    assert params2 == trial_params


def test_trial_wavefunction_data(
    fixture_4_qubit_ham_and_trial_wf,
):
    _, trial_params = fixture_4_qubit_ham_and_trial_wf
    data = trial_wf.TrialWavefunctionData(
        params=trial_params,
        ansatz_circuit=cirq.Circuit(),
        superposition_circuit=cirq.Circuit(),
        hf_energy=0.0,
        ansatz_energy=0.0,
        fci_energy=0.0,
        one_body_basis_change_mat=np.zeros((2, 2)),
        one_body_params=np.zeros((2, 2)),
        two_body_params=np.zeros((2, 2, 2, 2)),
    )
    data2 = cirq.read_json(json_text=cirq.to_json(data))

    assert data2 == data
