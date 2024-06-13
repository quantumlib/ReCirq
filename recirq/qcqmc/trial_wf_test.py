import cirq
import fqe
import numpy as np
import pytest
import scipy.special
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian

from recirq.qcqmc.hamiltonian import HamiltonianData
from recirq.qcqmc.trial_wf import (FermionicMode, LayerSpec,
                                   PerfectPairingPlusTrialWavefunctionParams,
                                   _get_ansatz_qubit_wf, _get_bitstrings_a_b,
                                   _get_pp_plus_gate_generators,
                                   build_pp_plus_trial_wavefunction,
                                   evaluate_gradient_and_cost_function,
                                   get_evolved_wf,
                                   get_two_body_params_from_qchem_amplitudes)


def test_fermionic_mode():
    fm = FermionicMode(5, "a")
    fm2 = cirq.read_json(json_text=cirq.to_json(fm))
    assert fm == fm2

    with pytest.raises(ValueError, match="spin.*"):
        _ = FermionicMode(10, "c")


def test_get_bitstrings_a_b():
    with pytest.raises(NotImplementedError):
        list(_get_bitstrings_a_b(n_orb=4, n_elec=3))

    bitstrings = np.array(list(_get_bitstrings_a_b(n_orb=4, n_elec=4)))

    assert bitstrings.shape[0] == scipy.special.binom(4, 2) ** 2
    assert bitstrings.shape[1] == 2 * 4  # n_qubits columns = 2 * n_orb.
    hamming_weight_left = np.sum(bitstrings[:, 0:4], axis=1)
    hamming_weight_right = np.sum(bitstrings[:, 4:8], axis=1)

    assert np.all(hamming_weight_left == 2)
    assert np.all(hamming_weight_right == 2)


def test_pp_wf_energy(fixture_4_qubit_ham: HamiltonianData):
    params = PerfectPairingPlusTrialWavefunctionParams(
        name="pp_test_wf_1",
        hamiltonian_params=fixture_4_qubit_ham.params,
        heuristic_layers=(),
        do_pp=True,
        restricted=True,
    )

    trial_wf = build_pp_plus_trial_wavefunction(
        params, dependencies={fixture_4_qubit_ham.params: fixture_4_qubit_ham}
    )

    assert np.isclose(trial_wf.ansatz_energy, fixture_4_qubit_ham.e_fci)


def test_pp_wf_energy_with_layer(fixture_4_qubit_ham: HamiltonianData):
    params = PerfectPairingPlusTrialWavefunctionParams(
        name="pp_test_wf_2",
        hamiltonian_params=fixture_4_qubit_ham.params,
        heuristic_layers=(LayerSpec("charge_charge", "cross_spin"),),
        do_pp=True,
        restricted=True,
    )

    trial_wf = build_pp_plus_trial_wavefunction(
        params, dependencies={fixture_4_qubit_ham.params: fixture_4_qubit_ham}
    )

    assert np.isclose(trial_wf.ansatz_energy, fixture_4_qubit_ham.e_fci)


def test_qchem_pp_eight_qubit_wavefunctions_consistent(
    fixture_8_qubit_ham: HamiltonianData,
):
    """Tests (without optimization) that the eight qubit wavefunctions work.

    Specifically, that constructing the wavefunction with FQE and then
    converting it to a cirq wavefunction yields the same result as constructing
    the parameters with the circuit directly.
    """
    params = PerfectPairingPlusTrialWavefunctionParams(
        name="pp_test_wf_qchem",
        hamiltonian_params=fixture_8_qubit_ham.params,
        heuristic_layers=tuple(),
        initial_orbital_rotation=None,
        initial_two_body_qchem_amplitudes=np.asarray([0.3, 0.4]),
        do_optimization=False,
    )

    trial_wf = build_pp_plus_trial_wavefunction(
        params,
        dependencies={fixture_8_qubit_ham.params: fixture_8_qubit_ham},
        do_print=False,
    )

    one_body_params = trial_wf.one_body_params
    two_body_params = trial_wf.two_body_params
    basis_change_mat = trial_wf.one_body_basis_change_mat

    np.testing.assert_array_almost_equal(one_body_params, np.zeros((12,)))
    np.testing.assert_array_almost_equal(basis_change_mat, np.diag(np.ones(8)))

    np.testing.assert_equal(two_body_params.shape, (2,))


def test_pp_plus_wf_energy_sloppy_1(fixture_8_qubit_ham: HamiltonianData):
    params = PerfectPairingPlusTrialWavefunctionParams(
        "pp_plus_test",
        hamiltonian_params=fixture_8_qubit_ham.params,
        heuristic_layers=tuple(),
        do_pp=True,
        restricted=False,
        random_parameter_scale=1,
    )

    trial_wf = build_pp_plus_trial_wavefunction(
        params,
        dependencies={fixture_8_qubit_ham.params: fixture_8_qubit_ham},
        do_print=True,
    )

    assert trial_wf.ansatz_energy < -1.947


# TODO: Speed up this test and add a similar one with non-trivial heuristic layers.


def test_diamond_pp_wf_energy(fixture_12_qubit_ham: HamiltonianData):
    params = PerfectPairingPlusTrialWavefunctionParams(
        name="diamind_pp_test_wf_1",
        hamiltonian_params=fixture_12_qubit_ham.params,
        heuristic_layers=tuple(),
        do_pp=True,
        restricted=True,
        random_parameter_scale=0.1,
        n_optimization_restarts=1,
    )

    trial_wf = build_pp_plus_trial_wavefunction(
        params,
        dependencies={fixture_12_qubit_ham.params: fixture_12_qubit_ham},
        do_print=True,
    )

    assert trial_wf.ansatz_energy < -10.4


@pytest.mark.parametrize(
    "initial_two_body_qchem_amplitudes, expected_ansatz_qubit_wf",
    [
        (
            [1],
            np.array(
                [
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.70710678 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.70710678 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                ]
            ),
        ),
        (
            [0],
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        ),
    ],
)
def test_qchem_pp_runs(
    initial_two_body_qchem_amplitudes,
    expected_ansatz_qubit_wf,
    fixture_4_qubit_ham: HamiltonianData,
):
    params = PerfectPairingPlusTrialWavefunctionParams(
        name="pp_test_wf_qchem",
        hamiltonian_params=fixture_4_qubit_ham.params,
        heuristic_layers=tuple(),
        initial_orbital_rotation=None,
        initial_two_body_qchem_amplitudes=initial_two_body_qchem_amplitudes,
        do_optimization=False,
    )

    trial_wf = build_pp_plus_trial_wavefunction(
        params,
        dependencies={fixture_4_qubit_ham.params: fixture_4_qubit_ham},
        do_print=False,
    )

    ansatz_qubit_wf = _get_ansatz_qubit_wf(
        ansatz_circuit=trial_wf.ansatz_circuit,
        ordered_qubits=params.qubits_jordan_wigner_ordered,
    )

    np.testing.assert_array_almost_equal(ansatz_qubit_wf, expected_ansatz_qubit_wf)


def test_qchem_conversion_negative(fixture_4_qubit_ham: HamiltonianData):
    qchem_amplitudes = np.asarray(-0.1)

    two_body_params = get_two_body_params_from_qchem_amplitudes(qchem_amplitudes)

    assert two_body_params.item() < 0

    params = PerfectPairingPlusTrialWavefunctionParams(
        name="pp_test_wf_qchem_neg",
        hamiltonian_params=fixture_4_qubit_ham.params,
        heuristic_layers=tuple(),
        initial_orbital_rotation=None,
        initial_two_body_qchem_amplitudes=qchem_amplitudes,
        do_optimization=False,
    )

    trial_wf = build_pp_plus_trial_wavefunction(
        params,
        dependencies={fixture_4_qubit_ham.params: fixture_4_qubit_ham},
        do_print=False,
    )

    ansatz_qubit_wf = _get_ansatz_qubit_wf(
        ansatz_circuit=trial_wf.ansatz_circuit,
        ordered_qubits=params.qubits_jordan_wigner_ordered,
    )

    assert any(ansatz_qubit_wf < 0)


def gen_random_restricted_ham(n_orb: int) -> RestrictedHamiltonian:
    """8-fold symmetry restricted hamiltonian"""
    h1e = np.random.random((n_orb,) * 2)
    h1e = h1e + h1e.T
    h2e = np.random.random((n_orb,) * 4)
    h2e = h2e + h2e.transpose(2, 3, 0, 1)
    h2e = h2e + h2e.transpose(3, 2, 1, 0)
    h2e = h2e + h2e.transpose(1, 0, 2, 3)
    h2e = np.asarray(h2e.transpose(0, 2, 3, 1), order="C")
    fqe_ham = RestrictedHamiltonian((h1e, np.einsum("ijlk", -0.5 * h2e)))
    return fqe_ham


def get_fd_grad(
    n_orb,
    n_elec,
    one_body_params,
    two_body_params,
    ham,
    initial_wf,
    dtheta=1e-4,
    restricted=False,
):
    generators = _get_pp_plus_gate_generators(
        n_elec=n_elec, heuristic_layers=tuple(), do_pp=True
    )
    one_body_gradient = np.zeros_like(one_body_params)
    for ig, _ in enumerate(one_body_gradient):
        new_param = one_body_params.copy()
        new_param[ig] = new_param[ig] + dtheta
        phi = get_evolved_wf(
            new_param,
            two_body_params,
            initial_wf,
            generators,
            n_orb,
            restricted=restricted,
        )[0]
        e_plus = phi.expectationValue(ham)
        new_param[ig] = new_param[ig] - 2 * dtheta
        phi = get_evolved_wf(
            new_param,
            two_body_params,
            initial_wf,
            generators,
            n_orb,
            restricted=restricted,
        )[0]
        e_minu = phi.expectationValue(ham)
        one_body_gradient[ig] = (e_plus - e_minu).real / (2 * dtheta)
    two_body_gradient = np.zeros_like(two_body_params)
    for ig, _ in enumerate(two_body_gradient):
        new_param = two_body_params.copy()
        new_param[ig] = new_param[ig] + dtheta
        phi = get_evolved_wf(
            one_body_params,
            new_param,
            initial_wf,
            generators,
            n_orb,
            restricted=restricted,
        )[0]
        e_plus = phi.expectationValue(ham)
        new_param[ig] = new_param[ig] - 2 * dtheta
        phi = get_evolved_wf(
            one_body_params,
            new_param,
            initial_wf,
            generators,
            n_orb,
            restricted=restricted,
        )[0]
        e_minu = phi.expectationValue(ham)
        two_body_gradient[ig] = (e_plus - e_minu).real / (2 * dtheta)
    return one_body_gradient, two_body_gradient


@pytest.mark.parametrize("n_elec, n_orb", ((2, 2), (4, 4), (6, 6)))
@pytest.mark.parametrize("restricted", (True, False))
def test_gradient(n_elec, n_orb, restricted):
    sz = 0
    initial_wf = fqe.Wavefunction([[n_elec, sz, n_orb]])
    initial_wf.set_wfn(strategy="hartree-fock")

    fqe_ham = gen_random_restricted_ham(n_orb)

    if restricted:
        n_one_body_params = n_orb * (n_orb - 1) // 2
    else:
        n_one_body_params = n_orb * (n_orb - 1)

    gate_generators = _get_pp_plus_gate_generators(
        n_elec=n_elec, heuristic_layers=tuple(), do_pp=True
    )
    # reference implementation
    one_body_params = np.random.random(n_one_body_params)
    two_body_params = np.random.random(len(gate_generators))
    phi = get_evolved_wf(
        one_body_params,
        two_body_params,
        initial_wf,
        gate_generators,
        n_orb,
        restricted=restricted,
    )[0]
    obj_val, grad = evaluate_gradient_and_cost_function(
        initial_wf,
        fqe_ham,
        n_orb,
        one_body_params,
        two_body_params,
        gate_generators,
        restricted,
        0.0,
    )
    ob_fd_grad, tb_fd_grad = get_fd_grad(
        n_orb,
        n_elec,
        one_body_params,
        two_body_params,
        fqe_ham,
        initial_wf,
        restricted=restricted,
    )
    assert np.isclose(obj_val, phi.expectationValue(fqe_ham))
    assert np.allclose(ob_fd_grad, grad[-n_one_body_params:])
    n_two_body_params = len(two_body_params)
    assert np.allclose(tb_fd_grad, grad[:n_two_body_params])