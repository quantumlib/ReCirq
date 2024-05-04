import cirq
import fqe
import numpy as np
import pytest

import recirq.qcqmc
import recirq.qcqmc.newtilities as newtilities
from recirq.qcqmc.analysis import OverlapAnalysisParams
from recirq.qcqmc.blueprint import BlueprintParamsTrialWf
from recirq.qcqmc.experiment import SimulatedExperimentParams
from recirq.qcqmc.hamiltonian import (
    LoadFromFileHamiltonianParams,
    PyscfHamiltonianParams,
)
from recirq.qcqmc.trial_wf import (
    _get_qubits_a_b_reversed,
    LayerSpec,
    PerfectPairingPlusTrialWavefunctionParams,
)

assert recirq.qcqmc, "Need to import the base package to register deserializers."


def test_serialize_hamiltonian_params():
    params = LoadFromFileHamiltonianParams(
        name="TEST_hamiltonian_4_qubits", integral_key="fh_sto3g", n_orb=2, n_elec=2
    )
    new_params = cirq.read_json(json_text=cirq.to_json(params))


def test_serialize_trial_wf_params():
    hamiltonian_params = LoadFromFileHamiltonianParams(
        name="test hamiltonian 4 qubits", integral_key="fh_sto3g", n_orb=2, n_elec=2
    )

    params = PerfectPairingPlusTrialWavefunctionParams(
        name="TEST_pp",
        hamiltonian_params=hamiltonian_params,
        heuristic_layers=(),
        do_pp=True,
        restricted=True,
    )

    new_params = cirq.read_json(json_text=cirq.to_json(params))
    assert params == new_params


def test_serialize_trial_wf_params_2():
    hamiltonian_params = LoadFromFileHamiltonianParams(
        name="TEST_hamiltonian_4_qubits", integral_key="fh_sto3g", n_orb=2, n_elec=2
    )

    params = PerfectPairingPlusTrialWavefunctionParams(
        name="TEST_pp",
        hamiltonian_params=hamiltonian_params,
        heuristic_layers=(
            LayerSpec("givens", "cross_pair"),
            LayerSpec("charge_charge", "in_pair"),
        ),
        restricted=True,
    )

    new_params = cirq.read_json(json_text=cirq.to_json(params))
    assert params == new_params


def test_serialize_blueprint_params():
    hamiltonian_params = LoadFromFileHamiltonianParams(
        name="TEST_hamiltonian_4_qubits", integral_key="fh_sto3g", n_orb=2, n_elec=2
    )

    trial_wf_params = PerfectPairingPlusTrialWavefunctionParams(
        name="TEST_pp",
        hamiltonian_params=hamiltonian_params,
        heuristic_layers=(),
        do_pp=True,
        restricted=True,
    )

    qubits = _get_qubits_a_b_reversed(n_orb=trial_wf_params.n_orb)
    params = BlueprintParamsTrialWf(
        name="TEST_blueprint",
        trial_wf_params=trial_wf_params,
        n_cliffords=1,
        qubit_partition=(qubits,),
        seed=4,
    )

    new_params = cirq.read_json(json_text=cirq.to_json(params))
    assert params == new_params


def test_serialize_experiment_params():
    hamiltonian_params = LoadFromFileHamiltonianParams(
        name="TEST_hamiltonian_4_qubits", integral_key="fh_sto3g", n_orb=2, n_elec=2
    )

    trial_wf_params = PerfectPairingPlusTrialWavefunctionParams(
        name="TEST_pp",
        hamiltonian_params=hamiltonian_params,
        heuristic_layers=(),
        do_pp=True,
        restricted=True,
    )

    qubits = _get_qubits_a_b_reversed(n_orb=trial_wf_params.n_orb)
    blueprint_params = BlueprintParamsTrialWf(
        name="TEST_blueprint_exp_3",
        trial_wf_params=trial_wf_params,
        n_cliffords=200,
        qubit_partition=(qubits,),
        seed=6,
    )

    params = SimulatedExperimentParams(
        name="TEST_experiment_2",
        blueprint_params=blueprint_params,
        n_samples_per_clifford=100,
        noise_model_name="ConstantTwoQubitGateDepolarizingNoiseModel",
        noise_model_params=(0.01,),
    )

    new_params = cirq.read_json(json_text=cirq.to_json(params))
    assert params == new_params


def test_serialize_analysis_params():
    hamiltonian_params = LoadFromFileHamiltonianParams(
        name="TEST_hamiltonian_4_qubits", integral_key="fh_sto3g", n_orb=2, n_elec=2
    )

    trial_wf_params = PerfectPairingPlusTrialWavefunctionParams(
        name="TEST_pp",
        hamiltonian_params=hamiltonian_params,
        heuristic_layers=(),
        do_pp=True,
        restricted=True,
    )

    qubits = _get_qubits_a_b_reversed(n_orb=trial_wf_params.n_orb)
    blueprint_params = BlueprintParamsTrialWf(
        name="TEST_blueprint_analysis",
        trial_wf_params=trial_wf_params,
        n_cliffords=100,
        qubit_partition=(qubits,),
        seed=6,
    )

    experiment_params = SimulatedExperimentParams(
        name="TEST_experiment_analysis",
        blueprint_params=blueprint_params,
        n_samples_per_clifford=100,
        noise_model_name="ConstantTwoQubitGateDepolarizingNoiseModel",
        noise_model_params=(0.01,),
    )

    params = OverlapAnalysisParams(
        name="TEST_analysis", experiment_params=experiment_params, k_to_calculate=(1, 3)
    )

    new_params = cirq.read_json(json_text=cirq.to_json(params))
    assert params == new_params


def test_nested_iterates():
    hamiltonian_params = LoadFromFileHamiltonianParams(
        name="TEST_hamiltonian_4_qubits", integral_key="fh_sto3g", n_orb=2, n_elec=2
    )

    trial_wf_params = PerfectPairingPlusTrialWavefunctionParams(
        name="TEST_pp",
        hamiltonian_params=hamiltonian_params,
        heuristic_layers=(),
        do_pp=True,
        restricted=True,
    )

    blueprint_params = BlueprintParamsTrialWf(
        name="TEST_blueprint_exp_2",
        trial_wf_params=trial_wf_params,
        n_cliffords=200,
        qubit_partition=(
            tuple(qubit for qubit in trial_wf_params.qubits_jordan_wigner_ordered),
        ),
        seed=3,
    )

    params = SimulatedExperimentParams(
        name="TEST_experiment_2",
        blueprint_params=blueprint_params,
        n_samples_per_clifford=100,
        noise_model_name="ConstantTwoQubitGateDepolarizingNoiseModel",
        noise_model_params=(0.01,),
    )

    test = list(
        thing for thing in newtilities.nested_params_iter(params, yield_self=False)
    )

    assert len(test) == 3
    assert test[0] == hamiltonian_params
    assert test[1] == trial_wf_params
    assert test[2] == blueprint_params

    test_with_self = list(thing for thing in newtilities.nested_params_iter(params))

    assert len(test_with_self) == 4
    assert test_with_self[3] == params


def test_hamiltonian_save_load_delete():
    params = LoadFromFileHamiltonianParams(
        name="TEST_hamiltonian_4_qubits", integral_key="fh_sto3g", n_orb=2, n_elec=2
    )

    newtilities.try_delete_data_file(params)
    assert not params.base_path.with_suffix(".gzip").exists()
    assert not params.base_path.with_suffix(".lock").exists()

    hamiltonian = newtilities.run(params, save=False)

    assert not params.base_path.with_suffix(".gzip").exists()
    assert params.base_path.with_suffix(".lock").exists()

    hamiltonian = newtilities.run(params)

    assert params.base_path.with_suffix(".gzip").exists()
    assert params.base_path.with_suffix(".lock").exists()

    loaded_hamiltonian = newtilities.load_data(params)

    assert loaded_hamiltonian == hamiltonian

    newtilities.try_delete_data_file(params)
    assert not params.base_path.with_suffix(".gzip").exists()
    assert not params.base_path.with_suffix(".lock").exists()


def test_pp_wf_save_load_delete():
    hamiltonian_params = LoadFromFileHamiltonianParams(
        name="TEST_hamiltonian_4_qubits", integral_key="fh_sto3g", n_orb=2, n_elec=2
    )

    params = PerfectPairingPlusTrialWavefunctionParams(
        name="TEST_pp",
        hamiltonian_params=hamiltonian_params,
        heuristic_layers=(),
        do_pp=True,
        restricted=True,
    )

    newtilities.try_delete_data_file(params)
    newtilities.try_delete_data_file(hamiltonian_params)
    assert not params.base_path.with_suffix(".gzip").exists()
    assert not params.base_path.with_suffix(".h5").exists()
    assert not params.base_path.with_suffix(".lock").exists()
    assert not hamiltonian_params.base_path.with_suffix(".gzip").exists()
    assert not hamiltonian_params.base_path.with_suffix(".lock").exists()

    trial_wf = newtilities.run(params, save=False, run_dependencies_if_necessary=True)

    assert not params.base_path.with_suffix(".gzip").exists()
    assert not params.base_path.with_suffix(".h5").exists()
    assert params.base_path.with_suffix(".lock").exists()
    assert not hamiltonian_params.base_path.with_suffix(".gzip").exists()
    assert hamiltonian_params.base_path.with_suffix(".lock").exists()

    trial_wf = newtilities.run(params, run_dependencies_if_necessary=True)

    assert params.base_path.with_suffix(".gzip").exists()
    assert params.base_path.with_suffix(".h5").exists()
    assert params.base_path.with_suffix(".lock").exists()
    assert hamiltonian_params.base_path.with_suffix(".gzip").exists()
    assert hamiltonian_params.base_path.with_suffix(".lock").exists()

    loaded_trial_wf = newtilities.load_data(params)

    assert loaded_trial_wf == trial_wf

    newtilities.try_delete_data_file(params)
    newtilities.try_delete_data_file(hamiltonian_params)
    assert not params.base_path.with_suffix(".gzip").exists()
    assert not params.base_path.with_suffix(".h5").exists()
    assert not params.base_path.with_suffix(".lock").exists()
    assert not hamiltonian_params.base_path.with_suffix(".gzip").exists()
    assert not hamiltonian_params.base_path.with_suffix(".lock").exists()


def test_blueprint_save_load_delete():
    hamiltonian_params = LoadFromFileHamiltonianParams(
        name="TEST_hamiltonian_4_qubits", integral_key="fh_sto3g", n_orb=2, n_elec=2
    )

    trial_wf_params = PerfectPairingPlusTrialWavefunctionParams(
        name="TEST_pp",
        hamiltonian_params=hamiltonian_params,
        heuristic_layers=(),
        do_pp=True,
        restricted=True,
    )

    params = BlueprintParamsTrialWf(
        name="TEST_blueprint",
        trial_wf_params=trial_wf_params,
        n_cliffords=10,
        qubit_partition=(
            tuple(qubit for qubit in trial_wf_params.qubits_jordan_wigner_ordered),
        ),
        seed=30,
    )

    newtilities.try_delete_data_file(params)
    newtilities.try_delete_data_file(trial_wf_params)
    newtilities.try_delete_data_file(hamiltonian_params)
    assert not params.base_path.with_suffix(".gzip").exists()
    assert not params.base_path.with_suffix(".lock").exists()
    assert not trial_wf_params.base_path.with_suffix(".gzip").exists()
    assert not trial_wf_params.base_path.with_suffix(".h5").exists()
    assert not trial_wf_params.base_path.with_suffix(".lock").exists()
    assert not hamiltonian_params.base_path.with_suffix(".gzip").exists()
    assert not hamiltonian_params.base_path.with_suffix(".lock").exists()

    blueprint = newtilities.run(params, run_dependencies_if_necessary=True)

    assert params.base_path.with_suffix(".gzip").exists()
    assert params.base_path.with_suffix(".lock").exists()
    assert trial_wf_params.base_path.with_suffix(".gzip").exists()
    assert trial_wf_params.base_path.with_suffix(".h5").exists()
    assert trial_wf_params.base_path.with_suffix(".lock").exists()
    assert hamiltonian_params.base_path.with_suffix(".gzip").exists()
    assert hamiltonian_params.base_path.with_suffix(".lock").exists()

    loaded_blueprint = newtilities.load_data(params)

    print(blueprint)
    print(loaded_blueprint)
    assert loaded_blueprint == blueprint

    newtilities.try_delete_data_file(params)
    newtilities.try_delete_data_file(trial_wf_params)
    newtilities.try_delete_data_file(hamiltonian_params)
    assert not params.base_path.with_suffix(".gzip").exists()
    assert not params.base_path.with_suffix(".lock").exists()
    assert not trial_wf_params.base_path.with_suffix(".gzip").exists()
    assert not trial_wf_params.base_path.with_suffix(".h5").exists()
    assert not trial_wf_params.base_path.with_suffix(".lock").exists()
    assert not hamiltonian_params.base_path.with_suffix(".gzip").exists()
    assert not hamiltonian_params.base_path.with_suffix(".lock").exists()


def test_experiment_save_load_delete():
    hamiltonian_params = LoadFromFileHamiltonianParams(
        name="TEST_hamiltonian_4_qubits", integral_key="fh_sto3g", n_orb=2, n_elec=2
    )

    trial_wf_params = PerfectPairingPlusTrialWavefunctionParams(
        name="TEST_pp",
        hamiltonian_params=hamiltonian_params,
        heuristic_layers=(),
        do_pp=True,
        restricted=True,
    )

    blueprint_params = BlueprintParamsTrialWf(
        name="TEST_blueprint_2",
        trial_wf_params=trial_wf_params,
        n_cliffords=10,
        qubit_partition=(
            tuple(qubit for qubit in trial_wf_params.qubits_jordan_wigner_ordered),
        ),
        seed=17,
    )

    params = SimulatedExperimentParams(
        name="TEST_experiment_3",
        blueprint_params=blueprint_params,
        n_samples_per_clifford=10,
        noise_model_name="None",
        noise_model_params=(0,),
    )

    newtilities.try_delete_data_file(params)
    newtilities.try_delete_data_file(blueprint_params)
    newtilities.try_delete_data_file(trial_wf_params)
    newtilities.try_delete_data_file(hamiltonian_params)
    assert not params.base_path.with_suffix(".gzip").exists()
    assert not params.base_path.with_suffix(".lock").exists()
    assert not blueprint_params.base_path.with_suffix(".gzip").exists()
    assert not blueprint_params.base_path.with_suffix(".lock").exists()
    assert not trial_wf_params.base_path.with_suffix(".gzip").exists()
    assert not trial_wf_params.base_path.with_suffix(".h5").exists()
    assert not trial_wf_params.base_path.with_suffix(".lock").exists()
    assert not hamiltonian_params.base_path.with_suffix(".gzip").exists()
    assert not hamiltonian_params.base_path.with_suffix(".lock").exists()

    experiment = newtilities.run(params, run_dependencies_if_necessary=True)

    assert params.base_path.with_suffix(".gzip").exists()
    assert params.base_path.with_suffix(".lock").exists()
    assert blueprint_params.base_path.with_suffix(".gzip").exists()
    assert blueprint_params.base_path.with_suffix(".lock").exists()
    assert trial_wf_params.base_path.with_suffix(".gzip").exists()
    assert trial_wf_params.base_path.with_suffix(".h5").exists()
    assert trial_wf_params.base_path.with_suffix(".lock").exists()
    assert hamiltonian_params.base_path.with_suffix(".gzip").exists()
    assert hamiltonian_params.base_path.with_suffix(".lock").exists()

    loaded_experiment = newtilities.load_data(params)

    print(experiment)
    print(loaded_experiment)
    assert loaded_experiment == experiment

    newtilities.try_delete_data_file(params)
    newtilities.try_delete_data_file(blueprint_params)
    newtilities.try_delete_data_file(trial_wf_params)
    newtilities.try_delete_data_file(hamiltonian_params)
    assert not params.base_path.with_suffix(".gzip").exists()
    assert not params.base_path.with_suffix(".lock").exists()
    assert not blueprint_params.base_path.with_suffix(".gzip").exists()
    assert not blueprint_params.base_path.with_suffix(".lock").exists()
    assert not trial_wf_params.base_path.with_suffix(".gzip").exists()
    assert not trial_wf_params.base_path.with_suffix(".h5").exists()
    assert not trial_wf_params.base_path.with_suffix(".lock").exists()
    assert not hamiltonian_params.base_path.with_suffix(".gzip").exists()
    assert not hamiltonian_params.base_path.with_suffix(".lock").exists()


def test_fqe_occ_mapping():
    n_elec = 4
    n_orb = 12
    fqe_wf = fqe.Wavefunction([[n_elec, 0, n_orb]])
    np.random.seed(7)
    # generate some random coefficients
    sector = fqe_wf.sector((n_elec, 0))
    coeffs = np.random.random(size=(sector._core.lena(), sector._core.lenb()))
    sector.coeff = coeffs.copy()
    occ_coeff = newtilities.get_occa_occb_coeff(fqe_wf)

    fqe_graph = sector.get_fcigraph()
    for coeff, occa, occb in zip(occ_coeff.coeffs, occ_coeff.occa, occ_coeff.occb):
        alpha_str = fqe.bitstring.reverse_integer_index(occa)
        beta_str = fqe.bitstring.reverse_integer_index(occb)
        inda = fqe_graph.index_alpha(alpha_str)
        indb = fqe_graph.index_alpha(beta_str)
        assert np.isclose(coeff, fqe_wf.sector((n_elec, 0)).coeff[inda, indb])

    fqe_wf_from_coeff = newtilities.get_fqe_wf_from_occ_coeff(
        occ_coeff, n_elec, 0, n_orb
    )
    sector_from_coeff = fqe_wf_from_coeff.sector((n_elec, 0))
    assert np.allclose(sector_from_coeff.coeff, sector.coeff)


def test_pyscf_ham_save_load_delete():
    params = PyscfHamiltonianParams(
        name="TEST_square_H4",
        n_orb=4,
        n_elec=4,
        geometry=(
            ("H", (0, 0, 0)),
            ("H", (0, 0, 1.23)),
            ("H", (1.23, 0, 0)),
            ("H", (1.23, 0, 1.23)),
        ),
        basis="sto3g",
        multiplicity=1,
        charge=0,
        save_chkfile=True,
        overwrite_chk_file=True,
    )

    newtilities.try_delete_data_file(params)
    assert not params.base_path.with_suffix(".gzip").exists()
    assert not params.base_path.with_suffix(".chk").exists()
    assert not params.base_path.with_suffix(".lock").exists()

    pyscf_ham = newtilities.run(params, save=False, run_dependencies_if_necessary=True)

    assert not params.base_path.with_suffix(".gzip").exists()
    assert params.base_path.with_suffix(".chk").exists()
    # Note that this behavior differs from the usual. Even if we don't ask for
    # the data to be saved we still get a .chk file from running pyscf.
    assert params.base_path.with_suffix(".lock").exists()

    pyscf_ham = newtilities.run(params, run_dependencies_if_necessary=True)

    assert params.base_path.with_suffix(".gzip").exists()
    assert params.base_path.with_suffix(".chk").exists()
    assert params.base_path.with_suffix(".lock").exists()

    loaded_pyscf_ham = newtilities.load_data(params)

    assert loaded_pyscf_ham == pyscf_ham

    newtilities.try_delete_data_file(params)
    newtilities.try_delete_data_file(params)
    assert not params.base_path.with_suffix(".gzip").exists()
    assert not params.base_path.with_suffix(".chk").exists()
    assert not params.base_path.with_suffix(".lock").exists()


@pytest.mark.slow()
def test_pp_wf_with_pyscf_ham_save_load_delete():
    pyscf_params = PyscfHamiltonianParams(
        name="TEST_square_H4",
        n_orb=4,
        n_elec=4,
        geometry=(
            ("H", (0, 0, 0)),
            ("H", (0, 0, 1.23)),
            ("H", (1.23, 0, 0)),
            ("H", (1.23, 0, 1.23)),
        ),
        basis="sto3g",
        multiplicity=1,
        charge=0,
        save_chkfile=True,
        overwrite_chk_file=True,
    )

    params = PerfectPairingPlusTrialWavefunctionParams(
        name="TEST_pp",
        hamiltonian_params=pyscf_params,
        heuristic_layers=(),
        do_pp=True,
        restricted=True,
    )

    newtilities.try_delete_data_file(params)
    newtilities.try_delete_data_file(pyscf_params)
    assert not params.base_path.with_suffix(".gzip").exists()
    assert not params.base_path.with_suffix(".h5").exists()
    assert not params.base_path.with_suffix(".lock").exists()
    assert not pyscf_params.base_path.with_suffix(".gzip").exists()
    assert not pyscf_params.base_path.with_suffix(".lock").exists()

    trial_wf = newtilities.run(params, save=False, run_dependencies_if_necessary=True)

    assert not params.base_path.with_suffix(".gzip").exists()
    assert not params.base_path.with_suffix(".h5").exists()
    assert params.base_path.with_suffix(".lock").exists()
    assert not pyscf_params.base_path.with_suffix(".gzip").exists()
    assert pyscf_params.base_path.with_suffix(".lock").exists()

    trial_wf = newtilities.run(params, run_dependencies_if_necessary=True)

    assert params.base_path.with_suffix(".gzip").exists()
    assert params.base_path.with_suffix(".h5").exists()
    assert params.base_path.with_suffix(".lock").exists()
    assert pyscf_params.base_path.with_suffix(".gzip").exists()
    assert pyscf_params.base_path.with_suffix(".lock").exists()

    loaded_trial_wf = newtilities.load_data(params)

    assert loaded_trial_wf == trial_wf

    newtilities.try_delete_data_file(params)
    newtilities.try_delete_data_file(pyscf_params)
    assert not params.base_path.with_suffix(".gzip").exists()
    assert not params.base_path.with_suffix(".h5").exists()
    assert not params.base_path.with_suffix(".lock").exists()
    assert not pyscf_params.base_path.with_suffix(".gzip").exists()
    assert not pyscf_params.base_path.with_suffix(".lock").exists()


def test_load_pyscf_ham_from_gzip_file():
    params = PyscfHamiltonianParams(
        name="TEST square_H4",
        n_orb=4,
        n_elec=4,
        geometry=(
            ("H", (0, 0, 0)),
            ("H", (0, 0, 1.23)),
            ("H", (1.23, 0, 0)),
            ("H", (1.23, 0, 1.23)),
        ),
        basis="sto3g",
        multiplicity=1,
        charge=0,
        save_chkfile=True,
        overwrite_chk_file=True,
    )

    newtilities.try_delete_data_file(params)
    assert not params.base_path.with_suffix(".gzip").exists()
    assert not params.base_path.with_suffix(".chk").exists()
    assert not params.base_path.with_suffix(".lock").exists()

    pyscf_ham = newtilities.run(params, run_dependencies_if_necessary=True)

    assert params.base_path.with_suffix(".gzip").exists()
    assert params.base_path.with_suffix(".chk").exists()
    assert params.base_path.with_suffix(".lock").exists()

    loaded_pyscf_ham = newtilities.load_data_from_gzip_file(
        params.base_path.with_suffix(".gzip")
    )

    assert loaded_pyscf_ham == pyscf_ham

    newtilities.try_delete_data_file(params)
    newtilities.try_delete_data_file(params)
