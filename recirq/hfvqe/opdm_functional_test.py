import cirq

import numpy as np

import pytest

from openfermioncirq.experiments.hfvqe.opdm_functionals import (RDMGenerator,
                                                                OpdmFunctional)
from openfermioncirq.experiments.hfvqe.analysis import compute_opdm
from openfermioncirq.experiments.hfvqe.molecular_example import make_h6_1_3


@pytest.mark.skip(reason='long-running systems test')
def test_opdm_func_vals():
    # coverage: ignore
    rhf_objective, molecule, parameters, obi, tbi = make_h6_1_3()
    qubits = [cirq.GridQubit(0, x) for x in range(molecule.n_orbitals)]
    np.random.seed(43)
    sampler = cirq.Simulator(dtype=np.complex128)
    opdm_func = OpdmFunctional(qubits=qubits,
                               sampler=sampler,
                               constant=molecule.nuclear_repulsion,
                               one_body_integrals=obi,
                               two_body_integrals=tbi,
                               num_electrons=molecule.n_electrons // 2)

    assert isinstance(opdm_func, OpdmFunctional)

    data = opdm_func.calculate_data(parameters)
    assert isinstance(data, dict)
    assert list(data.keys()) == ['z', 'xy_even', 'xy_odd', 'qubits',
                           'qubit_permutations', 'circuits',
                           'circuits_with_measurement']

    opdm_from_data = compute_opdm(data, return_variance=False)

    opdm_from_obj, var_dict = opdm_func.calculate_rdm(parameters)
    assert isinstance(var_dict, dict)
    assert np.linalg.norm(opdm_from_data - opdm_from_obj) < 1.0E-2

    assert np.isclose(opdm_func.energy_from_opdm(opdm_from_data),
                      rhf_objective.energy_from_opdm(opdm_from_data))

    rdm_gen = RDMGenerator(opdm_func)
    rdm_gen.opdm_generator(parameters)
    assert len(rdm_gen.noisy_opdms) == 1
    assert len(rdm_gen.variance_dicts) == 1
