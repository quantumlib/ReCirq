import numpy as np
from openfermioncirq.experiments.hfvqe.gradient_hf import (rhf_func_generator,
                               rhf_minimization)
from openfermioncirq.experiments.hfvqe.molecular_example import make_h6_1_3


def test_rhf_func_gen():
    rhf_objective, molecule, parameters, _, _ = make_h6_1_3()
    ansatz, energy, _ = rhf_func_generator(rhf_objective)
    assert np.isclose(molecule.hf_energy, energy(parameters))

    ansatz, energy, _, opdm_func = rhf_func_generator(
        rhf_objective, initial_occ_vec=[1] * 3 + [0] * 3, get_opdm_func=True)
    assert np.isclose(molecule.hf_energy, energy(parameters))
    test_opdm = opdm_func(parameters)
    u = ansatz(parameters)
    initial_opdm = np.diag([1] * 3 + [0] * 3)
    final_odpm = u @ initial_opdm @ u.T
    assert np.allclose(test_opdm, final_odpm)

    result = rhf_minimization(rhf_objective, initial_guess=parameters)
    assert np.allclose(result.x, parameters)
