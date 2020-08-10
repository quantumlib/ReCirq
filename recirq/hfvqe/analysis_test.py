from itertools import product
import numpy as np
import scipy as sp
from openfermioncirq.experiments.hfvqe.circuits import rhf_params_to_matrix
from openfermioncirq.experiments.hfvqe.analysis import (
    trace_distance,
    kdelta,
    energy_from_opdm,
    fidelity_witness,
    fidelity,
    mcweeny_purification
    )
from openfermioncirq.experiments.hfvqe.molecular_example import make_h6_1_3
from openfermioncirq.experiments.hfvqe.molecular_example_odd_qubits import (
    make_h3_2_5)
from openfermioncirq.experiments.hfvqe.gradient_hf import rhf_func_generator
# pylint: disable=C


def test_kdelta():
    assert np.isclose(kdelta(1, 1), 1.)
    assert np.isclose(kdelta(0, 1), 0.)


def test_trace_distance():
    rho = np.arange(16).reshape((4, 4))
    sigma = np.arange(16, 32).reshape((4, 4))
    assert np.isclose(trace_distance(rho, rho), 0.)
    assert np.isclose(trace_distance(rho, sigma), 32.0)


def test_energy_from_opdm():
    """Build test assuming sampling functions work"""

    rhf_objective, molecule, parameters, obi, tbi = make_h6_1_3()
    unitary, energy, _ = rhf_func_generator(rhf_objective)

    parameters = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    initial_opdm = np.diag([1] * 3 + [0] * 3)
    final_opdm = unitary(parameters) @ initial_opdm @ unitary(parameters).conj().T
    test_energy = energy_from_opdm(final_opdm,
                                   constant=molecule.nuclear_repulsion,
                                   one_body_tensor=obi,
                                   two_body_tensor=tbi)
    true_energy = energy(parameters)
    assert np.allclose(test_energy, true_energy)

def test_energy_from_opdm_odd_qubit():
    """Build test assuming sampling functions work"""

    rhf_objective, molecule, parameters, obi, tbi = make_h3_2_5()
    unitary, energy, _ = rhf_func_generator(rhf_objective)

    parameters = np.array([0.1, 0.2])
    initial_opdm = np.diag([1] * 1 + [0] * 2)
    print(initial_opdm)
    final_opdm = unitary(parameters) @ initial_opdm @ unitary(
        parameters).conj().T
    test_energy = energy_from_opdm(final_opdm,
                                   constant=molecule.nuclear_repulsion,
                                   one_body_tensor=obi,
                                   two_body_tensor=tbi)
    true_energy = energy(parameters)
    assert np.allclose(test_energy, true_energy)

def test_mcweeny():
    np.random.seed(82)
    opdm = np.array([[0.766034130, -0.27166330, -0.30936072, -0.08471057, -0.04878244, -0.01285432],
                     [-0.27166330,  0.67657015, -0.37519640, -0.02101843, -0.03568214, -0.05034585],
                     [-0.30936072, -0.37519640,  0.55896791,  0.04267370, -0.02258184, -0.08783738],
                     [-0.08471057, -0.02101843,  0.04267370,  0.05450848,  0.11291253,  0.17131658],
                     [-0.04878244, -0.03568214, -0.02258184,  0.11291253,  0.26821219,  0.42351185],
                     [-0.01285432, -0.05034585, -0.08783738,  0.17131658,  0.42351185,  0.67570713]])
    for i, j in product(range(6), repeat=2):
        opdm[i, j] += np.random.randn() * 1.0E-3
    opdm = 0.5 * (opdm + opdm.T)
    pure_opdm = mcweeny_purification(opdm)
    w, _ = np.linalg.eigh(pure_opdm)
    assert len(np.where(w < -1.0E-9)[0]) == 0


def test_fidelity():
    parameters = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    u = sp.linalg.expm(rhf_params_to_matrix(parameters, 6))
    opdm = np.array([[0.766034130, -0.27166330, -0.30936072, -0.08471057, -0.04878244, -0.01285432],
                     [-0.27166330,  0.67657015, -0.37519640, -0.02101843, -0.03568214, -0.05034585],
                     [-0.30936072, -0.37519640,  0.55896791,  0.04267370, -0.02258184, -0.08783738],
                     [-0.08471057, -0.02101843,  0.04267370,  0.05450848,  0.11291253,  0.17131658],
                     [-0.04878244, -0.03568214, -0.02258184,  0.11291253,  0.26821219,  0.42351185],
                     [-0.01285432, -0.05034585, -0.08783738,  0.17131658,  0.42351185,  0.67570713]])

    assert np.isclose(fidelity(u, opdm), 1.0)
    opdm += 0.1
    opdm = 0.5 * (opdm + opdm.T)
    assert np.isclose(fidelity(u, opdm), 0.3532702370138279)


def test_fidelity_witness():
    parameters = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    u = sp.linalg.expm(rhf_params_to_matrix(parameters, 6))
    omega = [1] * 3 + [0] * 3
    opdm = np.array([[0.766034130, -0.27166330, -0.30936072, -0.08471057, -0.04878244, -0.01285432],
                     [-0.27166330,  0.67657015, -0.37519640, -0.02101843, -0.03568214, -0.05034585],
                     [-0.30936072, -0.37519640,  0.55896791,  0.04267370, -0.02258184, -0.08783738],
                     [-0.08471057, -0.02101843,  0.04267370,  0.05450848,  0.11291253,  0.17131658],
                     [-0.04878244, -0.03568214, -0.02258184,  0.11291253,  0.26821219,  0.42351185],
                     [-0.01285432, -0.05034585, -0.08783738,  0.17131658,  0.42351185,  0.67570713]])

    assert np.isclose(fidelity_witness(u, omega, opdm), 1.0)

    opdm += 0.1
    opdm = 0.5 * (opdm + opdm.T)

    # higher than fidelity because of particle number breaking
    assert np.isclose(fidelity_witness(u, omega, opdm), 0.7721525013371697)
