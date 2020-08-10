"""
An implementation of gradient based Restricted-Hartree-Fock

This uses a bunch of the infrastructure already used in the experiment
should only need an RHF_object.
"""
# pylint: disable=C
from typing import Optional, Union
import numpy as np
import scipy as sp

from openfermioncirq.experiments.hfvqe.circuits import rhf_params_to_matrix
from openfermioncirq.experiments.hfvqe.objective import RestrictedHartreeFockObjective


def rhf_func_generator(rhf_objective: RestrictedHartreeFockObjective,
                       initial_occ_vec: Optional[Union[None, np.ndarray]] = None,
                       get_opdm_func: Optional[bool] = False):
    """
    Generate the energy, gradient, and unitary functions

    :param rhf_objective: objective function object
    :param initial_occ_vec: (optional) vector for occupation numbers of the alpha-opdm
    :return: functions for unitary, energy, gradient (in that order)
    """
    if initial_occ_vec is None:
        initial_opdm = np.diag([1] * rhf_objective.nocc + [0] * rhf_objective.nvirt)
    else:
        initial_opdm = np.diag(initial_occ_vec)

    def energy(params):
        u = unitary(params)
        final_opdm_aa = u @ initial_opdm @ np.conjugate(u).T
        tenergy = rhf_objective.energy_from_opdm(final_opdm_aa)
        return tenergy

    def gradient(params):
        u = unitary(params)
        final_opdm_aa = u @ initial_opdm @ np.conjugate(u).T
        return rhf_objective.global_gradient_opdm(params, final_opdm_aa).real

    def unitary(params):
        kappa = rhf_params_to_matrix(params,
                                     len(rhf_objective.occ) + len(rhf_objective.virt),
                                     rhf_objective.occ, rhf_objective.virt)
        return sp.linalg.expm(kappa)

    def get_opdm(params):
        u = unitary(params)
        return u @ initial_opdm @ np.conjugate(u).T

    if get_opdm_func:
        return unitary, energy, gradient, get_opdm
    return unitary, energy, gradient


def rhf_minimization(rhf_object, method='CG', initial_guess=None, verbose=True):
    """
    Perform Hartree-Fock energy minimization

    :param rhf_object: RestrictedHartreeFockObject
    :param method: (optional sp opt method)
    :return: sp result object
    """
    _, energy, gradient = rhf_func_generator(rhf_object)
    if initial_guess is None:
        init_guess = np.zeros(rhf_object.nocc * rhf_object.nvirt)
    else:
        init_guess = initial_guess.flatten()

    return sp.optimize.minimize(energy, init_guess, jac=gradient, method=method,
                                   options={'disp': verbose})
