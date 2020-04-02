# Copyright 2020 Google
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

from timeit import default_timer as timer

import networkx as nx
import numpy as np
import scipy.optimize

import recirq
from recirq.qaoa.simulation import create_ZZ_HamC, ising_qaoa_grad

EVEN_DEGREE_ONLY, ODD_DEGREE_ONLY = 0, 1


@recirq.json_serializable_dataclass(namespace='recirq.qaoa',
                                    registry=recirq.Registry,
                                    frozen=True)
class OptimizationResult:
    """The result of a classical angle optimization for QAOA.

    You may want to use :py:func:`optimize_instance_interp_heuristic`.

    Attributes:
        p: QAOA depth hyperparameter p. The number of parameters is 2*p.
        f_val: The value of the <C> at the optimal angles
        gammas: An array of p gamma angles
        betas: An array of p beta angles
        min_c: The minimum value of C over all bitstrings
        max_c: The maximum value of C over all bitstrings
    """
    p: int
    f_val: float
    gammas: np.ndarray
    betas: np.ndarray
    min_c: float
    max_c: float


def optimize_instance_interp_heuristic(graph: nx.Graph,
                                       p_max: int = 10,
                                       param_guess_at_p1=None,
                                       node_to_index_map=None,
                                       dtype=np.complex128,
                                       verbose=False):
    r"""
    Given a graph, find QAOA parameters that minimizes C=\sum_{<ij>} w_{ij} Z_i Z_j

    Uses the interpolation-based heuristic from arXiv:1812.01041

    Args:
        graph: The problem.
        p_max: Optimize each p up to this value.
        param_guess_at_p1: Initial angles at p=1. If None, try 10 initial
            parameter guesses and keep the best one. The first guess will
            be beta=gamma=pi/8. The following guesses will be random.
        node_to_index_map: dictionary that maps nodes to 0-based integer indices
            (optional, default None, which means use mapping based on iteration)
        dtype: The numpy datatype used during construction of the Hamiltonian
            matrix.
        verbose: Whether to print out status updates.

    Returns:
        A list of :py:class:`OptimizationResult`
    """
    # TODO: remove node_to_index_map; mandate 0-indexing.

    # construct function to be passed to scipy.optimize.minimize
    HamC = create_ZZ_HamC(graph, True, node_to_index_map, dtype=dtype)
    N = graph.number_of_nodes()

    def qaoa_fun(param):
        return ising_qaoa_grad(N, HamC, param, flag_z2_sym=True, dtype=dtype)

    min_c = np.real_if_close(np.min(HamC))
    max_c = np.real_if_close(np.max(HamC))

    # check if the node degrees are always odd or even
    # TODO: Why do we mod 2 twice?
    degree_list = np.array([deg for node, deg in graph.degree()]) % 2
    if np.all(degree_list % 2 == 0):
        parity = EVEN_DEGREE_ONLY
    elif np.all(degree_list % 2 == 1):
        parity = ODD_DEGREE_ONLY
    else:
        # Not all of one or another
        parity = None

    # start the optimization process incrementally from p = 1 to p_max
    Fvals = p_max * [0]
    params = p_max * [None]

    for p in range(p_max):  # note here, p goes from 0 to p_max - 1

        # use heuristic to produce good initial guess of parameters
        if p == 0:
            # Note: the following might be `None`. See the todo below
            param0 = param_guess_at_p1
        elif p == 1:
            param0 = [params[0][0], params[0][0], params[0][1], params[0][1]]
        else:
            xp = np.linspace(0, 1, p)
            xp1 = np.linspace(0, 1, p + 1)
            param0 = np.concatenate([np.interp(xp1, xp, params[p - 1][:p]),
                                     np.interp(xp1, xp, params[p - 1][p:])])

        start = timer()
        if param0 is not None:
            results = scipy.optimize.minimize(
                qaoa_fun, param0, jac=True, method='BFGS')
        else:
            # run with 10 random guesses of parameters and keep best one
            # will only apply to the lowest depth (p=0 here)
            # TODO: This is an inappropriate place for this `else` clause.
            # TODO: refactor out to its own function and call it during
            #       input validation.

            # first run with a guess known to work most of the time
            results = scipy.optimize.minimize(
                qaoa_fun, [np.ones(p + 1) * np.pi / 8, -np.ones(p + 1) * np.pi / 8],
                jac=True, method='BFGS')

            for _ in range(1, 10):
                # some reasonable random guess
                param0 = np.concatenate([np.random.rand(p + 1) * np.pi / 2,
                                         -np.ones(p + 1) * np.pi / 8])
                test_results = scipy.optimize.minimize(qaoa_fun, param0, jac=True, method='BFGS')
            if test_results.fun < results.fun:  # found a better minimum
                results = test_results

        if verbose:
            end = timer()
            print(f'-- p={p + 1}, '
                  f'F = {results.fun:0.3f} / {min_c}, '
                  f'nfev={results.nfev}, '
                  f'time={end - start:0.2f} s')

        Fvals[p] = results.fun
        params[p] = fix_param_gauge(results.x, degree_parity=parity)

    return [
        OptimizationResult(
            p=p,
            f_val=f_val,
            gammas=param[:p],
            betas=param[p:],
            min_c=min_c,
            max_c=max_c
        ) for p, f_val, param in zip(range(1, p_max + 1), Fvals, params)
    ]


def fix_param_gauge(param, gamma_period=np.pi, beta_period=np.pi / 2, degree_parity=None):
    """Use symmetries to reduce redundancies in the parameter space

    This is useful for the interp heuristic that relies on smoothness of parameters

    Based on arXiv:1812.01041 and https://github.com/leologist/GenQAOA/
    """
    p = len(param) // 2

    gammas = np.array(param[:p]) / gamma_period
    betas = -np.array(param[p:]) / beta_period
    # we expect gamma to be positive and beta to be negative,
    # so flip sign of beta for now and flip it back later

    # reduce the parameters to be between [0, 1] * period
    gammas = gammas % 1
    betas = betas % 1

    # use time-reversal symmetry to make first gamma small
    if (gammas[0] > 0.25 and gammas[0] < 0.5) or gammas[0] > 0.75:
        gammas = -gammas % 1
        betas = -betas % 1

    # further simplification if all nodes have same degree parity
    if degree_parity == EVEN_DEGREE_ONLY:  # Every node has even degree
        gamma_period = np.pi / 2
        gammas = (gammas * 2) % 1
    elif degree_parity == ODD_DEGREE_ONLY:  # Every node has odd degree
        for i in range(p):
            if gammas[i] > 0.5:
                gammas[i] = gammas[i] - 0.5
                betas[i:] = 1 - betas[i:]

    for i in range(1, p):
        # try to impose smoothness of gammas
        delta = gammas[i] - gammas[i - 1]
        if delta >= 0.5:
            gammas[i] -= 1
        elif delta <= -0.5:
            gammas[i] += 1

        #  try to impose smoothness of betas
        delta = betas[i] - betas[i - 1]
        if delta >= 0.5:
            betas[i] -= 1
        elif delta <= -0.5:
            betas[i] += 1

    return np.concatenate((gammas * gamma_period, -betas * beta_period)).tolist()
