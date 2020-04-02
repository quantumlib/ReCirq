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


import numpy as np
import networkx as nx

from recirq.qaoa.classical_angle_optimization import optimize_instance_interp_heuristic, \
    fix_param_gauge, ODD_DEGREE_ONLY, EVEN_DEGREE_ONLY


def test_optimize_instance_interp_heuristic():
    graph = nx.random_regular_graph(3, 10)
    nx.set_edge_attributes(graph, name='weight', values=1)
    opts = optimize_instance_interp_heuristic(
        graph=graph,
        p_max=7,
        verbose=False,
    )
    assert len(opts) == 7
    for i, opt in enumerate(opts):
        p = i + 1
        assert opt.p == p
        assert len(opt.betas) == p
        assert len(opt.gammas) == p


def test_fix_param_gauge():
    # Test to show optimize.fix_param_gauge()
    # can fully reduce redundancies of QAOA parameters

    tolerance = 1e-10

    # original parameters (at p=3) in preferred gauge
    param = np.array([0.2, 0.4, 0.7, -0.6, -0.5, -0.3])

    # copy of parameters
    param2 = param.copy()

    # test that gammas are periodic in pi
    param2[:3] += np.random.choice([1, -1], 3) * np.pi
    param_fixed = fix_param_gauge(param2)
    assert np.linalg.norm(param - param_fixed) <= tolerance

    # test that betas are periodic in pi/2
    param2[3:] += np.random.choice([1, -1], 3) * np.pi / 2
    param_fixed = fix_param_gauge(param2)
    assert np.linalg.norm(param - param_fixed) <= tolerance

    # Case: ODD_DEGREE_ONLY
    # test that shifting gamma_i by (n+1/2)*pi and beta_{j>=i} -> -beta_{j>=i}
    # gives equivalent parameters
    param2[2] -= np.pi / 2
    param2[5] = -param2[5]
    param_fixed = fix_param_gauge(param2, degree_parity=ODD_DEGREE_ONLY)
    assert np.linalg.norm(param - param_fixed) <= tolerance

    param2[1] += np.pi * 3 / 2
    param2[4:] = -param2[4:]
    param_fixed = fix_param_gauge(param2, degree_parity=ODD_DEGREE_ONLY)
    assert np.linalg.norm(param - param_fixed) <= tolerance

    # test that two inequivalent parameters should not be the same after fixing gauge
    param2 = param.copy()
    param2[0] -= np.pi * 3 / 2
    param_fixed = fix_param_gauge(param2, degree_parity=ODD_DEGREE_ONLY)
    assert np.linalg.norm(param - param_fixed) > tolerance

    # Case: EVEN_DEGREE_ONLY
    # test parameters are periodic in pi/2
    param2 = param.copy()
    param2 += np.random.choice([1, -1], 6) * np.pi / 2
    param_fixed = fix_param_gauge(param2, degree_parity=EVEN_DEGREE_ONLY)
    assert np.linalg.norm(param - param_fixed) <= tolerance
