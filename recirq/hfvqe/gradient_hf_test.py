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
from recirq.hfvqe.gradient_hf import (rhf_func_generator, rhf_minimization)
from recirq.hfvqe.molecular_example import make_h6_1_3


def test_rhf_func_gen():
    rhf_objective, molecule, parameters, _, _ = make_h6_1_3()
    ansatz, energy, _ = rhf_func_generator(rhf_objective)
    assert np.isclose(molecule.hf_energy, energy(parameters))

    ansatz, energy, _, opdm_func = rhf_func_generator(rhf_objective,
                                                      initial_occ_vec=[1] * 3 +
                                                      [0] * 3,
                                                      get_opdm_func=True)
    assert np.isclose(molecule.hf_energy, energy(parameters))
    test_opdm = opdm_func(parameters)
    u = ansatz(parameters)
    initial_opdm = np.diag([1] * 3 + [0] * 3)
    final_odpm = u @ initial_opdm @ u.T
    assert np.allclose(test_opdm, final_odpm)

    result = rhf_minimization(rhf_objective, initial_guess=parameters)
    assert np.allclose(result.x, parameters)
