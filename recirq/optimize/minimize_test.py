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

import recirq


def sum_of_squares(x):
    return np.sum(x**2).item()


def test_minimize():
    x0 = np.random.randn(5)

    result = recirq.optimize.minimize(sum_of_squares,
                                      x0,
                                      method='mgd',
                                      sample_radius=1e-1,
                                      n_sample_points=21,
                                      rate=1e-1,
                                      tol=1e-7,
                                      known_values=None)
    assert result.message == 'Optimization converged successfully.'

    result = recirq.optimize.minimize(sum_of_squares, x0, method='Nelder-Mead')
    assert result.message == 'Optimization terminated successfully.'
