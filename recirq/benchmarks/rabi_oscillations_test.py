# Copyright 2022 The Cirq Developers
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

import cirq
from recirq.benchmarks import rabi_oscillations


def test_rabi_oscillations():
    """Check that the excited state population matches the ideal case within a
    small statistical error.
    """
    simulator = cirq.sim.Simulator()
    qubit = cirq.GridQubit(0, 0)
    results = rabi_oscillations(
        simulator, qubit, np.pi, repetitions=1000, num_points=1000
    )
    data = np.asarray(results.data)
    angles = data[:, 0]
    actual_pops = data[:, 1]
    target_pops = 0.5 - 0.5 * np.cos(angles)
    rms_err = np.sqrt(np.mean((target_pops - actual_pops) ** 2))
    assert rms_err < 0.1
