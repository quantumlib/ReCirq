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

from recirq.fermi_hubbard.parameters import (
    GaussianTrappingPotential,
    PhasedGaussianSingleParticle,
    UniformSingleParticle,
    UniformTrappingPotential
)

import numpy as np


def test_phased_gaussian_single_particle():

    chain = PhasedGaussianSingleParticle(k=1.2 * 7,
                                         sigma=1.2 / 7,
                                         position=1.5 / 7)
    amplitudes = chain.get_amplitudes(sites_count=8)

    np.testing.assert_allclose(
        amplitudes,
        [
            -0.09060882 - 0.3883731j,
            0.46578491 - 0.3186606j,
            0.46578491 + 0.3186606j,
            -0.09060882 + 0.3883731j,
            -0.19714994 + 0.02810304j,
            -0.034451 - 0.06124629j,
            0.0111212 - 0.01354051j,
            0.00293383 + 0.00096188j
        ],
        rtol=1e-5
    )

    assert np.isclose(np.linalg.norm(amplitudes), 1)


def test_uniform_single_particle():
    chain = UniformSingleParticle()
    amplitudes = chain.get_amplitudes(sites_count=8)
    np.testing.assert_allclose(amplitudes, 1 / np.sqrt(8))


def test_gaussian_trapping_potential():

    chain = GaussianTrappingPotential(
        particles=2,
        center=0.5,
        sigma=1 / 7,
        scale=-4)
    potential = chain.get_potential(sites_count=8)

    np.testing.assert_allclose(
        potential,
        [
            -0.00874996,
            -0.17574773,
            -1.29860987,
            -3.52998761,
            -3.52998761,
            -1.29860987,
            -0.17574773,
            -0.00874996
        ],
        rtol=1e-5
    )


def test_uniform_trapping_potential():
    chain = UniformTrappingPotential(particles=2)
    potential = chain.get_potential(sites_count=8)
    np.testing.assert_allclose(potential, 0.0)