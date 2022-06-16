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

from itertools import product
import cirq
import numpy as np
import pytest

from recirq.fermi_hubbard.decomposition import (
    _corrected_cphase_ops,
    _corrected_gk_ops,
    _corrected_sqrt_iswap_ops,
    GateDecompositionError,
    ParticleConservingParameters
)


def create_particle_conserving_matrix(theta: float,
                                      delta: float,
                                      chi: float,
                                      gamma: float,
                                      phi: float) -> np.array:
    """Matrix form of the most general particle conserving two-qubit gate.

    The returned unitary is:

     [[1,                        0,                        0,         0],
      [0,  c e^{-i (gamma + delta)},  -is e^{-i (gamma - chi)},       0],
      [0,  -is e^{-i (gamma + chi)},  c e^{-i (gamma - delta)},       0],
      [0,                        0,       0,      e^{-i (2 gamma + phi}]]
    """

    matrix = [
        [1., 0., 0., 0.],
        [0.,
         np.exp(-1.j * (gamma + delta)) * np.cos(theta),
         -1.j * np.exp(-1.j * (gamma - chi)) * np.sin(theta),
         0.],
        [0.,
         -1.j * np.exp(-1.j * (gamma + chi)) * np.sin(theta),
         np.exp(-1.j * (gamma - delta)) * np.cos(theta),
         0],
        [0., 0., 0., np.exp(-1.j * (2. * gamma + phi))]
    ]
    return np.asarray(matrix)


@pytest.mark.parametrize('delta,chi,gamma',
                         product(np.linspace(0, 2 * np.pi, 5),
                                 np.linspace(0, 2 * np.pi, 5),
                                 np.linspace(0, 2 * np.pi, 5)))
def test_corrected_sqrt_iswap_ops(delta: float, chi: float, gamma: float
                                  ) -> None:
    a, b = cirq.LineQubit.range(2)

    matrix = cirq.Circuit(_corrected_sqrt_iswap_ops(
        qubits=(a, b),
        parameters=ParticleConservingParameters(
            delta=delta,
            chi=chi,
            gamma=gamma))).unitary(qubit_order=(a, b), dtype=np.complex128)

    expected = create_particle_conserving_matrix(
        np.pi / 4, -delta, -chi, -gamma, 0)

    assert cirq.equal_up_to_global_phase(matrix, expected)


@pytest.mark.parametrize('theta', np.linspace(0, 2 * np.pi, 5))
def test_corr_hop_gate(theta: float) -> None:
    a, b = cirq.LineQubit.range(2)

    matrix = cirq.Circuit(_corrected_gk_ops(
        qubits=(a, b),
        angle=theta,
        phase_exponent=0.0,
        parameters=ParticleConservingParameters(
            theta=theta,
            delta=0,
            chi=0,
            gamma=0))).unitary(qubit_order=(a, b), dtype=np.complex128)

    expected = create_particle_conserving_matrix(theta, 0, 0, 0, 0)

    assert cirq.equal_up_to_global_phase(matrix, expected)


@pytest.mark.parametrize('cphase', np.linspace(-0.32, -0.8 * np.pi, 5))
def test_corrected_cphase_ops(cphase: float) -> None:
    a, b = cirq.LineQubit.range(2)

    matrix = cirq.Circuit(_corrected_cphase_ops(
        qubits=(a, b),
        angle=cphase,
        parameters=ParticleConservingParameters(
            theta=np.pi / 4,
            delta=0,
            chi=0,
            gamma=0,
            phi=0))).unitary(qubit_order=(a, b), dtype=np.complex128)
    expected = create_particle_conserving_matrix(0, 0, 0, 0, cphase)
    assert cirq.equal_up_to_global_phase(matrix, expected)


def test_corrected_cphase_ops_throws() -> None:
    a, b = cirq.LineQubit.range(2)
    with pytest.raises(GateDecompositionError):
        _corrected_cphase_ops(
            qubits=(a, b),
            angle=np.pi / 13,
            parameters=ParticleConservingParameters(
                theta=np.pi / 4,
                delta=0,
                chi=0,
                gamma=0,
                phi=np.pi / 24))
