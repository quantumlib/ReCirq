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

from typing import Optional

import cirq
import pytest

from recirq.fermi_hubbard.decomposition import CPhaseEchoGate
from recirq.fermi_hubbard.execution import create_circuits
from recirq.fermi_hubbard.layouts import (
    LineLayout,
    QubitsLayout,
    ZigZagLayout
)
from recirq.fermi_hubbard.parameters import (
    FermiHubbardParameters,
    GaussianTrappingPotential,
    Hamiltonian,
    IndependentChainsInitialState,
    InitialState,
    PhasedGaussianSingleParticle,
    UniformSingleParticle,
    UniformTrappingPotential
)


@pytest.mark.parametrize('layout_type', [
    LineLayout, ZigZagLayout
])
def test_create_circuits_initial_trapping(layout_type: type) -> None:
    layout = layout_type(size=4)
    parameters = _create_test_parameters(
        layout,
        initial_state=IndependentChainsInitialState(
            up=UniformTrappingPotential(particles=2),
            down=GaussianTrappingPotential(
                particles=2,
                center=0.2,
                sigma=0.3,
                scale=0.4
            )
        )
    )

    initial, _, _ = create_circuits(parameters, trotter_steps=0)

    up1, up2, up3, up4 = layout.up_qubits
    down1, down2, down3, down4 = layout.down_qubits

    expected = cirq.Circuit([
        cirq.Moment(cirq.X(up1), cirq.X(up2), cirq.X(down1), cirq.X(down2)),
        cirq.Moment(
            cirq.PhasedISwapPowGate(phase_exponent=0.25,
                                    exponent=-0.7322795271987699).on(up2, up3),
            cirq.PhasedISwapPowGate(phase_exponent=0.25,
                                    exponent=-0.7502896835888355
                                    ).on(down2, down3),
        ),
        cirq.Moment((cirq.Z**0.0).on(up3), (cirq.Z**0.0).on(down3)),
        cirq.Moment(
            cirq.PhasedISwapPowGate(phase_exponent=0.25,
                                    exponent=-0.564094216848975).on(up1, up2),
            cirq.PhasedISwapPowGate(phase_exponent=0.25,
                                    exponent=-0.5768514417132005
                                    ).on(down1, down2),
            cirq.PhasedISwapPowGate(phase_exponent=0.25,
                                    exponent=-0.5640942168489748).on(up3, up4),
            cirq.PhasedISwapPowGate(phase_exponent=0.25,
                                    exponent=-0.5973464819433905
                                    ).on(down3, down4),
        ),
        cirq.Moment(
            (cirq.Z**0.0).on(up2),
            (cirq.Z**0.0).on(down2),
            (cirq.Z**0.0).on(up4),
            (cirq.Z**0.0).on(down4),
        ),
        cirq.Moment(
            cirq.PhasedISwapPowGate(phase_exponent=0.25,
                                    exponent=-0.26772047280123007).on(up2, up3),
            cirq.PhasedISwapPowGate(phase_exponent=0.25,
                                    exponent=-0.2994619470606122
                                    ).on(down2, down3),
        ),
        cirq.Moment((cirq.Z**0.0).on(up3), (cirq.Z**0.0).on(down3)),
    ])

    assert cirq.approx_eq(initial, expected)


@pytest.mark.parametrize('layout_type', [
    LineLayout, ZigZagLayout
])
def test_create_circuits_initial_single_particle(layout_type: type) -> None:
    layout = layout_type(size=4)
    parameters = _create_test_parameters(
        layout,
        initial_state=IndependentChainsInitialState(
            up=UniformSingleParticle(),
            down=PhasedGaussianSingleParticle(k=0.2, sigma=0.4, position=0.6)
        )
    )

    initial, _, _ = create_circuits(parameters, trotter_steps=0)

    up1, up2, up3, up4 = layout.up_qubits
    down1, down2, down3, down4 = layout.down_qubits

    expected = cirq.Circuit([
        cirq.Moment(
            cirq.X(up2),
            cirq.X(down2),
        ),
        cirq.Moment(
            cirq.PhasedISwapPowGate(phase_exponent=0.25,
                                    exponent=-0.5).on(up2, up3),
            cirq.PhasedISwapPowGate(phase_exponent=0.25,
                                    exponent=-0.5550026943884815
                                    ).on(down2, down3),
        ),
        cirq.Moment(
            cirq.PhasedISwapPowGate(phase_exponent=0.25,
                                    exponent=-0.5).on(up3, up4),
            cirq.PhasedISwapPowGate(phase_exponent=0.25,
                                    exponent=0.5).on(up1, up2),
            cirq.PhasedISwapPowGate(phase_exponent=0.25,
                                    exponent=-0.42338370832577876
                                    ).on(down3, down4),
            cirq.PhasedISwapPowGate(phase_exponent=0.25,
                                    exponent=0.3609629722553269
                                    ).on(down1, down2),
        ),
        cirq.Moment(
            (cirq.Z ** 0.0).on(up3),
            (cirq.Z ** 0.0).on(up4),
            (cirq.Z ** 0.0).on(up1),
            (cirq.Z ** 0.0).on(up2),
            (cirq.Z ** 0.004244131815783875).on(down3),
            (cirq.Z ** 0.02546479089470326).on(down4),
            (cirq.Z ** -0.03819718634205488).on(down1),
            (cirq.Z ** -0.016976527263135508).on(down2),
        ),
    ])

    assert cirq.approx_eq(initial, expected)


def test_create_circuits_trotter_line() -> None:
    layout = LineLayout(size=4)
    parameters = _create_test_parameters(layout, u=2.0)
    _, trotter, _ = create_circuits(parameters, trotter_steps=1)

    up1, up2, up3, up4 = layout.up_qubits
    down1, down2, down3, down4 = layout.down_qubits

    expected = cirq.Circuit([
        cirq.Moment(
            cirq.FSimGate(theta=-0.2, phi=0.0).on(up2, up1),
            cirq.FSimGate(theta=-0.2, phi=0.0).on(up4, up3),
            cirq.FSimGate(theta=-0.2, phi=0.0).on(down2, down1),
            cirq.FSimGate(theta=-0.2, phi=0.0).on(down4, down3),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=-0.2, phi=0.0).on(up3, up2),
            cirq.FSimGate(theta=-0.2, phi=0.0).on(down3, down2),
        ),
        cirq.Moment(
            (cirq.CZ**-0.12732395447351627).on(up1, down1),
            (cirq.CZ**-0.12732395447351627).on(up2, down2),
            (cirq.CZ**-0.12732395447351627).on(up3, down3),
            (cirq.CZ**-0.12732395447351627).on(up4, down4),
        ),
    ])

    assert cirq.approx_eq(trotter, expected)


def test_create_circuits_trotter_zigzag() -> None:
    layout = ZigZagLayout(size=4)
    parameters = _create_test_parameters(layout, u=2.0)
    _, trotter, _ = create_circuits(parameters, trotter_steps=1)

    up1, up2, up3, up4 = layout.up_qubits
    down1, down2, down3, down4 = layout.down_qubits

    expected = cirq.Circuit([
        cirq.Moment(
            cirq.FSimGate(theta=-0.2, phi=0.0).on(up2, up1),
            cirq.FSimGate(theta=-0.2, phi=0.0).on(up4, up3),
            cirq.FSimGate(theta=-0.2, phi=0.0).on(down2, down1),
            cirq.FSimGate(theta=-0.2, phi=0.0).on(down4, down3),
        ),
        cirq.Moment(
            (cirq.CZ**-0.12732395447351627).on(down2, up2),
            (cirq.CZ**-0.12732395447351627).on(down4, up4),
            CPhaseEchoGate().on(down1),
            CPhaseEchoGate().on(down3),
            CPhaseEchoGate().on(up1),
            CPhaseEchoGate().on(up3),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=-1.5707963267948966, phi=0.0).on(up3, up2),
            cirq.FSimGate(theta=-1.5707963267948966, phi=0.0).on(down3, down2),
        ),
        cirq.Moment(
            (cirq.CZ**-0.12732395447351627).on(down1, up1),
            (cirq.CZ**-0.12732395447351627).on(down2, up2),
            CPhaseEchoGate().on(down3),
            CPhaseEchoGate().on(down4),
            CPhaseEchoGate().on(up3),
            CPhaseEchoGate().on(up4),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=-1.7707963267948965, phi=0.0).on(up3, up2),
            cirq.FSimGate(theta=-1.7707963267948965, phi=0.0).on(down3, down2),
        ),
        cirq.Moment(
            cirq.Z(up2), cirq.Z(up3), cirq.Z(down2), cirq.Z(down3),
        ),
    ])

    assert cirq.approx_eq(trotter, expected)


@pytest.mark.parametrize('layout_type', [
    LineLayout, ZigZagLayout
])
def test_create_circuits_measurement(layout_type: type) -> None:
    layout = layout_type(size=4)
    parameters = _create_test_parameters(layout)
    _, _, measurement = create_circuits(parameters, trotter_steps=0)
    assert cirq.approx_eq(
        measurement,
        cirq.Circuit(cirq.measure(*layout.all_qubits, key='z')))


def _create_test_parameters(layout: QubitsLayout,
                            u: float = 0.0,
                            initial_state: Optional[InitialState] = None
                            ) -> FermiHubbardParameters:

    hamiltonian = Hamiltonian(
        sites_count=layout.size,
        j=1.0,
        u=u
    )

    if initial_state is None:
        initial_state = IndependentChainsInitialState(
            up=GaussianTrappingPotential(
                particles=3, center=0.5, sigma=1, scale=1),
            down=UniformTrappingPotential(particles=3)
        )

    return FermiHubbardParameters(
        hamiltonian=hamiltonian,
        initial_state=initial_state,
        layout=layout,
        dt=0.2
    )