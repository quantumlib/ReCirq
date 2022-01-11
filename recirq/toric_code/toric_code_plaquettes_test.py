# Copyright 2022 Google
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

import cirq
import pandas as pd

from . import toric_code_rectangle as tcr
from . import toric_code_plaquettes as tcp


ORIGIN = cirq.GridQubit(0, 0)
ROW_VECTOR = (1, 1)


def q(i: int, j: int) -> cirq.GridQubit:
    return cirq.GridQubit(i, j)


def test_for_uniform_parity():
    small_code = tcr.ToricCodeRectangle(ORIGIN, ROW_VECTOR, 1, 2)
    x_value = 0.3
    z_value = -1.0
    plaquettes = tcp.ToricCodePlaquettes.for_uniform_parity(
        small_code, x_value, z_value
    )

    expected_x_plaquettes = len(list(small_code.x_plaquette_indices()))
    expected_z_plaquettes = len(list(small_code.z_plaquette_indices()))
    assert len(plaquettes.x_plaquettes) == expected_x_plaquettes
    assert len(plaquettes.z_plaquettes) == expected_z_plaquettes

    assert all(value == x_value for value in plaquettes.x_plaquettes.values())
    assert all(value == z_value for value in plaquettes.z_plaquettes.values())


def test_expectation_value_x():
    small_code = tcr.ToricCodeRectangle(ORIGIN, ROW_VECTOR, 1, 2)
    # Construct data with -1 on plaquette (0, 0) and +1 on plaquette (0, 1)
    data = pd.DataFrame([0b1101100, 0b0010000, 0b0011101, 0b1000110])
    assert tcp.ToricCodePlaquettes.expectation_value(small_code, data, 0, 0, True) == -1
    assert tcp.ToricCodePlaquettes.expectation_value(small_code, data, 0, 1, True) == 1

    # Construct data with 0.5 on plaquette (0, 0) and 0 on plaquette (0, 1)
    data = pd.DataFrame([0b0000000, 0b1000000, 0b1010101, 0b1010011])
    assert (
        tcp.ToricCodePlaquettes.expectation_value(small_code, data, 0, 0, True) == 0.5
    )
    assert tcp.ToricCodePlaquettes.expectation_value(small_code, data, 0, 1, True) == 0


def test_compute_parity():
    assert tcp.ToricCodePlaquettes.compute_parity(0b0001, {0}, 2) == 1
    assert tcp.ToricCodePlaquettes.compute_parity(0b0001, {0}, 1) == -1
    assert tcp.ToricCodePlaquettes.compute_parity(0b1111, {0, 1}, 4) == 1
    assert tcp.ToricCodePlaquettes.compute_parity(0b1111, {0, 1, 2}, 4) == -1
