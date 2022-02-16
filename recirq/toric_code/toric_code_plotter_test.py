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

"""Tests for toric_code_plotter."""
import pytest
import cirq

from . import toric_code_rectangle as tcr
from . import toric_code_plaquettes as tcp
from . import toric_code_plotter as tcplot


def test_plot_expectation_values():
    code = tcr.ToricCodeRectangle(cirq.GridQubit(0, 0), (1, 1), 3, 2)
    data = tcp.ToricCodePlaquettes.for_uniform_parity(code, 0.5, -0.5)
    _ = tcplot.ToricCodePlotter().plot_expectation_values(data)


def test_plot_code():
    code = tcr.ToricCodeRectangle(cirq.GridQubit(0, 0), (1, 1), 3, 2)
    _ = tcplot.ToricCodePlotter().plot_code(code)


def test_get_patch():
    code = tcr.ToricCodeRectangle(cirq.GridQubit(0, 0), (1, 1), 3, 2)
    data = tcp.ToricCodePlaquettes.for_uniform_parity(code, 0.5, -0.5)
    plotter = tcplot.ToricCodePlotter()
    for row, col in code.x_plaquette_indices():
        _ = plotter.get_patch(data, row, col, x_basis=True)
    for row, col in code.z_plaquette_indices():
        _ = plotter.get_patch(data, row, col, x_basis=False)


def test_cmap():
    plotter = tcplot.ToricCodePlotter()
    assert plotter.cmap(x_basis=True) == plotter.x_cmap
    assert plotter.cmap(x_basis=False) == plotter.z_cmap


@pytest.mark.parametrize("x_basis", [False, True])
def test_make_colorbar(x_basis):
    _ = tcplot.ToricCodePlotter().make_colorbar(x_basis)
