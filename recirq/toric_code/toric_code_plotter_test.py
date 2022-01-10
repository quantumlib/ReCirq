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
from . import toric_code_plotter as plotter


class TestToricCodePlotter:
    @classmethod
    def setup_class(cls):
        cls.rows = 3
        cls.cols = 2
        cls.origin = cirq.GridQubit(0, 0)
        cls.row_vector = (1, 1)
        cls.code = tcr.ToricCodeRectangle(
            cls.origin, cls.row_vector, cls.rows, cls.cols
        )
        cls.plotter = plotter.ToricCodePlotter()
        cls.data = tcp.ToricCodePlaquettes.for_uniform_parity(cls.code, 0.5, -0.5)

    def test_plot_expectation_values(self):
        _ = self.plotter.plot_expectation_values(self.data)

    def test_plot_code(self):
        _ = self.plotter.plot_code(self.code)

    def test_get_patch(self):
        for row, col in self.code.x_plaquette_indices():
            _ = self.plotter.get_patch(self.data, row, col, x_basis=True)
        for row, col in self.code.z_plaquette_indices():
            _ = self.plotter.get_patch(self.data, row, col, x_basis=False)

    def test_cmap(self):
        assert self.plotter.cmap(x_basis=True) == self.plotter.x_cmap
        assert self.plotter.cmap(x_basis=False) == self.plotter.z_cmap

    @pytest.mark.parametrize("x_basis", [False, True])
    def test_make_colorbar(self, x_basis):
        _ = self.plotter.make_colorbar(x_basis)
