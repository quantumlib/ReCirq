# Copyright 2024 Google
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
import pytest

from recirq.qcqmc import layer_spec


@pytest.mark.parametrize("base_gate", ("charge_charge", "givens"))
@pytest.mark.parametrize("layout", ("in_pair", "cross_pair", "cross_spin"))
def test_layer_spec(base_gate, layout):
    ls = layer_spec.LayerSpec(base_gate=base_gate, layout=layout)
    ls2 = cirq.read_json(json_text=cirq.to_json(ls))
    assert ls2 == ls
    with pytest.raises(ValueError, match=r"base_gate is set*"):
        ls = layer_spec.LayerSpec(base_gate=base_gate + "y", layout=layout)
    with pytest.raises(ValueError, match=r"layout is set*"):
        ls = layer_spec.LayerSpec(base_gate=base_gate, layout=layout + "y")
    with pytest.raises(ValueError, match=r"base_gate is set*"):
        ls = layer_spec.LayerSpec(base_gate=base_gate + "x", layout=layout + "y")


@pytest.mark.parametrize("n_elec", range(1, 10))
def test_get_indices_heuristic_layer_cross_pair(n_elec):
    n_pairs = max(n_elec // 2 - 1, 0)
    n_terms = 0
    for x in layer_spec.get_indices_heuristic_layer_cross_pair(n_elec):
        assert len(x) == 2
        if n_terms % 2 == 0:
            assert x[0] + x[1] == 2 * n_elec - 4
        else:
            assert x[0] + x[1] == 2 * n_elec - 2
        n_terms += 1
    assert n_terms == 2 * n_pairs


@pytest.mark.parametrize("n_elec", range(1, 10))
def test_get_indices_heuristic_layer_cross_spin(n_elec):
    n_pairs = n_elec // 2
    n_terms = 0
    for x in layer_spec.get_indices_heuristic_layer_cross_spin(n_elec):
        assert len(x) == 2
        assert x[1] - x[0] == 1
        n_terms += 1
    assert n_terms == 2 * n_pairs


@pytest.mark.parametrize("n_elec", range(1, 10))
def test_get_indices_heuristic_layer_in_pair(n_elec):
    n_pairs = n_elec // 2
    n_terms = 0
    for x in layer_spec.get_indices_heuristic_layer_in_pair(n_elec):
        assert len(x) == 2
        if n_terms % 2 == 0:
            assert x[1] + x[0] == 2 * n_elec - 2
        else:
            assert x[1] + x[0] == 2 * n_elec
        n_terms += 1
    assert n_terms == 2 * n_pairs
