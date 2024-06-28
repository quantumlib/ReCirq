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
"""Tests for afqmc_generators.py."""

import openfermion as of

from recirq.qcqmc import afqmc_generators, layer_spec


def test_get_pp_plus_gate_generators():
    n_elec = 4
    heuristic_layers = (
        layer_spec.LayerSpec(base_gate="charge_charge", layout="in_pair"),
    )
    gate_generators = afqmc_generators.get_pp_plus_gate_generators(
        n_elec=n_elec, heuristic_layers=heuristic_layers, do_pp=False
    )
    assert len(gate_generators) == 4
    assert gate_generators[0] == of.FermionOperator("2^ 2 4^ 4", 1.0)
    assert gate_generators[1] == of.FermionOperator("3^ 3 5^ 5", 1.0)
    assert gate_generators[2] == of.FermionOperator("0^ 0 6^ 6", 1.0)
    assert gate_generators[3] == of.FermionOperator("1^ 1 7^ 7", 1.0)

    gate_generators_w_pp = afqmc_generators.get_pp_plus_gate_generators(
        n_elec=n_elec, heuristic_layers=heuristic_layers, do_pp=True
    )
    assert len(gate_generators_w_pp) == 6
    assert gate_generators_w_pp[0] == of.FermionOperator(
        "2 3 4^ 5^", -1.0j
    ) + of.FermionOperator("5 4 3^ 2^", +1.0j)
    assert gate_generators_w_pp[1] == of.FermionOperator(
        "0 1 6^ 7^", -1.0j
    ) + of.FermionOperator("7 6 1^ 0^", 1.0j)
    assert gate_generators_w_pp[2:] == gate_generators


def test_get_pair_hopping_gate_generators():
    n_pairs = 2
    n_elec = 4
    gate_generators = afqmc_generators.get_pair_hopping_gate_generators(
        n_pairs=n_pairs, n_elec=n_elec
    )
    assert len(gate_generators) == 2
    assert gate_generators[0] == of.FermionOperator(
        "2 3 4^ 5^", -1.0j
    ) + of.FermionOperator("5 4 3^ 2^", 1.0j)
    assert gate_generators[1] == of.FermionOperator(
        "0 1 6^ 7^", -1.0j
    ) + of.FermionOperator("7 6 1^ 0^", 1.0j)


def test_get_charge_charge_generator():
    """Test get_charge_charge_generator."""
    indices = (2, 3)
    gate_generator = afqmc_generators.get_charge_charge_generator(indices=indices)
    assert gate_generator == of.FermionOperator("2^ 2 3^ 3", 1.0)


def test_get_givens_generator():
    """Test get_givens_generator."""
    indices = (2, 3)
    gate_generator = afqmc_generators.get_givens_generator(indices=indices)
    assert gate_generator == of.FermionOperator("2^ 3", 1.0j) - of.FermionOperator(
        "3^ 2", 1.0j
    )


def test_get_layer_generators():
    """Test get_layer_generators."""
    n_elec = 4
    layer_spec_in_pair = layer_spec.LayerSpec(
        base_gate="charge_charge", layout="in_pair"
    )
    gate_generators = afqmc_generators.get_layer_generators(
        layer_spec=layer_spec_in_pair, n_elec=n_elec
    )
    assert len(gate_generators) == 4
    assert gate_generators[0] == of.FermionOperator("2^ 2 4^ 4", 1.0)
    assert gate_generators[1] == of.FermionOperator("3^ 3 5^ 5", 1.0)
    assert gate_generators[2] == of.FermionOperator("0^ 0 6^ 6", 1.0)
    assert gate_generators[3] == of.FermionOperator("1^ 1 7^ 7", 1.0)

    layer_spec_cross_pair = layer_spec.LayerSpec(
        base_gate="givens", layout="cross_pair"
    )
    gate_generators = afqmc_generators.get_layer_generators(
        layer_spec=layer_spec_cross_pair, n_elec=n_elec
    )
    assert len(gate_generators) == 2
    assert gate_generators[0] == of.FermionOperator("0^ 4", -1.0j) + of.FermionOperator(
        "4^ 0", 1.0j
    )
    assert gate_generators[1] == of.FermionOperator("1^ 5", -1.0j) + of.FermionOperator(
        "5^ 1", 1.0j
    )


def test_get_heuristic_gate_generators():
    """Test get_heuristic_gate_generators."""
    n_elec = 4
    heuristic_layers = (
        layer_spec.LayerSpec(base_gate="charge_charge", layout="in_pair"),
        layer_spec.LayerSpec(base_gate="givens", layout="cross_pair"),
    )
    gate_generators = afqmc_generators.get_heuristic_gate_generators(
        n_elec=n_elec, layer_specs=heuristic_layers
    )
    assert len(gate_generators) == 6
    assert gate_generators[0] == of.FermionOperator("2^ 2 4^ 4", 1.0)
    assert gate_generators[1] == of.FermionOperator("3^ 3 5^ 5", 1.0)
    assert gate_generators[2] == of.FermionOperator("0^ 0 6^ 6", 1.0)
    assert gate_generators[3] == of.FermionOperator("1^ 1 7^ 7", 1.0)
    assert gate_generators[4] == of.FermionOperator("0^ 4", -1.0j) + of.FermionOperator(
        "4^ 0", 1.0j
    )
    assert gate_generators[5] == of.FermionOperator("1^ 5", -1.0j) + of.FermionOperator(
        "5^ 1", 1.0j
    )
