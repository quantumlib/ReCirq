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
r"""Module for building generators for various fermionic wavefunction ansatzes for AFQMC.

Given a unitary defined in terms of fermionic modes of the form
$$
U = e^{\alpha a_p^\dagger a_q}
$$
the generator of this unitary (gate) is $a_p^\dagger a_q$ where $a_p^{(\dagger)}
$ is the fermionic annihilation (creation) operator for a fermion in spin
orbital $p$.
"""

from typing import List, Sequence, Tuple

import openfermion as of

from recirq.qcqmc import layer_spec as lspec


def get_pp_plus_gate_generators(
    *,
    n_elec: int,
    heuristic_layers: Tuple[lspec.LayerSpec, ...],
    do_pp: bool = True,
) -> List[of.FermionOperator]:
    """Get PP+ gate generators for a given number of electrons.

    Args:
        n_elec: The number of electrons
        heuristic_layers: The heuristic HW layers
        do_pp: Whether to add the PP gates to the ansatz too.

    Returns:
        The list of generators necessary to construct the ansatz.
    """
    heuristic_gate_generators = get_heuristic_gate_generators(n_elec, heuristic_layers)
    if not do_pp:
        return heuristic_gate_generators

    n_pairs = n_elec // 2
    pair_gate_generators = get_pair_hopping_gate_generators(n_pairs, n_elec)
    return pair_gate_generators + heuristic_gate_generators


def get_pair_hopping_gate_generators(
    n_pairs: int, n_elec: int
) -> List[of.FermionOperator]:
    """Get the generators of the pair-hopping unitaries.

    Args:
        n_pairs: The number of pair coupling terms.
        n_elec: The total number of electrons.

    Returns:
        A list of gate generators
    """
    gate_generators = []
    for pair in range(n_pairs):
        to_a = n_elec + 2 * pair
        to_b = n_elec + 2 * pair + 1
        from_a = n_elec - 2 * pair - 2
        from_b = n_elec - 2 * pair - 1

        fop_string = f"{to_b} {to_a} {from_b}^ {from_a}^"

        gate_generator = of.FermionOperator(fop_string, 1.0)
        gate_generator = 1j * (gate_generator - of.hermitian_conjugated(gate_generator))
        gate_generators.append(gate_generator)

    return gate_generators


def get_charge_charge_generator(indices: Tuple[int, int]) -> of.FermionOperator:
    """Returns the generator for density evolution between the indices

    Args:
        indices: The indices to for charge-charge terms.:w

    Returns:
        The generator for density evolution for this pair of electrons.
    """

    fop_string = "{:d}^ {:d} {:d}^ {:d}".format(
        indices[0], indices[0], indices[1], indices[1]
    )
    gate_generator = of.FermionOperator(fop_string, 1.0)

    return gate_generator


def get_givens_generator(indices: Tuple[int, int]) -> of.FermionOperator:
    """Returns the generator for givens rotation between two orbitalspec.

    Args:
        indices: The two indices for the givens rotation.

    Returns:
        The givens generator for evolution for this pair of electrons.
    """

    fop_string = "{:d}^ {:d}".format(indices[0], indices[1])
    gate_generator = of.FermionOperator(fop_string, 1.0)
    gate_generator = 1j * (gate_generator - of.hermitian_conjugated(gate_generator))

    return gate_generator


def get_layer_generators(
    layer_spec: lspec.LayerSpec, n_elec: int
) -> List[of.FermionOperator]:
    """Gets the generators for rotations in a hardware efficient layer of the ansatz.

    Args:
        layer_spec: The layer specification.
        n_elec: The number of electrons.

    Returns:
        A list of generators for the layers.
    """

    indices_list = lspec.get_layer_indices(layer_spec, n_elec)

    gate_funcs = {
        "givens": get_givens_generator,
        "charge_charge": get_charge_charge_generator,
    }
    gate_func = gate_funcs[layer_spec.base_gate]

    return [gate_func(indices) for indices in indices_list]


def get_heuristic_gate_generators(
    n_elec: int, layer_specs: Sequence[lspec.LayerSpec]
) -> List[of.FermionOperator]:
    """Get gate generators for the heuristic ansatz.

    Args:
        n_elec: The number of electrons.
        layer_specs: The layer specifications.

    Returns:
        A list of generators for the layers.
    """
    gate_generators = []

    for layer_spec in layer_specs:
        gate_generators += get_layer_generators(layer_spec, n_elec)

    return gate_generators
