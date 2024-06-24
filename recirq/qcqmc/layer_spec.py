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

from typing import Iterator, List, Tuple

import attrs


@attrs.frozen
class LayerSpec:
    """A specification of a hardware-efficient layer of gates.

    Args:
        base_gate: 'charge_charge' for the e^{i n_i n_j} gate and 'givens' for a givens rotation.
        layout: Should be 'in_pair', 'cross_pair', or 'cross_spin' only.
    """

    base_gate: str
    layout: str

    def __attrs_post_init__(self):
        if self.base_gate not in ["charge_charge", "givens"]:
            raise ValueError(
                f'base_gate is set to {self.base_gate}, it should be either "charge_charge or "givens".'
            )
        if self.layout not in ["in_pair", "cross_pair", "cross_spin"]:
            raise ValueError(
                f'layout is set to {self.layout}, it should be either "cross_pair", "in_pair", or "cross_spin".'
            )

    @classmethod
    def _json_namespace_(cls):
        return "recirq.qcqmc"

    def _json_dict_(self):
        return attrs.asdict(self)


def get_indices_heuristic_layer_in_pair(n_elec: int) -> Iterator[Tuple[int, int]]:
    """Get the indicies for the heuristic layers.

    Args:
        n_elec: The number of electrons

    Returns:
        An iterator of the indices
    """
    n_pairs = n_elec // 2

    for pair in range(n_pairs):
        to_a = n_elec + 2 * pair
        to_b = n_elec + 2 * pair + 1
        from_a = n_elec - 2 * pair - 2
        from_b = n_elec - 2 * pair - 1
        yield (from_a, to_a)
        yield (from_b, to_b)


def get_indices_heuristic_layer_cross_pair(n_elec) -> Iterator[Tuple[int, int]]:
    """Indices that couple adjacent pairs.

    Args:
        n_elec: The number of electrons

    Returns:
        An iterator of the indices
    """
    n_pairs = n_elec // 2

    for pair in range(n_pairs - 1):
        to_a = n_elec + 2 * pair
        to_b = n_elec + 2 * pair + 1
        from_next_a = n_elec - 2 * (pair + 1) - 2
        from_next_b = n_elec - 2 * (pair + 1) - 1
        yield (to_a, from_next_a)
        yield (to_b, from_next_b)


def get_indices_heuristic_layer_cross_spin(n_elec) -> Iterator[Tuple[int, int]]:
    """Get indices that couple the two spin sectors.

    Args:
        n_elec: The number of electrons

    Returns:
        An iterator of the indices that couple spin sectors.
    """
    n_pairs = n_elec // 2

    for pair in range(n_pairs):
        to_a = n_elec + 2 * pair
        to_b = n_elec + 2 * pair + 1
        from_a = n_elec - 2 * pair - 2
        from_b = n_elec - 2 * pair - 1
        yield (to_a, to_b)
        yield (from_a, from_b)


def get_layer_indices(layer_spec: LayerSpec, n_elec: int) -> List[Tuple[int, int]]:
    """Get the indices for the heuristic layers.

    Args:
        layer_spec: The layer specification.
        n_elec: The number of electrons.

    Returns:
        A list of indices for the layer.
    """
    indices_generators = {
        "in_pair": get_indices_heuristic_layer_in_pair(n_elec),
        "cross_pair": get_indices_heuristic_layer_cross_pair(n_elec),
        "cross_spin": get_indices_heuristic_layer_cross_spin(n_elec),
    }
    indices_generator = indices_generators[layer_spec.layout]

    return [indices for indices in indices_generator]
