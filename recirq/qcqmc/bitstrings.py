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

import itertools
from typing import Iterable, Tuple


def get_bitstrings_a_b(*, n_orb: int, n_elec: int) -> Iterable[Tuple[bool, ...]]:
    """Iterates over bitstrings with the right symmetry assuming a_b ordering.

    This function assumes that the first n_orb qubits correspond to the alpha
    orbitals and the second n_orb qubits correspond to the beta orbitals. The
    ordering within the alpha and beta sectors doesn't matter (because we
    iterate over all bitstrings with Hamming weight n_elec//2 in each sector.
    """

    if n_orb != n_elec:
        raise NotImplementedError("n_orb must equal n_elec.")

    initial_bitstring = tuple(False for _ in range(n_orb - n_elec // 2)) + tuple(
        True for _ in range(n_elec // 2)
    )

    spin_sector_bitstrings = set()
    for perm in itertools.permutations(initial_bitstring):
        spin_sector_bitstrings.add(perm)

    for bitstring_a, bitstring_b in itertools.product(spin_sector_bitstrings, repeat=2):
        yield bitstring_a + bitstring_b
