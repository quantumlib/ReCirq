# Copyright 2023 Google
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

from typing import List, Optional, Tuple


def rotate_list(items: List, shift: int, clockwise: bool = True) -> List:
    """Rotates a list, clockwise or anti-clockwise

    Args:
        items (List): list to rotate
        shift (int): how much to rotate
        clockwise (bool): whether to rotate clockwise
    """
    if clockwise:
        return items[-shift:] + items[:-shift]
    else:
        return items[shift:] + items[:shift]


class SnakeMeasurementGroup:
    def __init__(self, pairs: Optional[List[Tuple]] = None, swaps: Optional[List] = None, shift=0):
        """Index container to keep track of measurement scheduling

        Args:
            pairs: a list of correlated pairs of pairs. The first pair contains
                physical indices (corresponding to those in the cirq.Circuit)
                the second list contains virtual indices (corresponding to
                those in the openfermion.QubitOperator).
            swaps (List): list of pairs of qubits to be swapped before
                measurement.
            shift (int): tells how far the virtual indices have been shifted
                from the logical indices.
        """
        self.pairs = pairs if pairs else []
        self.swaps = swaps if swaps else []
        self.shift = shift


def get_tqbg_groups(n_qubits: int) -> List[SnakeMeasurementGroup]:
    """Generates a complete list of SnakeMeasurementGroups.

    By complete, we mean that each pair of virtual indices is guaranteed
    to appear in at least one group.

    Args:
        n_qubits: number of qubits in loop
    """

    # This used to take in a list of qubits rather than the number;
    # we keep this here for backwards compatibility.
    if type(n_qubits) is not int:
        n_qubits = len(n_qubits)
    if n_qubits % 2:
        raise NotImplementedError

    msmt_groups = []

    # First we take all groupings without any swaps
    for shift in range(n_qubits // 2):
        log_inds = list(range(n_qubits - shift, n_qubits)) + list(range(n_qubits - shift))
        pairs = [
            ((j, n_qubits - 1 - j), (log_inds[j], log_inds[n_qubits - 1 - j]))
            for j in range(n_qubits // 2)
        ]
        msmt_groups.append(SnakeMeasurementGroup(pairs, shift=shift))

    # Now groupings with swaps
    for shift in range(n_qubits // 2):
        log_inds = list(range(n_qubits - shift, n_qubits)) + list(range(n_qubits - shift))

        # Depending on the parity of shift, we either start swapping from
        # qubit 0 or qubit 1. We then swap qubits in pairs till the end of
        # the top row.
        swaps = [(j, j + 1) for j in range(shift % 2, n_qubits // 2 - 1, 2)]
        # Escape the case for four qubits where we only have one option for swaps
        if len(swaps) == 0:
            continue
        swapped_qubits = [q for pair in swaps for q in pair]

        # Perform the swap on the logical indices to keep track
        for swap in swaps:
            log_inds[swap[0]], log_inds[swap[1]] = log_inds[swap[1]], log_inds[swap[0]]

        # Now make the pairs as before, but only include those where a swap
        # was performed.
        pairs = [
            ((j, n_qubits - 1 - j), (log_inds[j], log_inds[n_qubits - 1 - j]))
            for j in range(n_qubits // 2)
            if j in swapped_qubits
        ]

        msmt_groups.append(SnakeMeasurementGroup(pairs, swaps, shift))
    msmt_groups.append(SnakeMeasurementGroup())
    return msmt_groups
