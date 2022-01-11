# Copyright 2021 Google
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
"""Implementation of the swap update algorithm described in the paper 'A Dynamic
Look-Ahead Heuristic for the Qubit Mapping Problem of NISQ Computers'
(https://ieeexplore.ieee.org/abstract/document/8976109).

This transforms circuits by adding additional SWAP gates to ensure that all operations are on adjacent qubits.
"""
from collections import deque
from typing import Callable, Dict, Generator, Iterable, List, Optional, Tuple

import cirq

import recirq.quantum_chess.mcpe_utils as mcpe
from cirq_google.optimizers.convert_to_sqrt_iswap import swap_to_sqrt_iswap


def _satisfies_adjacency(gate: cirq.Operation) -> bool:
    """Returns true iff the input gate operates on adjacent qubits.

    Requires the input to be either a binary or unary operation on GridQubits.
    """
    if len(gate.qubits) > 2:
        raise ValueError(
            "Cannot determine physical adjacency for gates with > 2 qubits"
        )
    if len(gate.qubits) < 2:
        return True
    q1, q2 = gate.qubits
    return q1.is_adjacent(q2)


def _pairwise_shortest_distances(
    adjacencies: Dict[cirq.GridQubit, List[cirq.GridQubit]]
) -> Dict[Tuple[cirq.GridQubit, cirq.GridQubit], int]:
    """Precomputes the shortest path length between each pair of qubits.

    This function runs in O(V**2) where V=len(qubits).

    Args:
        adjacencies: adjacency list representation of the qubit graph
    Returns:
        dictionary mapping a pair of qubits to the length of the shortest path
        between them (disconnected qubits will be absent).
    """
    # Do BFS starting from each qubit and collect the shortest path lengths as
    # we go.
    # For device graphs where all edges are the same (where each edge can be
    # treated as unit length) repeated BFS is O(V**2 + V E). On sparse device
    # graphs (like GridQubit graphs where qubits are locally connected to their
    # neighbors) that is faster than the Floyd-Warshall algorithm which is
    # O(V**3).
    # However this won't work if in the future we figure out a way to
    # incorporate edge weights (for example in order to give negative preference
    # to gates with poor calibration metrics).
    shortest = {}
    for starting_qubit in adjacencies:
        to_be_visited = deque()
        shortest[(starting_qubit, starting_qubit)] = 0
        to_be_visited.append((starting_qubit, 0))
        while to_be_visited:
            qubit, cur_dist = to_be_visited.popleft()
            for neighbor in adjacencies[qubit]:
                if (starting_qubit, neighbor) not in shortest:
                    shortest[(starting_qubit, neighbor)] = cur_dist + 1
                    to_be_visited.append((neighbor, cur_dist + 1))
    return shortest


def generate_decomposed_swap(
    q1: cirq.Qid, q2: cirq.Qid
) -> Generator[cirq.Operation, None, None]:
    """Generates a SWAP operation using sqrt-iswap gates."""
    yield from swap_to_sqrt_iswap(q1, q2, 1.0)


class SwapUpdater:
    """SwapUpdater runs the swap update algorithm in order to incrementally update a circuit with SWAPs.

    The SwapUpdater's internal state is modified as the algorithm runs, so each instance is one-time use.

    Args:
      circuit: the circuit to be updated with additional SWAPs
      device_qubits: the allowed set of qubits on the device. If None, behaves
        as though operating on an unconstrained infinite device.
      initial_mapping: the initial logical-to-physical qubit mapping, which
        must contain an entry for every qubit in the circuit
      swap_factory: the factory used to produce operations representing a swap
        of two qubits
    """

    def __init__(
        self,
        circuit: cirq.Circuit,
        device_qubits: Optional[Iterable[cirq.GridQubit]],
        initial_mapping: Dict[cirq.Qid, cirq.GridQubit] = {},
        swap_factory: Callable[
            [cirq.Qid, cirq.Qid], Iterable[cirq.Operation]
        ] = generate_decomposed_swap,
    ):
        self.device_qubits = device_qubits
        self.dlists = mcpe.DependencyLists(circuit)
        self.mapping = mcpe.QubitMapping(initial_mapping)
        self.swap_factory = swap_factory
        self.adjacent = {q: q.neighbors(device_qubits) for q in device_qubits}
        self.pairwise_distances = _pairwise_shortest_distances(self.adjacent)
        # Tracks swaps that have been made since the last circuit gate was
        # output.
        self.prev_swaps = set()

    def _distance_between(self, q1: cirq.GridQubit, q2: cirq.GridQubit) -> int:
        """Returns the precomputed length of the shortest path between two qubits."""
        return self.pairwise_distances[(q1, q2)]

    def generate_candidate_swaps(
        self, gates: Iterable[cirq.Operation]
    ) -> Generator[Tuple[cirq.GridQubit, cirq.GridQubit], None, None]:
        """Generates the candidate SWAPs that would have a positive effect on at
        least one of the given physical gates.

        Args:
          gates: the list of gates to consider which operate on GridQubits
        """
        for gate in gates:
            for gate_q in gate.qubits:
                for swap_q in gate_q.neighbors(self.device_qubits):
                    swap_qubits = (gate_q, swap_q)
                    effect = mcpe.effect_of_swap(
                        swap_qubits, gate.qubits, self._distance_between
                    )
                    if swap_qubits not in self.prev_swaps and effect > 0:
                        yield swap_qubits

    def _mcpe(self, swap_q1: cirq.GridQubit, swap_q2: cirq.GridQubit) -> int:
        """Returns the maximum consecutive positive effect of swapping two qubits."""
        return self.dlists.maximum_consecutive_positive_effect(
            swap_q1, swap_q2, self.mapping, self._distance_between
        )

    def update_iteration(self) -> Generator[cirq.Operation, None, None]:
        """Runs one iteration of the swap update algorithm and updates internal
        state about the original circuit.

        Returns:
          the operations on GridQubits in the final updated circuit generated by
          this iteration
        """
        # Handle the already-satisfied active gates.
        # Those can be immediately added into the final circuit.
        active_physical_gates = []
        for gate in set(self.dlists.active_gates):
            physical_gate = gate.transform_qubits(self.mapping.physical)
            if _satisfies_adjacency(physical_gate):
                # physical_gate is ready to be popped off the dependecy lists
                # and added to the final circuit.
                self.dlists.pop_active(gate)
                self.prev_swaps.clear()
                yield physical_gate
            else:
                # physical_gate needs to be fixed up with some swaps.
                active_physical_gates.append(physical_gate)

        # If all the active gates in this pass were already optimal + added to
        # the final circuit, then we have nothing left to do until we make
        # another pass and get the newly active gates.
        if not active_physical_gates:
            return

        candidates = set(self.generate_candidate_swaps(active_physical_gates))
        if not candidates:
            # This should never happen for reasonable initial mappings.
            # For example, it can happen when the initial mapping placed a
            # gate's qubits on disconnected components in the device
            # connectivity graph.
            raise ValueError("no swaps founds that will improve the circuit")
        chosen_swap = max(candidates, key=lambda swap: self._mcpe(*swap))
        self.prev_swaps.add(chosen_swap)
        self.prev_swaps.add(tuple(reversed(chosen_swap)))
        self.mapping.swap_physical(*chosen_swap)
        yield from self.swap_factory(*chosen_swap)

    def add_swaps(self) -> Generator[cirq.Operation, None, None]:
        """Iterates the swap update algorithm to completion.

        If the updater already completed, does nothing.

        Returns:
          the generated operations on physical GridQubits in the final circuit
        """
        while not self.dlists.all_empty():
            yield from self.update_iteration()
