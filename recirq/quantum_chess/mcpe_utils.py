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
"""Utilities related to the maximum consecutive positive effect (mcpe) heuristic
cost function.

These are necessary for implementing the swap-based update algorithm described
in the paper 'A Dynamic Look-Ahead Heuristic for the Qubit Mapping Problem of
NISQ Computers'
(https://ieeexplore.ieee.org/abstract/document/8976109).
"""
from collections import defaultdict, deque
from typing import Callable, Dict, Generator, Iterable, List, Optional, Set, Tuple

import cirq


def manhattan_dist(q1: cirq.GridQubit, q2: cirq.GridQubit) -> int:
    """Returns the Manhattan distance between two GridQubits.

    On grid devices this is the shortest path length between the two qubits.
    """
    return abs(q1.row - q2.row) + abs(q1.col - q2.col)


def swap_map_fn(q1: cirq.Qid, q2: cirq.Qid) -> Callable[[cirq.Qid], cirq.Qid]:
    """Returns a function which applies the effect of swapping two qubits."""
    swaps = {q1: q2, q2: q1}
    return lambda q: swaps.get(q, q)


def effect_of_swap(swap_qubits: Tuple[cirq.GridQubit, cirq.GridQubit],
                   gate_qubits: Tuple[cirq.GridQubit, cirq.GridQubit]) -> int:
    """Returns the net effect of a swap on the distance between a gate's qubits.

    Note that this returns >0 if the distance would decrease and <0 if it would
    increase, which is somewhat counter-intuitive.

    Args:
      swap_qubits: the pair of qubits to swap
      gate_qubits: the pair of qubits that the gate operates on
    """
    gate_after = map(swap_map_fn(*swap_qubits), gate_qubits)
    # TODO(https://github.com/quantumlib/ReCirq/issues/149):
    # Using manhattan distance only works for grid devices when all qubits
    # are usable (no holes, hanging strips, or disconnected qubits).
    # Update this to use the shortest path length computed on the device's
    # connectivity graph (ex using the output of Floyd-Warshall).
    return manhattan_dist(*gate_qubits) - manhattan_dist(*gate_after)


class QubitMapping:
    """Data structure representing a 1:1 map between logical and physical GridQubits.

    Args:
      initial_mapping: initial logical-to-physical qubit map.
    """
    def __init__(self, initial_mapping: Dict[cirq.Qid, cirq.GridQubit] = {}):
        self.logical_to_physical = initial_mapping
        self.physical_to_logical = {v: k for k, v in initial_mapping.items()}

    def swap_physical(self, q1: cirq.GridQubit, q2: cirq.GridQubit) -> None:
        """Updates the mapping by swapping two physical qubits."""
        logical_q1 = self.physical_to_logical.get(q1)
        logical_q2 = self.physical_to_logical.get(q2)
        self.physical_to_logical[q1], self.physical_to_logical[
            q2] = logical_q2, logical_q1
        self.logical_to_physical[logical_q1], self.logical_to_physical[
            logical_q2] = q2, q1

    def logical(self, qubit: cirq.GridQubit) -> cirq.Qid:
        """Returns the logical qubit for a given physical qubit."""
        return self.physical_to_logical.get(qubit)

    def physical(self, qubit: cirq.Qid) -> cirq.GridQubit:
        """Returns the physical qubit for a given logical qubit."""
        return self.logical_to_physical.get(qubit)


class DependencyLists:
    """Data structure representing the interdependencies between qubits and
    gates in a circuit.

    The DependencyLists maps qubits to linked lists of gates that depend on that
    qubit in execution order.
    Additionally, the DependencyLists can compute the MCPE heuristic cost
    function for candidate qubit swaps.
    """
    def __init__(self, circuit: cirq.Circuit):
        self.dependencies = defaultdict(deque)
        for moment in circuit:
            for operation in moment:
                for qubit in operation.qubits:
                    self.dependencies[qubit].append(operation)

    def peek_front(self, qubit: cirq.Qid) -> Iterable[cirq.Operation]:
        """Returns the first gate in a qubit's dependency list."""
        return self.dependencies[qubit][0]

    def pop_front(self, qubit: cirq.Qid) -> None:
        """Removes the first gate in a qubit's dependency list."""
        self.dependencies[qubit].popleft()

    def empty(self, qubit: cirq.Qid) -> bool:
        """Returns true iff the qubit's dependency list is empty."""
        return qubit not in self.dependencies or not self.dependencies[qubit]

    def all_empty(self) -> bool:
        """Returns true iff all dependency lists are empty."""
        return all(len(dlist) == 0 for dlist in self.dependencies.values())

    def active_gates(self) -> Set[cirq.Operation]:
        """Returns the currently active gates of the circuit represented by the
        dependency lists.

        The active gates are the ones which operate on qubits that have no other
        preceding gate operations.
        """
        ret = set()
        # Recomputing the active gates from scratch is less efficient than
        # maintaining them as the dependency lists are updated (e.g. maintaining
        # additional state for front gates, active gates, and frozen qubits).
        # This can be optimized later if necessary.
        for dlist in self.dependencies.values():
            if not dlist:
                continue
            gate = dlist[0]
            if gate in ret:
                continue
            if all(not self.empty(q) and self.peek_front(q) == gate
                   for q in gate.qubits):
                ret.add(gate)
        return ret

    def _maximum_consecutive_positive_effect_impl(
            self, swap_q1: cirq.GridQubit, swap_q2: cirq.GridQubit,
            gates: Iterable[cirq.Operation], mapping: QubitMapping) -> int:
        """Computes the MCPE contribution from a single qubit's dependency list.

        This is where the dynamic look-ahead window is applied -- the window of
        gates that contribute to the MCPE ends after the first gate encountered
        which would be made worse by applying the swap (the first one with
        effect_of_swap() < 0).

        Args:
          swap_q1: the source qubit to swap
          swap_q2: the target qubit to swap
          gates: the dependency list of gate operations on logical qubits
          mapping: the mapping between logical and physical qubits for gates
        """
        total_cost = 0
        for gate in gates:
            if len(gate.qubits) > 2:
                raise ValueError(
                    "Cannot compute maximum consecutive positive effect on gates with >2 qubits."
                )
            if len(gate.qubits) != 2:
                # Single-qubit gates would not be affected by the swap. We can
                # treat the change in cost as 0 for those.
                continue
            swap_cost = effect_of_swap((swap_q1, swap_q2),
                                       tuple(map(mapping.physical,
                                                 gate.qubits)))
            if swap_cost < 0:
                break
            total_cost += swap_cost
        return total_cost

    def maximum_consecutive_positive_effect(self, swap_q1: cirq.GridQubit,
                                            swap_q2: cirq.GridQubit,
                                            mapping: QubitMapping) -> int:
        """Computes the MCPE heuristic cost function of applying the swap to the
        circuit represented by this set of DependencyLists.

        Args:
          swap_q1: the source qubit to swap
          swap_q2: the target qubit to swap
          mapping: the mapping between logical and physical qubits for gate in the dependency lists
        """
        return sum(
            self._maximum_consecutive_positive_effect_impl(
                swap_q1, swap_q2, self.dependencies[mapping.logical(q)],
                mapping) for q in (swap_q1, swap_q2))
