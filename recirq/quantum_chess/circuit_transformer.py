# Copyright 2020 Google
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
import copy
from typing import Dict, Iterable, List, Optional, Set

import cirq

import recirq.quantum_chess.controlled_iswap as controlled_iswap

ADJACENCY = [(0, 1), (0, -1), (1, 0), (-1, 0)]


class CircuitTransformer:
    """Abstract interface for circuit transformations.

    For example: NamedQubit -> GridQubit transformations.
    """
    def __init__(self):
        pass

    def transform(self, circuit: cirq.Circuit) -> cirq.Circuit:
        """Applies the transformation to the circuit."""
        return None


class ConnectivityHeuristicCircuitTransformer(CircuitTransformer):
    """Optimizer that will transform a circuit using NamedQubits
    and transform it to use GridQubits.  This will use a breadth-first
    search to find a suitable mapping into the specified grid.

    It will then transform all operations to use the new qubits.
    """
    def __init__(self, device: cirq.Device):
        super().__init__()
        self.device = device
        self.mapping = None
        self.starting_qubit = self.find_start_qubit(device.qubits)
        self.qubit_list = device.qubits

    def qubits_within(self, depth: int, qubit: cirq.GridQubit,
                      qubit_list: Iterable[cirq.GridQubit]) -> int:
        """Returns the number of qubits within `depth` of the input `qubit`.

        Args:
          depth: how many connections to traverse
          qubit: starting qubit
          qubit_list: iterable of valid qubits
        """
        if qubit not in qubit_list:
            return 0
        if depth <= 0:
            return 1
        c = 1
        for diff in ADJACENCY:
            c += self.qubits_within(depth - 1, qubit + diff, qubit_list)
        return c

    def find_start_qubit(self,
                         qubit_list: List[cirq.Qid],
                         depth=3) -> Optional[cirq.GridQubit]:
        """Finds a reasonable starting qubit to start the mapping.

        Uses the heuristic of the most connected qubit. """
        best = None
        best_count = -1
        for q in qubit_list:
            c = self.qubits_within(depth, q, qubit_list)
            if c > best_count:
                best_count = c
                best = q
        return best

    def edges_within(self, depth: int, node: cirq.Qid,
                     graph: Dict[cirq.Qid, Iterable[cirq.Qid]]) -> int:
        """Returns the number of qubits within `depth` of the specified `node`.

        Args:
          depth: how many connections to traverse
          node: starting qubit
          graph: edge graph of connections between qubits,
            representing by a dictionary from qubit to adjacent qubits.
        """
        if depth <= 0:
            return 1
        c = 1
        for adj_node in graph[node]:
            c += self.edges_within(depth - 1, adj_node, graph)
        return c

    def find_start_node(self, graph: Dict[cirq.Qid, Iterable[cirq.Qid]],
                        mapping: Dict[cirq.Qid, cirq.GridQubit]) -> cirq.Qid:
        """Finds a reasonable starting qubit from an adjacency graph.

        Args:
            graph: edge graph of connections between qubits,
            representing by a dictionary from qubit to adjacent qubits.
        """
        best = None
        best_count = -1
        for node in graph:
            if node in mapping:
                continue
            c = self.edges_within(3, node, graph)
            if c > best_count:
                best_count = c
                best = node
        return best

    def map_helper(self,
                   cur_node: cirq.Qid,
                   mapping: Dict[cirq.Qid, cirq.GridQubit],
                   available_qubits: Set[cirq.GridQubit],
                   graph: Dict[cirq.Qid, Iterable[cirq.Qid]],
                   print_debug: bool = False) -> bool:
        """Helper function to construct mapping.

        Traverses a graph and performs recursive depth-first
        search to construct a mapping one node at a time.
        On failure, raises an error and back-tracks until a
        suitable mapping can be found.  Assumes all qubits in
        the graph are connected.

        Args:
          cur_node: node to examine.
          mapping: current mapping of named qubits to `GridQubits`
          available_qubits: current set of unassigned qubits
          graph: adjacency graph of connections between qubits,
            representing by a dictionary from qubit to adjacent qubits

        Returns:
          True if mapping was successful, False if no mapping was possible.
        """
        # cur_node is the named qubit
        # cur_qubit is the currently assigned GridQubit
        cur_qubit = mapping[cur_node]
        if print_debug:
            print(f'{cur_node} -> {cur_qubit}')

        # Determine the list of adjacent nodes that still need to be mapped
        nodes_to_map = []
        for node in graph[cur_node]:
            if node not in mapping:
                # Unmapped node.
                nodes_to_map.append(node)
            else:
                # Mapped adjacent node.
                # Verify that the previous mapping is adjacent in the Grid.
                if not mapping[node].is_adjacent(cur_qubit):
                    if print_debug:
                        print(f'Not adjacent {node} and {cur_node}')
                    return False
        if not nodes_to_map:
            # All done with this node.
            return True

        # Find qubits that are adjacent in the grid
        valid_adjacent_qubits = []
        for a in ADJACENCY:
            q = cur_qubit + a
            if q in available_qubits:
                valid_adjacent_qubits.append(q)

        # Not enough adjacent qubits to map all qubits
        if len(valid_adjacent_qubits) < len(nodes_to_map):
            if print_debug:
                print(f'Cannot fit adjacent nodes into {cur_node}')
            return False

        # Only map one qubit at a time
        # This makes back-tracking easier.
        node_to_map = nodes_to_map[0]

        for node_to_try in valid_adjacent_qubits:
            # Add proposed qubit to the mapping
            # and remove it from available qubits
            mapping[node_to_map] = node_to_try
            available_qubits.remove(node_to_try)

            # Recurse
            # Move on to the qubit we just mapped.
            # Then, come back to this node and
            # map the rest of the adjacent nodes

            success = self.map_helper(node_to_map, mapping, available_qubits,
                                      graph)
            if success:
                success = self.map_helper(cur_node, mapping, available_qubits,
                                          graph)

            if not success:
                # We have failed.  Undo this mapping and try another one.
                del mapping[node_to_map]
                available_qubits.add(node_to_try)
            else:
                # We have successfully mapped all qubits!
                return True

        # All available qubits were not valid.
        # Fail upwards and back-track if possible.
        if print_debug:
            print('Warning: could not map all qubits!')
            return False

    def qubit_mapping(self,
                      circuit: cirq.Circuit) -> Dict[cirq.Qid, cirq.GridQubit]:
        """Create a mapping from NamedQubits to Grid Qubits

       This function analyzes the circuit to determine which
       qubits need to be adjacent, then maps to the grid of the device
       based on the generated mapping.
       """
        # Build up an adjacency graph based on the circuits.
        # Two qubit gates will turn into edges in the graph
        g = {}
        # Keep track of single qubits that don't interact.
        sq = []
        for m in circuit:
            for op in m:
                if len(op.qubits) == 1:
                    if op.qubits[0] not in g:
                        sq.append(op.qubits[0])
                if len(op.qubits) == 2:
                    q1, q2 = op.qubits
                    if q1 not in g:
                        g[q1] = []
                    if q2 not in g:
                        g[q2] = []
                    if q1 in sq:
                        sq.remove(q1)
                    if q2 in sq:
                        sq.remove(q2)
                    if q2 not in g[q1]:
                        g[q1].append(q2)
                    if q1 not in g[q2]:
                        g[q2].append(q1)
        for q in g:
            if len(g[q]) > 4:
                raise ValueError(
                    f'Qubit {q} needs more than 4 adjacent qubits!')

        # Initialize mappings and available qubits
        start_qubit = self.starting_qubit
        mapping = {}
        start_list = set(copy.copy(self.qubit_list))
        start_list.remove(start_qubit)

        last_node = None
        cur_node = self.find_start_node(g, mapping)
        if not cur_node and sq:
            for q in sq:
                start_qubit = self.find_start_qubit(start_list)
                mapping[q] = start_qubit
                start_list.remove(start_qubit)
            self.mapping = mapping
            return mapping

        mapping[cur_node] = start_qubit

        while last_node != cur_node:
            # Depth first seach for a valid mapping
            self.map_helper(cur_node, mapping, start_list, g)
            last_node = cur_node
            cur_node = self.find_start_node(g, mapping)
            if not cur_node:
                for q in sq:
                    start_qubit = self.find_start_qubit(start_list)
                    mapping[q] = start_qubit
                    start_list.remove(start_qubit)
                break
            # assign a new start qubit
            start_qubit = self.find_start_qubit(start_list)
            mapping[cur_node] = start_qubit
            start_list.remove(start_qubit)

        if len(mapping) != len(g):
            print('Warning: could not map all qubits!')
        # Sanity check, ensure qubits not mapped twice
        assert len(mapping) == len(set(mapping.values()))

        self.mapping = mapping
        return mapping

    def transform(self, circuit: cirq.Circuit) -> cirq.Circuit:
        """ Creates a new qubit mapping for a circuit and transforms it.

        This uses `qubit_mapping` to create a mapping from the qubits
        in the circuit to the qubits on the device, then substitutes
        in those qubits in the circuit and returns the new circuit.

        Returns:
          The circuit with the mapped qubits.
        """
        self.qubit_mapping(circuit)
        return circuit.transform_qubits(lambda q: self.mapping[q])


class SycamoreDecomposer(cirq.PointOptimizer):
    """Optimizer that decomposes all three qubit operations into
    sqrt-ISWAPs.

    Currently supported are controlled ISWAPs with a single control
    and control-X gates with multiple controls (TOFFOLI gates).:w
    """
    def optimization_at(
            self, circuit: cirq.Circuit, index: int,
            op: cirq.Operation) -> Optional[cirq.PointOptimizationSummary]:
        if len(op.qubits) > 3:
            raise ValueError(f'Four qubit ops not yet supported: {op}')
        new_ops = None
        if op.gate == cirq.SWAP or op.gate == cirq.CNOT:
            new_ops = cirq.google.optimized_for_sycamore(cirq.Circuit(op))
        if isinstance(op, cirq.ControlledOperation):
            qubits = op.sub_operation.qubits
            if op.gate.sub_gate == cirq.ISWAP:
                new_ops = controlled_iswap.controlled_iswap(
                    *qubits, *op.controls)
            if op.gate.sub_gate == cirq.ISWAP ** -1:
                new_ops = controlled_iswap.controlled_iswap(*qubits,
                                                            *op.controls,
                                                            inverse=True)
            if op.gate.sub_gate == cirq.ISWAP ** 0.5:
                new_ops = controlled_iswap.controlled_sqrt_iswap(
                    *qubits, *op.controls)
            if op.gate.sub_gate == cirq.ISWAP ** -0.5:
                new_ops = controlled_iswap.controlled_inv_sqrt_iswap(
                    *qubits, *op.controls)
            if op.gate.sub_gate == cirq.X:
                if len(op.qubits) == 2:
                    new_ops = cirq.google.optimized_for_sycamore(
                        cirq.Circuit(cirq.CNOT(*op.controls, *qubits)))
                if len(op.qubits) == 3:
                    new_ops = cirq.google.optimized_for_sycamore(
                        cirq.Circuit(cirq.TOFFOLI(*op.controls, *qubits)))
        if new_ops:
            return cirq.PointOptimizationSummary(clear_span=1,
                                                 clear_qubits=op.qubits,
                                                 new_operations=new_ops)
