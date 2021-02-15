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
import math
from collections import defaultdict, deque
from typing import (
    Deque,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    ValuesView,
)

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


class DynamicLookAheadHeuristicCircuitTransformer(CircuitTransformer):
    """Optimizer that transforms a circuit to satify a device's constraints.

    This implements the initial mapping algorithm and the SWAP update algorithm
    proposed by the paper "A Dynamic Look-Ahead Heuristic for the Qubit Mapping
    Problem of NISQ Computer:
    https://ieeexplore.ieee.org/abstract/document/8976109.

    Reference:
    P. Zhu, Z. Guan and X. Cheng, "A Dynamic Look-Ahead Heuristic for the
    Qubit Mapping Problem of NISQ Computers," in IEEE Transactions on Computer-
    Aided Design of Integrated Circuits and Systems, vol. 39, no. 12, pp. 4721-
    4735, Dec. 2020, doi: 10.1109/TCAD.2020.2970594.
    """
    def __init__(self, device: cirq.Device):
        super().__init__()
        self.device = device

    def build_physical_qubits_graph(
        self,
    ) -> Dict[cirq.GridQubit, List[cirq.GridQubit]]:
        """Returns an adjacency graph of physical qubits of the device.

        Each edge is bidirectional, and represents a valid two-qubit gate.
        """
        g = defaultdict(list)
        for q in self.device.qubit_set():
            neighbors = [n for n in q.neighbors() if n in self.device.qubit_set()]
            for n in neighbors:
                g[q].append(n)
        return g

    def get_least_connected_qubit(
        self,
        g: Dict[cirq.Qid, List[Tuple[cirq.Qid, int]]],
        component: Deque[cirq.Qid],
    ) -> cirq.Qid:
        """Returns the least connected qubit.

        Args:
          g: A logical qubits graph.
          component: A deque of qubits belonging to the same component.
        """
        return min(component, key=lambda q: len(g[q]))

    def build_logical_qubits_graph(
        self,
        circuit: cirq.Circuit,
    ) -> Dict[cirq.Qid, List[Tuple[cirq.Qid, int]]]:
        """Returns an adjacency graph of logical qubits of the circuit.

        Uses the heuristic of adding an edge between the nodes of each
        disjoint component that are least connected if the graph contains more
        than one connected component.

        Each edge is a tuple containing an adjacent node and the index of the
        moment at which the operation occurs.

        Arg:
          circuit: The circuit from which to build a logical qubits graph.
        """
        g = defaultdict(list)
        moment_index = 0

        # Build an adjacency graph based on the circuit.
        for i, m in enumerate(circuit):
            moment_index = i
            for op in m:
                if len(op.qubits) == 1:
                    q = op.qubits[0]
                    if q not in g:
                        g[q] = []
                if len(op.qubits) == 2:
                    q1, q2 = op.qubits
                    q1_neighbors = [n[0] for n in g[q1]]
                    if q2 not in q1_neighbors:
                        g[q1].append((q2, i))
                    q2_neighbors = [n[0] for n in g[q2]]
                    if q1 not in q2_neighbors:
                        g[q2].append((q1, i))

        # Find the connected components in the graph.
        components = deque()
        visited = set()
        for q in g:
            if q not in visited:
                components.append(self.traverse(g, q, visited))

        if len(components) == 1:
            return g

        # Connect disjoint components by adding an edge between the nodes of
        # each disjoint component that are least connected.
        while len(components) > 1:
            moment_index += 1
            first_comp = components.pop()
            first_q = self.get_least_connected_qubit(g, first_comp)
            second_comp = components.pop()
            second_q = self.get_least_connected_qubit(g, second_comp)

            # Add an edge between the two least connected nodes.
            g[first_q].append((second_q, moment_index))
            g[second_q].append((first_q, moment_index))

            # Combine the two components and add it back to the components
            # deque to continue connecting disjoint components.
            first_comp += second_comp
            components.append(first_comp)

        return g

    def find_graph_center(
        self,
        g: Union[
            Dict[cirq.GridQubit, List[cirq.GridQubit]],
            Dict[cirq.Qid, List[Tuple[cirq.Qid, int]]],
        ],
    ) -> Union[cirq.GridQubit, cirq.Qid]:
        """Returns a qubit that is a graph center.

        Uses the Floyd-Warshall algorithm to calculate the length of the
        shortest path between each pair of nodes. Then, finds the graph center
        such that the length of the shortest path to the farthest node is the
        smallest. Returns the first graph center if there are multiple.

        Args:
          g: A physical qubits graph or a logical qubits graph.
        """
        qubit_to_index_mapping = defaultdict()
        index_to_qubit_mapping = defaultdict()
        for i, q in enumerate(g):
            qubit_to_index_mapping[q] = i
            index_to_qubit_mapping[i] = q

        v = len(g)

        # Use the Floyd–Warshall algorithm to calculate the length of the
        # shortest path between each pair of nodes.
        shortest = [[math.inf for j in range(v)] for i in range(v)]
        for q in g:
            i = qubit_to_index_mapping[q]
            shortest[i][i] = 0
            for neighbor in g[q]:
                if isinstance(neighbor, tuple):
                    neighbor = neighbor[0]
                j = qubit_to_index_mapping[neighbor]
                shortest[i][j] = 1
        for k in range(v):
            for i in range(v):
                for j in range(v):
                    shortest[i][j] = min(shortest[i][j], shortest[i][k] + shortest[k][j])

        # For each node, find the length of the shortest path to the farthest
        # node
        farthest = [0 for i in range(v)]
        for i in range(v):
            for j in range(v):
                if i != j and shortest[i][j] > farthest[i]:
                    farthest[i] = shortest[i][j]

        # Find the graph center such that the length of the shortest path to the
        # farthest node is the smallest. Use the first graph center if there are
        # multiple graph centers.
        center = 0
        for i in range(v):
            if farthest[i] < farthest[center]:
                center = i

        return index_to_qubit_mapping[center]

    def traverse(
        self,
        g: Dict[cirq.Qid, List[Tuple[cirq.Qid, int]]],
        s: cirq.Qid,
        visited: Optional[set] = None,
    ) -> Deque[cirq.Qid]:
        """Returns a deque of qubits ordered by breadth-first search traversal.

        During each iteration of breadth-first search, the adjacent nodes are
        sorted by their corresponding moments before being traversed.

        Args:
          g: A logical qubits graph.
          s: The source qubit from which to start breadth-first search.
        """
        order = deque()
        if visited is None:
            visited = set()
        visited.add(s)
        queue = deque()
        queue.append(s)
        while queue:
            q = queue.popleft()
            order.append(q)
            neighbors_sorted_by_moment = sorted(g[q], key=lambda x: x[1])
            for neighbor in neighbors_sorted_by_moment:
                neighbor = neighbor[0]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return order

    def find_reference_qubits(
        self,
        mapping: Dict[cirq.Qid, cirq.GridQubit],
        lg: Dict[cirq.Qid, List[Tuple[cirq.Qid, int]]],
        lq: cirq.Qid,
    ) -> List[cirq.GridQubit]:
        """Returns a list of physical qubits from which to find the next mapping.

        The nodes adjacent to the logical qubit parameter are sorted by their
        corresponding moments before being traversed. For each adjacent node
        that has been mapped to a physical qubit, the mapped physical qubit is
        added to the result.

        Args:
          mapping: The current mapping of logical qubits to physical qubits.
          lg: A logical qubits graph.
          lq: The logical qubit from which to find reference qubits.
        """
        qubits = []
        neighbors_sorted_by_moment = sorted(lg[lq], key=lambda x: x[1])
        for neighbor in neighbors_sorted_by_moment:
            neighbor = neighbor[0]
            if neighbor in mapping:
                # This neighbor has been mapped to a physical qubit. Add the
                # physical qubit to reference qubits.
                qubits.append(mapping[neighbor])
        return qubits

    def find_candidate_qubits(
        self,
        mapped: ValuesView[cirq.GridQubit],
        pg: Dict[cirq.GridQubit, List[cirq.GridQubit]],
        pq: cirq.GridQubit,
    ) -> List[cirq.GridQubit]:
        """Returns a list of physical qubits available to be mapped.

        Uses level order traversal until a level with free adjacent node(s) is
        found.

        Args:
          mapped: The set of currently mapped physical qubits.
          lg: A physical qubits graph.
          lq: The physical qubit from which to find candidate qubits.
        """
        qubits = []
        visited = set()
        visited.add(pq)
        queue = deque()
        queue.append(pq)
        while queue:
            level = len(queue)
            while level > 0:
                q = queue.popleft()
                for neighbor in pg[q]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                    if neighbor not in mapped and neighbor not in qubits:
                        qubits.append(neighbor)
                level -= 1
            if len(qubits) > 0:
                break
        return qubits

    def find_shortest_path_distance(
        self,
        g: Dict[cirq.GridQubit, List[cirq.GridQubit]],
        s: cirq.GridQubit,
        t: cirq.GridQubit,
    ) -> int:
        """Returns the shortest distance between the source and target qubits.

        Uses breadth-first search traversal.

        Args:
          g: A physical qubits graph.
          s: The source qubit from which to start breadth-first search.
          t: The target qubit to search.
        """
        dist = defaultdict(int)
        visited = set()
        visited.add(s)
        queue = deque()
        queue.append(s)
        while queue:
            q = queue.popleft()
            if q == t:
                return dist[t]
            for neighbor in g[q]:
                if neighbor not in visited:
                    dist[neighbor] = dist[q] + 1
                    visited.add(neighbor)
                    queue.append(neighbor)
        return math.inf

    def calculate_initial_mapping(
        self,
        pg: Dict[cirq.Qid, List[cirq.Qid]],
        lg: Dict[cirq.Qid, List[Tuple[cirq.Qid, int]]],
    ) -> Dict[cirq.Qid, cirq.GridQubit]:
        """Returns an initial mapping of logical qubits to physical qubits.

        This initial mapping algorithm is proposed by the paper "A Dynamic
        Look-Ahead Heuristic for the Qubit Mapping Problem of NISQ Computer:
        https://ieeexplore.ieee.org/abstract/document/8976109.

        Args:
          pg: A physical qubits graph.
          lg: A logical qubits graph.
        """
        mapping = defaultdict()

        pg_center = self.find_graph_center(pg)
        lg_center = self.find_graph_center(lg)
        mapping[lg_center] = pg_center

        traversal_order = self.traverse(lg, lg_center)

        while traversal_order:
            lq = traversal_order.popleft()
            if lq == lg_center:
                continue
            pq = None
            reference_qubits = self.find_reference_qubits(mapping, lg, lq)
            ref_q = reference_qubits[0]
            candidate_qubits = self.find_candidate_qubits(mapping.values(), pg, ref_q)
            if len(reference_qubits) > 1:
                # For each reference location, find the shortest path distance
                # to each of the candidate qubits. Only keep the nearest
                # candidate qubits with smallest distance.
                for ref_q in reference_qubits[1:]:
                    distances = defaultdict(list)
                    for cand_q in candidate_qubits:
                        d = self.find_shortest_path_distance(pg, ref_q, cand_q)
                        distances[d].append(cand_q)
                    nearest_candidate_qubits = None
                    min_dist = math.inf
                    for dist in distances:
                        if dist < min_dist:
                            min_dist = dist
                            nearest_candidate_qubits = distances[dist]
                    candidate_qubits = nearest_candidate_qubits
                    if len(candidate_qubits) == 1:
                        break
            if len(candidate_qubits) == 1:
                pq = candidate_qubits[0]
            # If there are still more than one candidate qubit at this point,
            # choose the one with the closest degree to the logical qubit.
            if len(candidate_qubits) > 1:
                lq_degree = len(lg[lq])
                min_diff = math.inf
                for cand_q in candidate_qubits:
                    cand_q_degree = len(pg[cand_q])
                    diff = abs(cand_q_degree - lq_degree)
                    if diff < min_diff:
                        min_diff = diff
                        pq = cand_q
            mapping[lq] = pq

        return mapping

    def transform(self, circuit: cirq.Circuit) -> cirq.Circuit:
        """Returns a transformed circuit.

        The transformed circuit satisfies all physical adjacency constraints
        of the device.

        Args:
          circuit: The circuit to transform.
        """
        pg = self.build_physical_qubits_graph()
        lg = self.build_logical_qubits_graph(circuit)
        initial_mapping = self.calculate_initial_mapping(pg, lg)
        # TODO: change this to use the result of the SWAP update algorithm.
        return circuit.transform_qubits(lambda q: initial_mapping[q])

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
