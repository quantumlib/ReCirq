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
import math
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple, Union, ValuesView

import cirq


def build_physical_qubits_graph(
    device: cirq.Device,
) -> Dict[cirq.GridQubit, List[cirq.GridQubit]]:
    """Returns an adjacency graph of physical qubits of the device.

    Each edge is bidirectional, and represents a valid two-qubit gate.

    Args:
      device: The device from which to build a physical qubits graph.
    """
    g = defaultdict(list)
    for q in device.qubit_set():
        g[q] = [n for n in q.neighbors() if n in device.qubit_set()]
    return g


def get_least_connected_qubit(
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
            if isinstance(op.gate, cirq.MeasurementGate):
                # Skip measurement gates.
                continue
            if len(op.qubits) == 1:
                q = op.qubits[0]
                if q not in g:
                    g[q] = []
            elif len(op.qubits) == 2:
                q1, q2 = op.qubits
                q1_neighbors = [n[0] for n in g[q1]]
                if q2 not in q1_neighbors:
                    g[q1].append((q2, i))
                q2_neighbors = [n[0] for n in g[q2]]
                if q1 not in q2_neighbors:
                    g[q2].append((q1, i))
            else:
                raise ValueError(f"Operation {op} has more than 2 qubits!")

    # Find the connected components in the graph.
    components = deque()
    visited = set()
    for q in g:
        if q not in visited:
            components.append(traverse(g, q, visited))

    if len(components) == 1:
        return g

    # Connect disjoint components by adding an edge between the nodes of
    # each disjoint component that are least connected.
    while len(components) > 1:
        moment_index += 1
        first_comp = components.pop()
        first_q = get_least_connected_qubit(g, first_comp)
        second_comp = components.pop()
        second_q = get_least_connected_qubit(g, second_comp)

        # Add an edge between the two least connected nodes.
        g[first_q].append((second_q, moment_index))
        g[second_q].append((first_q, moment_index))

        # Combine the two components and add it back to the components
        # deque to continue connecting disjoint components.
        first_comp += second_comp
        components.append(first_comp)

    return g


def find_all_pairs_shortest_paths(
    g: Union[
        Dict[cirq.GridQubit, List[cirq.GridQubit]],
        Dict[cirq.Qid, List[Tuple[cirq.Qid, int]]],
    ],
) -> Dict[Tuple[cirq.Qid, cirq.Qid], int]:
    """Returns a dict of the shortest distance between each pair of nodes.

    Implements repeated BFS, which is faster than the Floyd–Warshall algorithm
    when the input graph is sparse.

    Args:
      g: A physical qubits graph or a logical qubits graph.
    """
    all_qubits = set()
    adjacent = defaultdict(list)
    for k, v in g.items():
        all_qubits.add(k)
        for neighbor in v:
            if isinstance(neighbor, tuple):
                neighbor = neighbor[0]
            all_qubits.add(neighbor)
            adjacent[k].append(neighbor)

    shortest = defaultdict(lambda: math.inf)
    for starting_qubit in all_qubits:
        to_be_visited = deque()
        shortest[(starting_qubit, starting_qubit)] = 0
        to_be_visited.append((starting_qubit, 0))
        while to_be_visited:
            qubit, cur_dist = to_be_visited.popleft()
            for neighbor in adjacent[qubit]:
                if (starting_qubit, neighbor) not in shortest:
                    shortest[(starting_qubit, neighbor)] = cur_dist + 1
                    to_be_visited.append((neighbor, cur_dist + 1))
    return shortest


def find_graph_center(
    g: Union[
        Dict[cirq.GridQubit, List[cirq.GridQubit]],
        Dict[cirq.Qid, List[Tuple[cirq.Qid, int]]],
    ],
) -> cirq.Qid:
    """Returns a qubit that is a graph center.

    Uses the Floyd-Warshall algorithm to calculate the length of the
    shortest path between each pair of nodes. Then, finds the graph center
    such that the length of the shortest path to the farthest node is the
    smallest. Returns the first graph center if there are multiple.

    Args:
      g: A physical qubits graph or a logical qubits graph.
    """
    shortest = find_all_pairs_shortest_paths(g)

    # For each node, find the length of the shortest path to the farthest
    # node.
    farthest = defaultdict(int)
    for i in g:
        for j in g:
            if i != j and shortest[(i, j)] > farthest[i]:
                farthest[i] = shortest[(i, j)]

    # Find the graph center such that the length of the shortest path to the
    # farthest node is the smallest. Use the first graph center if there are
    # multiple graph centers.
    center = None
    for q in g:
        if not center or farthest[q] < farthest[center]:
            center = q

    return center


def traverse(
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
        for neighbor, _ in neighbors_sorted_by_moment:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return order


def find_reference_qubits(
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
    for neighbor, _ in neighbors_sorted_by_moment:
        if neighbor in mapping:
            # This neighbor has been mapped to a physical qubit. Add the
            # physical qubit to reference qubits.
            qubits.append(mapping[neighbor])
    return qubits


def find_candidate_qubits(
    mapped: ValuesView[cirq.GridQubit],
    pg: Dict[cirq.GridQubit, List[cirq.GridQubit]],
    pq: cirq.GridQubit,
) -> List[cirq.GridQubit]:
    """Returns a list of physical qubits available to be mapped.

    Uses level order traversal until a level with free adjacent node(s) is
    found.

    Args:
      mapped: The set of currently mapped physical qubits.
      pg: A physical qubits graph.
      pq: The physical qubit from which to find candidate qubits.
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


def find_shortest_path(
    g: Dict[cirq.GridQubit, List[cirq.GridQubit]],
    s: cirq.GridQubit,
    t: cirq.GridQubit,
) -> Union[int, float]:
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
    device: cirq.Device,
    circuit: cirq.Circuit,
) -> Dict[cirq.Qid, cirq.GridQubit]:
    """Returns an initial mapping of logical qubits to physical qubits.

    This initial mapping algorithm is proposed by the paper "A Dynamic
    Look-Ahead Heuristic for the Qubit Mapping Problem of NISQ Computer:
    https://ieeexplore.ieee.org/abstract/document/8976109.

    Args:
      device: The device on which to run the circuit.
      circuit: The circuit from which to calculate an initial mapping.
    """
    mapping = defaultdict()

    pg = build_physical_qubits_graph(device)
    lg = build_logical_qubits_graph(circuit)
    pg_center = find_graph_center(pg)
    lg_center = find_graph_center(lg)
    mapping[lg_center] = pg_center

    traversal_order = traverse(lg, lg_center)

    while traversal_order:
        lq = traversal_order.popleft()
        if lq == lg_center:
            continue
        pq = None
        reference_qubits = find_reference_qubits(mapping, lg, lq)
        candidate_qubits = find_candidate_qubits(
            mapping.values(), pg, reference_qubits[0]
        )
        if len(reference_qubits) > 1:
            # For each reference location, find the shortest path distance
            # to each of the candidate qubits. Only keep the nearest
            # candidate qubits with smallest distance.
            for ref_q in reference_qubits[1:]:
                distances = defaultdict(list)
                for cand_q in candidate_qubits:
                    d = find_shortest_path(pg, ref_q, cand_q)
                    distances[d].append(cand_q)
                min_dist = min(distances.keys())
                candidate_qubits = distances[min_dist]
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
