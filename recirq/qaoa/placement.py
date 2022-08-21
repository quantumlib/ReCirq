import os
from collections import defaultdict
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

import cirq
import cirq.contrib.routing as ccr
import cirq_google as cg

try:
    # Set the 'RECIRQ_IMPORT_FAILSAFE' environment variable to treat PyTket as an optional
    # dependency. We do this for CI testing against the next, pre-release Cirq version.
    import pytket
    import pytket.extensions.cirq
    from pytket.circuit import Node, Qubit
    from pytket.passes import SequencePass, RoutingPass, PlacementPass
    from pytket.predicates import CompilationUnit, ConnectivityPredicate
    try:
        from pytket.placement import GraphPlacement
    except ImportError:
        from pytket.routing import GraphPlacement

    try:
        from pytket.architecture import Architecture
    except ImportError:
        from pytket.routing import Architecture
except ImportError as e:
    if 'RECIRQ_IMPORT_FAILSAFE' in os.environ:
        pytket = NotImplemented
    else:
        raise e

import recirq


def calibration_data_to_graph(calib_dict: cg.Calibration) -> nx.Graph:
    """Take the calibration data in dictionary form and return a graph
    representing the errors.

    The edge weights are two_qubit_sycamore_gate_xeb_cycle_total_error.
    The node weights are single_qubit_readout_p0_error
                       + single_qubit_readout_p1_error.
    """
    err_graph = nx.Graph()
    for (q1, q2), err in calib_dict['two_qubit_sycamore_gate_xeb_cycle_total_error'].items():
        err_graph.add_edge(q1, q2, weight=err[0])

    for (q,), err in calib_dict['single_qubit_readout_p0_error'].items():
        err_graph.nodes[q]['weight'] = err[0]

    for (q,), err in calib_dict['single_qubit_readout_p1_error'].items():
        err_graph.nodes[q]['weight'] += err[0]

    return err_graph


def _qubit_index_edges(device: cirq.Device):
    """Helper function in `_device_to_tket_device`"""
    qubits = device.metadata.qubit_set if device.metadata else ()
    dev_graph = ccr.gridqubits_to_graph_device(qubits)
    for n1, n2 in dev_graph.edges:
        yield Node('grid', n1.row, n1.col), Node('grid', n2.row, n2.col)


def _device_to_tket_device(device: cirq.Device):
    """Custom function to turn a device into a pytket device.

    This supports any device that supports `ccr.xmon_device_to_graph`.
    """
    return Architecture(
        list(_qubit_index_edges(device))
    )


def tk_to_cirq_qubit(tk: 'Qubit'):
    """Convert a tket Qubit to either a LineQubit or GridQubit.

    """
    ind = tk.index
    return (
        cirq.LineQubit(ind[0])
        if len(ind) == 1
        else cirq.GridQubit(*ind)
    )


def place_on_device(circuit: cirq.Circuit,
                    device: cirq.Device,
                    ) -> Tuple[cirq.Circuit,
                               Dict[cirq.Qid, cirq.Qid],
                               Dict[cirq.Qid, cirq.Qid]]:
    """Place a circuit on an device.

    Converts a circuit to a new circuit that respects the adjacency of a given
    device and is equivalent to the given circuit up to qubit ordering.

    Args:
        circuit: The circuit to place on a grid.
        device: The device to place the circuit on.

    Returns:
        routed_circuit: The new circuit
        initial_map: Initial placement of qubits
        final_map: The final placement of qubits after action of the circuit
    """
    tk_circuit = pytket.extensions.cirq.cirq_to_tk(circuit)
    tk_device = _device_to_tket_device(device)

    unit = CompilationUnit(tk_circuit, [ConnectivityPredicate(tk_device)])
    passes = SequencePass([
        PlacementPass(GraphPlacement(tk_device)),
        RoutingPass(tk_device)])
    passes.apply(unit)
    valid = unit.check_all_predicates()
    if not valid:
        raise RuntimeError("Routing failed")

    initial_map = {tk_to_cirq_qubit(n1): tk_to_cirq_qubit(n2)
                   for n1, n2 in unit.initial_map.items()}
    final_map = {tk_to_cirq_qubit(n1): tk_to_cirq_qubit(n2)
                 for n1, n2 in unit.final_map.items()}
    routed_circuit = pytket.extensions.cirq.tk_to_cirq(unit.circuit)

    return routed_circuit, initial_map, final_map


def path_weight(graph: nx.Graph, path,
                include_node_weights=True) -> float:
    """Returns total weight of edges along a path.

    Args:
        graph: a nx.Graph object with specified edge weights
        path: a list of nodes specifying a path on graph
        include_node_weights: whether include node weight in
               overall path weight for minimization (default=True)

    Returns:
        total weight of edges along the path
    """
    if path is None:
        return float('inf')
    weight = 0
    for i in range(len(path) - 1):
        weight += graph[path[i]][path[i + 1]]['weight']
    if include_node_weights and 'weight' in graph.nodes[path[0]]:
        n = len(path)
        for node in path:
            weight += graph.nodes[node]['weight'] / n * 2
    return weight


def min_weight_simple_paths_brute_force(
        graph: nx.Graph,
        weight_fun: Callable[[nx.Graph, List], float] = path_weight):
    """Find all simple paths of various lengths that has minimum total weight
    using brute-force.

    The strategy is look at all possible simple path, calculate their path
    weight, keep best one. This works reasonably quickly for sparse graph of
    size <= 25 (~10 seconds). Worst case complexity is O(n!) for complete graph

    Args:
        graph: a networkx.Graph object with specified edge weights
        weight_fun: a function that takes (graph, path) and gives a value
            based on edge and node weights that we want to minimize
            (default: uses path_weight)

    Returns:
        a dictionary in the form
        {n: path containing n nodes with min weight, or None if doesn't exist}
    """
    best_weights = defaultdict(lambda: float('inf'))
    best_paths = {}
    nodelist = list(graph.nodes())
    for i in range(len(nodelist) - 1):
        for j in range(i + 1, len(nodelist)):
            for path in nx.all_simple_paths(graph, nodelist[i], nodelist[j]):
                n = len(path)
                my_weight = weight_fun(graph, path)
                if my_weight < best_weights[n]:
                    best_paths[n] = path
                    best_weights[n] = my_weight
    return best_paths


def min_weight_simple_path_brute_force(
        graph: nx.Graph,
        n: int,
        weight_fun: Callable[[nx.Graph, List], float] = path_weight):
    return min_weight_simple_paths_brute_force(graph, weight_fun).get(n, None)


def join_path(path1, path2):
    """Join two paths, assuming that they share an end.
    A path is a list of nodes.
    """
    if path1[-1] == path2[0]:
        return path1 + path2[1:]
    elif path2[-1] == path1[0]:
        return path2 + path1[1:]
    elif path1[-1] == path2[-1]:
        return path1 + path2[1::-1]
    elif path1[0] == path2[0]:
        return path2[:0:-1] + path1

    raise ValueError('Paths cannot be joined as they do not share any ends')


def min_weight_simple_path_greedy(
        graph: nx.Graph,
        n: int,
        weight_fun: Callable[[nx.Graph, List], float] = path_weight):
    """A greedy algorithm that tries to find a simple path consisting of n
    nodes in the given graph, with minimal total weight

    Args:
        graph: a nx.Graph object with specified edge weights
        n: desired number of nodes in the simple path
        weight_fun: a function that takes (graph, path) and gives a value
            based on edge and node weights that we want to minimize
            (default: uses path_weight)

    Returns:
        a list of nodes that describes the path, or None if not found
    """

    def _grow_path_lowest_weight(path, partial_graph):
        # grow path by one neighboring edge of lowest weight on partial_graph
        # assuming partial_graph contains no edge between head and tail of path
        adjacent_edges = sorted(list(partial_graph.edges([path[0], path[-1]],
                                                         data='weight')), key=lambda e: e[2])
        if len(adjacent_edges) == 0:
            return path

        u, v, _ = adjacent_edges[0]
        if path[0] == u or path[-1] == u:
            partial_graph.remove_node(u)
        else:
            partial_graph.remove_node(v)
        return join_path(path, [u, v])

    edges_sorted = sorted(list(graph.edges.data('weight')), key=lambda e: e[2])

    # keep only the first half since they have the lower weights, as it
    # seems unnecessary to start the greedy search with high-weight edges
    edges_sorted = edges_sorted[:len(edges_sorted) // 2]

    best_weight = float('inf')
    best_path = None
    for e in edges_sorted:
        subgraph = graph.copy()

        path = [e[0], e[1]]  # start with the given edge
        subgraph.remove_edge(e[0], e[1])
        while len(path) < n:
            # tries to grow the path by one neighboring edge of lowest weight
            path2 = _grow_path_lowest_weight(path, subgraph)
            if path2 == path:  # can't grow anymore
                break
            else:
                path = path2
                if subgraph.has_edge(path[0], path[-1]):
                    subgraph.remove_edge(path[0], path[-1])

        my_weight = weight_fun(graph, path)

        if len(path) == n and my_weight < best_weight:
            best_path = path
            best_weight = my_weight

    return best_path


def make_simple_path(graph: nx.Graph, n: int):
    """Make a simple path on a graph by starting with lowest degree node
    and adding nodes with low degree.
    For a given graph object, this is deterministic.

    Args:
        graph: a nx.Graph object
        n: desired number of nodes in the simple path

    Returns:
        a list of nodes that describes the path, or None if no such path exists
    """
    subgraph = graph.copy()
    path = n * [None]

    degree_dict = dict(subgraph.degree())
    path[0] = min(degree_dict, key=degree_dict.get)
    len_so_far = 1
    while len_so_far < n:
        neighbors = sorted([(v, dv) for v, dv
                            in subgraph.degree(subgraph.neighbors(path[len_so_far - 1]))],
                           key=lambda x: x[1])

        if len(neighbors) == 0:
            return None  # dead end, can't add any more nodes

        next_node = neighbors[0][0]
        for i in range(len(neighbors)):
            if neighbors[i][1] >= 2:  # if there's a neighbor with degree 2
                next_node = neighbors[i][0]  # change the next node to that
                break

        path[len_so_far] = next_node
        subgraph.remove_node(path[len_so_far - 1])
        len_so_far += 1
    return path


def random_simple_path(graph: nx.Graph,
                       n: int,
                       num_tries=10):
    """Tries to find a simple path by growing it from a random edge.
    Will use make_simple_path if it fails.

    Args:
        graph: a networkx.Graph object with specified edge weights
        n: desired number of nodes in the simple path
        num_tries: number of tries before resorting to make_simple_path

    Returns:
        a list of nodes that describes the path, or None if no such path exists
    """
    # first try the deterministic algorithm to see if a simple path of n nodes
    # even exists
    default_path = make_simple_path(graph, n)
    if default_path is None:
        return None

    def _grow_simple_path(path, subgraph):
        # grow path by adding one neighboring edge
        adjacent_edges = list(subgraph.edges([path[0], path[-1]]))
        if len(adjacent_edges) == 0:  # no neighboring edge
            return path

        # randomly choose a neighboring edge
        u, v = adjacent_edges[np.random.randint(len(adjacent_edges))]

        if path[0] == u or path[-1] == u:
            subgraph.remove_node(u)
        else:
            subgraph.remove_node(v)
        return join_path(path, [u, v])

    edges = list(graph.edges)

    for _ in range(num_tries):
        u, v = edges[np.random.randint(len(edges))]
        path = [u, v]

        subgraph = graph.copy()
        subgraph.remove_edge(u, v)
        while len(path) < n:
            new_path = _grow_simple_path(path, subgraph)
            if new_path == path:
                break
            path = new_path
            if subgraph.has_edge(path[0], path[-1]):
                subgraph.remove_edge(path[0], path[-1])

        if len(path) == n:
            return path

    # all tries have failed
    return default_path


class Snake:
    """A Snake is a simple path that can wiggle, slither, or reassemble.

    Used in simulated annealing algorithm for finding simple paths with minimum
    total weight.
    """

    def __init__(self,
                 graph: nx.Graph,
                 path: List,
                 weight_fun: Callable[[nx.Graph, List], float] = path_weight):
        if not nx.is_simple_path(graph, path):
            raise ValueError("Invalid input: path must be simple")
        self.graph = graph
        self.path = path
        self.weight_fun = weight_fun

    def wiggle(self):
        """Randomly change a node on the path to next-nearest neighbor,
        and returns a new Snake if the new path is a simple path.

        For example, on a grid graph, wiggling means:

            +--+--            +--
            |       =>        |
          --+            --+--+
        """
        node_index_list = list(range(len(self.path)))
        np.random.shuffle(node_index_list)
        for index in node_index_list:
            node = self.path[index]
            neighbors = nx.single_source_shortest_path_length(self.graph,
                                                              node, cutoff=2)
            for q in neighbors:
                # check if changing path[index] to q still gives a simple path
                if (q not in self.path
                        and neighbors[q] == 2
                        and (index == 0 or self.graph.has_edge(q, self.path[index - 1]))
                        and (index == len(self.path) - 1
                             or self.graph.has_edge(q, self.path[index + 1]))
                ):
                    new_path = self.path.copy()
                    new_path[index] = q
                    return Snake(self.graph, new_path, self.weight_fun)

        return self  # wasn't able to wiggle

    def slither(self, head=True):
        """Tries to move forward to a neighboring node.

        Set head=False to move backwards."""
        if not head:
            self.path.reverse()

        head_neighbors = list(self.graph.neighbors(self.path[0]))
        np.random.shuffle(head_neighbors)
        for q in head_neighbors:
            if q not in self.path or q == self.path[-1]:
                test_path = self.path.copy()
                test_path[1:] = test_path[:-1]
                test_path[0] = q
                return Snake(self.graph, test_path, self.weight_fun)

        if not head:
            # reverse back so you can try other moves in same orientation
            self.path.reverse()

        return self  # wasn't able to slither

    def reassemble(self, head=True):
        """If the head is near a part of the main body, break the path
        and reassemble to get a new head.

        Set head=False to reassemble at the tail.

        For example, on a grid graph, reassembling means:

            +--+--+           +--+--+
            |     |     =>          |
          --+  +--+         --+--+--+
        """
        if not head:
            self.path.reverse()

        head_neighbors = list(self.graph.neighbors(self.path[0]))
        np.random.shuffle(head_neighbors)
        for q in head_neighbors:
            if q != self.path[1] and q in self.path:
                new_path = self.path.copy()
                q_index = new_path.index(q)
                new_path[:q_index] = reversed(new_path[:q_index])
                return Snake(self.graph, new_path, self.weight_fun)

        if not head:
            # reverse back so you can try other moves in same orientation
            self.path.reverse()

        return self  # wasn't able to reassemble

    def random_move(self):
        """Randomly wiggle, slither, or reassemble."""
        coin = np.random.randint(3)
        if coin == 0:
            return self.wiggle()
        elif coin == 1:
            return self.slither(head=np.random.choice(True, False))
        else:
            return self.reassemble(head=np.random.choice(True, False))

    def force_random_move(self):
        """Force a random move among wiggle, slither forward or backward,
        or reassemble at head or tail.
        Returns self only if there is no move possible.
        """
        moves = [self.wiggle,
                 self.slither,
                 self.reassemble,
                 lambda: self.slither(head=False),
                 lambda: self.reassemble(head=False)]

        np.random.shuffle(moves)
        for mv in moves:
            new_snake = mv()
            if new_snake is not self:
                return new_snake
        return self  # wasn't able to make any moves

    def total_weight(self):
        """Returns the total weight of path"""
        return self.weight_fun(self.graph, self.path)

    def to_path(self):
        return self.path


def min_weight_simple_path_anneal(
        graph: nx.Graph,
        n: int,
        start_path=None,
        anneal_schedule: Optional[np.array] = None,
        weight_fun: Callable[[nx.Graph, List], float] = path_weight):
    """A simulated-annealing algorithm for finding a simple path consisting of
    n nodes in the given graph, with minimal total weight

    Args:
        graph: a networkx.Graph object with specified edge weights
        n: desired number of nodes in the simple path
        start_path: a path serving as starting point of simulated annealing
            (default is None, in which case uses random_simple_path)
        anneal_schedule: an array of temperature values used for annealing.
            The number of elements in the array is the number of steps.
            (default is None, in which case uses a linear schedule)

    Returns:
        a list of nodes that describes the path,
        or None if no simple path is found
    """
    if n < 2:
        raise ValueError('n needs to be >= 2')

    if start_path is None:
        if start_path is None:
            start_path = random_simple_path(graph, n)
        if start_path is None:
            return None
        my_snake = Snake(graph, start_path, weight_fun)
    else:
        my_snake = Snake(graph, start_path, weight_fun)

    my_E = my_snake.total_weight()

    best_E = my_E
    best_snake = my_snake

    if anneal_schedule is None:
        weights = [w for u, v, w in graph.edges(data='weight')]
        T_max = max(weights) * 2
        T_min = min(weights) / 2
        anneal_schedule = np.linspace(T_max, T_min, 3 * n * len(graph))

    for T in anneal_schedule:
        new_snake = my_snake.force_random_move()
        if new_snake is my_snake:
            # stuck as no move is possible
            break
        new_E = new_snake.total_weight()

        if new_E <= best_E:
            best_snake = new_snake
            best_E = new_E

        if np.exp(-(new_E - my_E) / T) >= np.random.rand():
            my_snake = new_snake
            my_E = new_E

    return best_snake.to_path()


def min_weight_simple_paths_mst(
        graph: nx.Graph,
        weight_fun: Callable[[nx.Graph, List], float] = path_weight):
    """A heuristic algorithm to find minimal weight simple paths by
    constructing the minimum spanning tree (MST).

    This works best to for simple paths of short lengths.
    """
    mst = nx.minimum_spanning_tree(graph)
    path_of_node_pairs = dict(nx.all_pairs_shortest_path(mst))
    best_paths = {}
    for u in path_of_node_pairs:
        for v in path_of_node_pairs[u]:
            n = len(path_of_node_pairs[u][v])
            if n < 2:
                continue
            if (n not in best_paths
                    or weight_fun(mst, best_paths[n])
                    > weight_fun(mst, path_of_node_pairs[u][v])):
                best_paths[n] = path_of_node_pairs[u][v]
    return best_paths


def min_weight_simple_path_mst(graph: nx.Graph, n: int,
                               weight_func: Callable[[nx.Graph, List], float] = path_weight):
    return min_weight_simple_paths_mst(graph, weight_func).get(n, None)


def min_weight_simple_path_mixed_strategy(
        graph: nx.Graph,
        n: int,
        num_restarts=10,
        weight_fun: Callable[[nx.Graph, List], float] = path_weight):
    """Find a simple path of minimal weight on a graph using a mixed strategy.

    We use the better of MST-based and greedy algorithm to generate a good
    starting point, and then further optimize using simulated annealing
    with restarts.

    Args:
        graph: a networkx.Graph object with specified edge weights
        n: desired number of nodes in the simple path
        num_restarts: number of restarts in simulated annealing

    Returns:
        a list of nodes that describes the best simple path found,
        or None if no simple path is found
    """
    paths_mst = min_weight_simple_paths_mst(graph)
    start_path = paths_mst.get(n, None)
    path_greedy = min_weight_simple_path_greedy(graph, n)
    if weight_fun(graph, path_greedy) < weight_fun(graph, start_path):
        start_path = path_greedy

    best_path = start_path

    for _ in range(num_restarts):
        path = min_weight_simple_path_anneal(graph, n, start_path=start_path)
        if weight_fun(graph, path) < weight_fun(graph, best_path):
            best_path = path
    return best_path


@lru_cache()
def _get_device_calibration(device_name: str):
    """Get device calibration. Use an LRU cache to avoid repeated calls to
    the web interface. It's possible this is not what you want.

    TODO: move to recirq.engine_utils.
    """
    processor_id = recirq.get_processor_id_by_device_name(device_name)
    if processor_id is None:
        # TODO: https://github.com/quantumlib/ReCirq/issues/14
        device_obj = recirq.get_device_obj_by_name(device_name)
        dummy_graph = ccr.gridqubits_to_graph_device(
            device_obj.metadata.qubit_set if device_obj.metadata is not None else ()
        )
        nx.set_edge_attributes(dummy_graph, name='weight', values=0.01)
        return dummy_graph

    calibration = cg.get_engine_calibration(processor_id)
    err_graph = calibration_data_to_graph(calibration)
    return err_graph


PLACEMENT_STRATEGIES = {
    'brute_force': min_weight_simple_path_brute_force,
    'random': random_simple_path,
    'greedy': min_weight_simple_path_greedy,
    'anneal': min_weight_simple_path_anneal,
    'mst': min_weight_simple_path_mst,
    'mixed': min_weight_simple_path_mixed_strategy,
}


def place_line_on_device(
        device_name: str,
        n: int,
        line_placement_strategy: str,
        err_graph=None,
) -> List[cirq.GridQubit]:
    if line_placement_strategy not in PLACEMENT_STRATEGIES.keys():
        raise ValueError(f"Unknown line placement strategy {line_placement_strategy}")

    if err_graph is None:
        err_graph = _get_device_calibration(device_name)

    return PLACEMENT_STRATEGIES[line_placement_strategy](err_graph, n)
