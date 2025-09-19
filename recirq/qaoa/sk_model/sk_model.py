# Copyright 2022 Google
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
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Sequence

import networkx as nx
import numpy as np

import cirq
from cirq.protocols import dataclass_json_dict
from cirq_google.workflow import QuantumExecutable, BitstringsMeasurement, QuantumExecutableGroup, \
    ExecutableSpec
from recirq.qaoa.classical_angle_optimization import optimize_instance_interp_heuristic
from recirq.qaoa.problem_circuits import get_routed_sk_model_circuit


def _graph_from_row_major_upper_triangular(
        all_to_all_couplings: Sequence[float], *, n: int
) -> nx.Graph:
    """Get `all_to_all_couplings` in the form of a NetworkX graph."""
    if not len(all_to_all_couplings) == n * (n - 1) / 2:
        raise ValueError("Number of couplings does not match the number of nodes.")

    g = nx.Graph()
    for (u, v), coupling in zip(itertools.combinations(range(n), r=2), all_to_all_couplings):
        g.add_edge(u, v, weight=coupling)
    return g


def _all_to_all_couplings_from_graph(graph: nx.Graph) -> Tuple[int, ...]:
    """Given a networkx graph, turn it into a tuple of all-to-all couplings."""
    n = graph.number_of_nodes()
    if not sorted(graph.nodes) == sorted(range(n)):
        raise ValueError("Nodes must be contiguous and zero-indexed.")

    edges = graph.edges
    return tuple(edges[u, v]['weight'] for u, v in itertools.combinations(range(n), r=2))


@dataclass(frozen=True)
class SKModelQAOASpec(ExecutableSpec):
    """ExecutableSpec for running SK-model QAOA.

    QAOA uses alternating applications of a problem-specific entangling unitary and a
    problem-agnostic driver unitary. It is a variational algorithm, but for this spec
    we rely on optimizing the angles via classical simulation.

    The SK model is an all-to-all 2-body spin problem that we can route using the
    "swap network" to require only linear connectivity (but circuit depth scales with problem
    size)

    Args:
        n_nodes: The number of nodes in the SK problem. This is equal to the number of qubits.
        all_to_all_couplings: The n(n-1)/2 pairwise coupling constants that defines the problem
            as a serializable tuple of the row-major upper triangular coupling matrix.
        p_depth: The depth hyperparemeter that presecribes the number of U_problem * U_driver
            repetitions.
        n_repetitions: The number of shots to take when running the circuits.
        executable_family: `recirq.qaoa.sk_model`.

    """

    n_nodes: int
    all_to_all_couplings: Tuple[int, ...]
    p_depth: int
    n_repetitions: int
    executable_family: str = 'recirq.qaoa.sk_model'

    def __post_init__(self):
        object.__setattr__(self, 'all_to_all_couplings', tuple(self.all_to_all_couplings))

    def get_graph(self) -> nx.Graph:
        """Get `all_to_all_couplings` in the form of a NetworkX graph."""
        return _graph_from_row_major_upper_triangular(self.all_to_all_couplings, n=self.n_nodes)

    @staticmethod
    def get_all_to_all_couplings_from_graph(graph: nx.Graph) -> Tuple[int, ...]:
        """Given a networkx graph, turn it into a tuple of all-to-all couplings."""
        return _all_to_all_couplings_from_graph(graph)

    @classmethod
    def _json_namespace_(cls):
        return 'recirq.qaoa'

    def _json_dict_(self):
        return dataclass_json_dict(self, namespace=self._json_namespace_())


def _classically_optimize_qaoa_parameters(graph: nx.Graph, *, n: int, p_depth: int):
    param_guess = [
        np.arccos(np.sqrt((1 + np.sqrt((n - 2) / (n - 1))) / 2)),
        -np.pi / 8
    ]

    optima = optimize_instance_interp_heuristic(
        graph=graph,
        # Potential performance improvement: To optimize for a given p_depth,
        # we also find the optima for lower p values.
        # You could cache these instead of re-finding for each executable.
        p_max=p_depth,
        param_guess_at_p1=param_guess,
        verbose=True,
    )
    # The above returns a list, but since we asked for p_max = spec.p_depth,
    # we always want the last one.
    optimum = optima[-1]
    assert optimum.p == p_depth
    return optimum


def sk_model_qaoa_spec_to_exe(
        spec: SKModelQAOASpec,
) -> QuantumExecutable:
    """Create a full `QuantumExecutable` from a given `SKModelQAOASpec`

    Args:
        spec: The spec

    Returns:
        a QuantumExecutable corresponding to the input specification.
    """
    n = spec.n_nodes
    graph = spec.get_graph()

    # Get params
    optimum = _classically_optimize_qaoa_parameters(graph, n=n, p_depth=spec.p_depth)

    # Make the circuit
    qubits = cirq.LineQubit.range(n)
    circuit = get_routed_sk_model_circuit(
        graph, qubits, optimum.gammas, optimum.betas, keep_zzswap_as_one_op=False)

    # QAOA code optionally finishes with a QubitPermutationGate, which we want to
    # absorb into measurement. Maybe at some point this can be part of
    # `cg.BitstringsMeasurement`, but for now we'll do it implicitly in the analysis code.
    if spec.p_depth % 2 == 1:
        assert len(circuit[-1]) == 1
        permute_op, = circuit[-1]
        assert isinstance(permute_op.gate, cirq.QubitPermutationGate)
        circuit = circuit[:-1]

    # Measure
    circuit += cirq.measure(*qubits, key='z')

    return QuantumExecutable(
        spec=spec,
        problem_topology=cirq.LineTopology(n),
        circuit=circuit,
        measurement=BitstringsMeasurement(spec.n_repetitions),
    )
