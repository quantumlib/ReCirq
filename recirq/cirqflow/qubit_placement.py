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

"""NOTE: A lot of classes in this module will eventually be available in Cirq. They
are reproduced here for convenience."""
import abc
from functools import lru_cache
from typing import Optional, AbstractSet, Dict, List, Any, Tuple

import cirq
import networkx as nx
import numpy as np
from cirq.protocols import obj_to_dict_helper
from cirq.devices.named_topologies import TiltedSquareLattice, get_placements, NamedTopology


class GraphDevice(cirq.Device):
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.qubits = sorted(cirq.GridQubit(*rc) for rc in graph.nodes)

    def qubit_set(self) -> Optional[AbstractSet['cirq.Qid']]:
        return frozenset(self.qubits)

    def validate_circuit(self, circuit: 'cirq.Circuit') -> None:
        circuit_qubits = circuit.all_qubits()
        dev_qubits = self.qubit_set()

        bad_qubits = circuit_qubits - dev_qubits
        if bad_qubits:
            raise ValueError(f"Circuit qubits {bad_qubits} don't exist on "
                             f"device with qubits {dev_qubits}.")


class QubitPlacer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def place_circuit(self, circuit: cirq.AbstractCircuit, problem_topo: NamedTopology) -> Tuple[
        cirq.FrozenCircuit, Dict[Any, cirq.Qid]]:
        pass


class NaiveQubitPlacer(QubitPlacer):
    def place_circuit(self, circuit: cirq.AbstractCircuit, problem_topo: NamedTopology) -> Tuple[
        cirq.FrozenCircuit, Dict[Any, cirq.Qid]]:
        return circuit.freeze(), {q: q for q in circuit.all_qubits()}


    def _json_dict_(self):
        return obj_to_dict_helper(self, attribute_names=[], namespace='cirq.google')


@lru_cache()
def cached_get_placements(problem_topo: TiltedSquareLattice, device: GraphDevice) -> List[Dict]:
    """Cache placements onto the specific device."""
    return get_placements(big_graph=device.graph, small_graph=problem_topo.graph)


class CouldNotPlaceError(RuntimeError):
    """Raised if a problem topology could not be placed on a device graph."""


def get_random_placement(problem_topo: TiltedSquareLattice, device: GraphDevice,
                         rs: np.random.RandomState) -> Dict:
    """Place `problem_topo` randomly onto a device."""
    placements = cached_get_placements(problem_topo, device)
    if len(placements) == 0:
        raise CouldNotPlaceError
    random_i = int(rs.random_integers(0, len(placements) - 1, size=1))
    placement = placements[random_i]
    placement_gq = {cirq.GridQubit(*k): cirq.GridQubit(*v) for k, v in placement.items()}
    return placement_gq


class RandomDevicePlacer(QubitPlacer):
    def __init__(
            self,
            device: GraphDevice,
            rs: np.random.RandomState,
    ):
        self.device = device
        self.rs = rs

    def place_circuit(self, circuit: cirq.AbstractCircuit, problem_topo: NamedTopology) -> Tuple[
        cirq.FrozenCircuit, Dict[Any, cirq.Qid]]:
        placement = get_random_placement(problem_topo, self.device, rs=self.rs)
        return circuit.unfreeze().transform_qubits(placement).freeze(), placement
