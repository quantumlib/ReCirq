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
from typing import List

import numpy as np

import cirq
from cirq import TiltedSquareLattice
from cirq.experiments import random_rotations_between_grid_interaction_layers_circuit
from cirq.protocols import dataclass_json_dict
from cirq_google.workflow import QuantumExecutable, BitstringsMeasurement, QuantumExecutableGroup, \
    ExecutableSpec


def create_tilted_square_lattice_loschmidt_echo_circuit(
        topology: TiltedSquareLattice,
        macrocycle_depth: int,
        twoq_gate: cirq.Gate = cirq.FSimGate(np.pi / 4, 0.0),
        rs: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
) -> cirq.FrozenCircuit:
    """Returns a Loschmidt echo circuit using a random unitary U.

    Args:
        topology: The TiltedSquareLattice topology.
        macrocycle_depth: Number of complete, macrocycles in the random unitary. A macrocycle is
            4 cycles corresponding to the 4 grid directions. Each cycle is one layer of entangling
            gates and one layer of random single qubit rotations. The total circuit depth is
            twice as large because we also do the inverse.
        twoq_gate: Two-qubit gate to use.
        rs: The random state for random circuit generation.
    """

    # Forward (U) operations.
    exponents = np.linspace(0, 7 / 4, 8)
    single_qubit_gates = [
        cirq.PhasedXZGate(x_exponent=0.5, z_exponent=z, axis_phase_exponent=a)
        for a, z in itertools.product(exponents, repeat=2)
    ]
    qubits = sorted(topology.nodes_as_gridqubits())
    forward = random_rotations_between_grid_interaction_layers_circuit(
        # note: this function should take a topology probably.
        qubits=qubits,
        # note: in this function, `depth` refers to cycles.
        depth=4 * macrocycle_depth,
        two_qubit_op_factory=lambda a, b, _: twoq_gate.on(a, b),
        pattern=cirq.experiments.GRID_STAGGERED_PATTERN,
        single_qubit_gates=single_qubit_gates,
        seed=rs
    )

    # Reverse (U^\dagger) operations.
    reverse = cirq.inverse(forward)

    return (forward + reverse + cirq.measure(*qubits, key='z')).freeze()


def _get_all_tilted_square_lattices(min_side_length=2, max_side_length=8, side_length_step=2):
    """Helper function to get a range of tilted square lattices."""
    width_heights = np.arange(min_side_length, max_side_length + 1, side_length_step)
    return [TiltedSquareLattice(width, height)
            for width, height in itertools.combinations_with_replacement(width_heights, r=2)]


@dataclass(frozen=True)
class TiltedSquareLatticeLoschmidtSpec(ExecutableSpec):
    """The ExecutableSpec for the tilted square lattice loschmidt benchmark.

    The loschmidt echo runs a random unitary forward and backwards and measures how often
    we return to the starting state. This benchmark checks random unitaries generated
    on the TiltedSquareLattice topology.

    Args:
        topology: The topology
        macrocycle_depth: Number of complete, macrocycles in the random unitary. A macrocycle is
            4 cycles corresponding to the 4 grid directions. Each cycle is one layer of entangling
            gates and one layer of random single qubit rotations. The total circuit depth is
            twice as large because we also do the inverse.
        instance_i: An arbitary index into the random instantiation
        n_repetitions: The number of repetitions to sample to measure the return probability.
        executable_family: The globally unique string identifier for this benchmark.
        twoq_gate_name: The name of the two-qubit entangling gate used in the random unitary. See
            `tilted_square_lattice_spec_to_exe` for currently supported values.
    """

    topology: TiltedSquareLattice
    macrocycle_depth: int
    instance_i: int
    n_repetitions: int
    twoq_gate_name: str = 'sqrt_iswap'
    executable_family: str = 'recirq.otoc.loschmidt.tilted_square_lattice'

    @classmethod
    def _json_namespace_(cls):
        return 'recirq.otoc'

    def _json_dict_(self):
        return dataclass_json_dict(self)


def get_all_tilted_square_lattice_specs(
        *, n_instances=10, n_repetitions=1_000,
        min_side_length=2, max_side_length=8,
        side_length_step=2,
        macrocycle_depths=None,
        twoq_gate_name='sqrt_iswap',
) -> List[TiltedSquareLatticeLoschmidtSpec]:
    """Return a collection of quantum executables for various parameter settings of the tilted
    square lattice loschmidt benchmark.

    Args:
        n_instances: The number of random instances to make per setting
        n_repetitions: The number of circuit repetitions to use for measuring the return
            probability.
        min_side_length, max_side_length, side_length_step: generate a range of
            TiltedSquareLattice topologies with widths and heights in this range.
        macrocycle_depths: The collection of macrocycle depths to use per setting.
        twoq_gate_name: The name of the two-qubit entangling gate used in the random unitary.
    """

    if macrocycle_depths is None:
        macrocycle_depths = np.arange(2, 8 + 1, 2)
    topologies = _get_all_tilted_square_lattices(
        min_side_length=min_side_length,
        max_side_length=max_side_length,
        side_length_step=side_length_step,
    )

    return [
        TiltedSquareLatticeLoschmidtSpec(
            topology=topology,
            macrocycle_depth=macrocycle_depth,
            instance_i=instance_i,
            n_repetitions=n_repetitions,
            twoq_gate_name=twoq_gate_name,
        )
        for topology, macrocycle_depth, instance_i in itertools.product(
            topologies, macrocycle_depths, range(n_instances))
    ]


def tilted_square_lattice_spec_to_exe(
        spec: TiltedSquareLatticeLoschmidtSpec, *, rs: np.random.RandomState
) -> QuantumExecutable:
    """Create a full `QuantumExecutable` from a given `TiltedSquareLatticeLoschmidtSpec`

    This "fleshes out" the specification into a complete executable with a random circuit.
    The spec's `twoq_gate_name` must be one of "sqrt_iswap" or "cz".

    Args:
        spec: The spec
        rs: A random state. The ExecutableSpec only specifies an `instance_i` and this function
            is responsible for generating a pseudo-random circuit for each `instance_i`. Therefore,
            some care should be taken when using this function to share a `RandomState` among
            all calls to this function.

    Returns:
        a QuantumExecutable corresponding to the input specification.
    """
    twoq_gates = {
        'sqrt_iswap': cirq.SQRT_ISWAP,
        'cz': cirq.CZ,
    }

    return QuantumExecutable(
        spec=spec,
        problem_topology=spec.topology,
        circuit=create_tilted_square_lattice_loschmidt_echo_circuit(
            topology=spec.topology,
            macrocycle_depth=spec.macrocycle_depth,
            twoq_gate=twoq_gates[spec.twoq_gate_name],
            rs=rs,
        ),
        measurement=BitstringsMeasurement(spec.n_repetitions)
    )


def get_all_tilted_square_lattice_executables(
        *, n_instances=10, n_repetitions=1_000,
        min_side_length=2, max_side_length=8,
        side_length_step=2,
        macrocycle_depths=None, seed=52,
        twoq_gate_name='sqrt_iswap',
) -> QuantumExecutableGroup:
    """Return a collection of quantum executables for various parameter settings of the tilted
    square lattice loschmidt benchmark.

    Args:
        n_instances: The number of random instances to make per setting
        n_repetitions: The number of circuit repetitions to use for measuring the return
            probability.
        min_side_length, max_side_length, side_length_step: generate a range of
            TiltedSquareLattice topologies with widths and heights in this range.
        seed: The random seed to make this deterministic.
        macrocycle_depths: The collection of macrocycle depths to use per setting.
        twoq_gate_name: The name of the two-qubit entangling gate used in the random unitary.
    """
    rs = np.random.RandomState(seed)
    specs = get_all_tilted_square_lattice_specs(
        n_instances=n_instances, n_repetitions=n_repetitions, min_side_length=min_side_length,
        max_side_length=max_side_length, side_length_step=side_length_step,
        macrocycle_depths=macrocycle_depths, twoq_gate_name=twoq_gate_name,
    )

    return QuantumExecutableGroup([
        tilted_square_lattice_spec_to_exe(spec, rs=rs)
        for spec in specs
    ])
