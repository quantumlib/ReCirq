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

import numpy as np

import cirq
import cirq_google as cg
# Need this exact import for monkeypatching to work (below)
import recirq.otoc.loschmidt.tilted_square_lattice.tilted_square_lattice
from recirq.otoc.loschmidt.tilted_square_lattice import (
    create_tilted_square_lattice_loschmidt_echo_circuit, TiltedSquareLatticeLoschmidtSpec,
    get_all_tilted_square_lattice_executables, get_all_tilted_square_lattice_specs,
    tilted_square_lattice_spec_to_exe,
)


def test_create_tilted_square_lattice_loschmidt_echo_circuit():
    topology = cirq.TiltedSquareLattice(width=3, height=2)
    macrocycle_depth = 2
    circuit = create_tilted_square_lattice_loschmidt_echo_circuit(
        topology=topology, macrocycle_depth=macrocycle_depth, rs=np.random.RandomState(52)
    )
    assert isinstance(circuit, cirq.FrozenCircuit)

    assert len(circuit.all_qubits()) == topology.n_nodes
    assert sorted(circuit.all_qubits()) == sorted(topology.nodes_as_gridqubits())

    edge_coloring_n = 4  # grid
    forward_backward = 2
    n_moment_per_microcycle = 2  # layer of single- and two- qubit gate
    measure_moment = 1
    extra_single_q_layer = 1
    assert len(circuit) == (edge_coloring_n * macrocycle_depth *
                            n_moment_per_microcycle + extra_single_q_layer) \
           * forward_backward + measure_moment


def test_tilted_square_lattice_loschmidt_spec(tmpdir):
    topology = cirq.TiltedSquareLattice(width=3, height=2)
    macrocycle_depth = 2
    spec1 = TiltedSquareLatticeLoschmidtSpec(
        topology=topology,
        macrocycle_depth=macrocycle_depth,
        instance_i=0,
        n_repetitions=10_000,
    )
    assert spec1.executable_family == 'recirq.otoc.loschmidt.tilted_square_lattice'

    fn = f'{tmpdir}/spec.json'
    cirq.to_json(spec1, fn)
    spec2 = cirq.read_json(fn)
    assert spec1 == spec2


def test_get_all_tilted_square_lattice_executables_default_args(monkeypatch):
    call_count = 0

    # Creating random circuits is expensive, but this test tests the default arguments for
    # this function and asserts that it's called the correct number of times.

    def mock_get_circuit(topology: cirq.TiltedSquareLattice, macrocycle_depth: int,
                         twoq_gate: cirq.Gate = cirq.FSimGate(np.pi / 4, 0.0),
                         rs: cirq.RANDOM_STATE_OR_SEED_LIKE = None):
        nonlocal call_count
        call_count += 1
        return cirq.Circuit()

    monkeypatch.setattr(recirq.otoc.loschmidt.tilted_square_lattice.tilted_square_lattice,
                        "create_tilted_square_lattice_loschmidt_echo_circuit", mock_get_circuit)
    get_all_tilted_square_lattice_executables()
    n_instances = 10
    n_macrocycle_depths = 4  # 2,4,6,8
    n_side_lengths = 4  # width or height # of possibilities
    n_topos = n_side_lengths * (n_side_lengths + 1) / 2
    assert call_count == n_instances * n_macrocycle_depths * n_topos


def test_get_all_tilted_square_lattice_executables():
    exes = get_all_tilted_square_lattice_executables(
        n_instances=2,
        min_side_length=2,
        max_side_length=3,
        side_length_step=1,
        macrocycle_depths=[2, 4],
    )
    assert sorted({exe.spec.instance_i for exe in exes}) == [0, 1]
    assert sorted({exe.spec.macrocycle_depth for exe in exes}) == [2, 4]
    assert sorted({exe.spec.topology.width for exe in exes}) == [2, 3]
    assert sorted({exe.spec.topology.height for exe in exes}) == [2, 3]


def test_get_all_tilted_square_lattice_specs():
    specs = get_all_tilted_square_lattice_specs()
    assert len(specs) == 400
    for spec in specs:
        assert spec.twoq_gate_name == 'sqrt_iswap'


def test_tilted_square_lattice_spec_to_exe():
    rs = np.random.RandomState()
    exe = tilted_square_lattice_spec_to_exe(TiltedSquareLatticeLoschmidtSpec(
        topology=cirq.TiltedSquareLattice(2, 2),
        macrocycle_depth=1, instance_i=123, n_repetitions=1_000
    ), rs=rs)
    assert isinstance(exe, cg.QuantumExecutable)
    assert exe.measurement.n_repetitions == 1_000
    assert len(exe.circuit) == 18 + 1
