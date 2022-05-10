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

import networkx as nx
import numpy as np
import pytest

import cirq
import cirq_google as cg
from recirq.qaoa.circuit_structure import get_moment_class, get_moment_classes, \
    find_circuit_structure_violations, BadlyStructuredCircuitError, validate_well_structured
from recirq.qaoa.gates_and_compilation import ProblemUnitary, DriverUnitary, \
    compile_problem_unitary_to_swap_network, compile_swap_network_to_zzswap, \
    compile_driver_unitary_to_rx, compile_to_syc
from recirq.qaoa.problems import random_plus_minus_1_weights


def test_get_moment_class():
    q0, q1 = cirq.LineQubit.range(2)

    # Non-hardware
    assert get_moment_class(cirq.Moment([cirq.CNOT(q0, q1)])) is None
    # Non-homogenous
    assert get_moment_class(cirq.Moment([cirq.X(q0), cirq.Z(q1)])) is None

    # Both phasedxpow
    assert get_moment_class(cirq.Moment([
        cirq.PhasedXPowGate(phase_exponent=0).on(q0),
        cirq.PhasedXPowGate(phase_exponent=0.5).on(q1)])) == cirq.PhasedXPowGate

    # Both Z
    assert get_moment_class(cirq.Moment([
        cirq.ZPowGate(exponent=0.123).on(q0),
        cirq.ZPowGate(exponent=0.5).on(q1)])) == cirq.ZPowGate

    # SYC
    assert get_moment_class(cirq.Moment([cg.SYC(q0, q1)])) == cg.SycamoreGate

    # Measurement
    assert get_moment_class(cirq.Moment([cirq.measure(q0, q1)])) == cirq.MeasurementGate

    # Permutation
    assert get_moment_class(cirq.Moment([
        cirq.QubitPermutationGate([1, 0]).on(q0, q1)])) == cirq.QubitPermutationGate


def test_get_moment_classes():
    n = 5
    problem = nx.complete_graph(n=n)
    problem = random_plus_minus_1_weights(problem)
    qubits = cirq.LineQubit.range(n)
    p = 1
    c1 = cirq.Circuit(
        cirq.H.on_each(qubits),
        [
            [
                ProblemUnitary(problem, gamma=np.random.random()).on(*qubits),
                DriverUnitary(5, beta=np.random.random()).on(*qubits)
            ]
            for _ in range(p)
        ]
    )
    c2 = compile_problem_unitary_to_swap_network(c1)
    c3 = compile_swap_network_to_zzswap(c2)
    c4 = compile_driver_unitary_to_rx(c3)
    c5 = compile_to_syc(c4)
    mom_classes = get_moment_classes(c5)
    should_be = [cirq.PhasedXPowGate, cirq.ZPowGate, cg.SycamoreGate] * (n * p * 3)
    should_be += [cirq.PhasedXPowGate, cirq.ZPowGate, cirq.QubitPermutationGate]
    assert mom_classes == should_be


def test_find_circuit_structure_violations():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([
        cirq.Moment([cirq.PhasedXPowGate(phase_exponent=0).on(q0)]),
        cirq.Moment([
            cirq.PhasedXPowGate(phase_exponent=0.5).on(q0),
            cirq.ZPowGate(exponent=1).on(q1)]),
        cirq.Moment([cg.SYC(q0, q1)]),
        cirq.measure(q0, q1, key='z'),
        cirq.CNOT(q0, q1)
    ])
    violations = find_circuit_structure_violations(circuit)
    np.testing.assert_array_equal(violations, [1, 4])
    assert violations.tolist() == [1, 4]


def test_validate_well_structured_inhomo():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([
        cirq.Moment([cirq.PhasedXPowGate(phase_exponent=0).on(q0)]),
        cirq.Moment([
            cirq.PhasedXPowGate(phase_exponent=0.5).on(q0),
            cirq.ZPowGate(exponent=1).on(q1)]),
        cirq.Moment([cg.SYC(q0, q1)]),
        cirq.measure(q0, q1, key='z'),
    ])

    with pytest.raises(BadlyStructuredCircuitError) as e:
        validate_well_structured(circuit)
    assert e.match('Inhomogeneous')


def test_validate_well_structured_bad_gate():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([
        cirq.Moment([cirq.PhasedXPowGate(phase_exponent=0).on(q0)]),
        cirq.Moment([
            cirq.XPowGate(exponent=0.5).on(q0), ]),
        cirq.Moment([cg.SYC(q0, q1)]),
        cirq.measure(q0, q1, key='z'),
    ])

    with pytest.raises(BadlyStructuredCircuitError) as e:
        validate_well_structured(circuit)
    assert e.match('non-device')


def test_validate_well_structured_too_many():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([
        cirq.Moment([cirq.PhasedXPowGate(phase_exponent=0).on(q0)]),
        cirq.Moment([
            cirq.PhasedXPowGate(phase_exponent=0.5).on(q0), ]),
        cirq.Moment([cg.SYC(q0, q1)]),
        cirq.measure(q0, q1, key='z'),
    ])

    with pytest.raises(BadlyStructuredCircuitError) as e:
        validate_well_structured(circuit)
    assert e.match('Too many PhX')


def test_validate_well_structured_non_term_meas():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([
        cirq.Moment([cirq.PhasedXPowGate(phase_exponent=0).on(q0)]),
        cirq.Moment([
            cirq.PhasedXPowGate(phase_exponent=0.5).on(q0), ]),
        cirq.measure(q0, q1, key='z'),
        cirq.Moment([cg.SYC(q0, q1)]),
    ])

    with pytest.raises(BadlyStructuredCircuitError) as e:
        validate_well_structured(circuit)
    assert e.match('Measurements must be terminal')


def test_validate_well_structured():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([
        cirq.Moment([cirq.PhasedXPowGate(phase_exponent=0).on(q0)]),
        cirq.Moment([
            cirq.ZPowGate(exponent=0.5).on(q0), ]),
        cirq.Moment([cg.SYC(q0, q1)]),
        cirq.measure(q0, q1, key='z'),
    ])
    validate_well_structured(circuit)
