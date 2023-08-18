# Copyright 2023 Google
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

import cirq
import numpy as np
from openfermion import QubitOperator

from recirq.seniority_zero.data_processing.general import (
    energy_from_expvals,
    get_ls_echo_fidelity,
    get_p0m_count_verified,
    get_signed_count,
    get_signed_count_postselected,
    get_signed_count_verified,
    multi_measurement_histogram,
    order_parameter_from_expvals,
    order_parameter_var_from_expvals,
    vectorize_expvals,
)


def test_multi_measurement_histogram():
    circuit = cirq.Circuit(
        [
            cirq.H(cirq.GridQubit(0, 0)),
            cirq.CNOT(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
            cirq.measure(cirq.GridQubit(0, 0), key='a'),
            cirq.measure(cirq.GridQubit(0, 1), key='b'),
        ]
    )
    simulator = cirq.Simulator()
    res = simulator.run(circuit, repetitions=10000)
    mmh = multi_measurement_histogram(res, keys=['a', 'b'])
    assert (0, 0) in mmh
    assert (1, 1) in mmh


def test_vectorize_expvals():
    num_qubits = 2
    expvals = {
        str(QubitOperator('Z0')): 1,
        str(QubitOperator('Z1')): 0.5,
        str(QubitOperator('X0 X1')): 0.25,
    }
    vector, covars = vectorize_expvals(num_qubits, expvals)
    assert vector[0] == 1
    assert vector[1] == 0.5
    assert vector[2] == 0.25


def test_energy_from_expvals():
    num_qubits = 2
    expvals = {
        str(QubitOperator('Z0')): 1,
        str(QubitOperator('Z1')): 0.5,
        str(QubitOperator('X0 X1')): 0.25,
    }
    hamiltonian = QubitOperator('Z0') * 2
    hamiltonian += QubitOperator('Z1') * 1
    hamiltonian += QubitOperator('X0 X1') * 5
    energy, variance = energy_from_expvals(num_qubits, expvals, hamiltonian)
    assert variance is None
    assert abs(energy - 3.75) < 1e-6


def test_order_parameter_from_expvals():
    num_qubits = 2
    expvals = {
        str(QubitOperator('Z0')): 1,
        str(QubitOperator('Z1')): 0,
        str(QubitOperator('X0 X1')): 0.25,
    }
    op = order_parameter_from_expvals(num_qubits, expvals)
    assert abs(op - 0.5) < 1e-6


def test_order_parameter_var_from_expvals():
    num_qubits = 2
    key1 = str(QubitOperator('Z0'))
    key2 = str(QubitOperator('Z1'))
    expvals = {key1: 1, key2: 0}
    covars = {key1: {key1: 0, key2: 0}, key2: {key1: 0, key2: 0}}
    op_var = order_parameter_var_from_expvals(num_qubits, expvals, covars)
    assert abs(op_var) < 1e-6


def test_get_ls_echo_fidelity():
    q = cirq.GridQubit(0, 0)
    q2 = cirq.GridQubit(0, 1)
    circuit = cirq.Circuit(
        [
            cirq.H(q),
            cirq.CNOT(q, q2),
            cirq.CNOT(q, q2),
            cirq.H(q),
            cirq.measure(q, key=str(q)),
            cirq.measure(q2, key=str(q2)),
        ]
    )
    simulator = cirq.Simulator()
    res = simulator.run(circuit, repetitions=10000)
    fidel, fvar = get_ls_echo_fidelity(res.data, [q, q2], None)
    assert np.abs(fidel - 1) < 1e-6
    assert np.abs(fvar) < 1e-6


def test_get_signed_count():
    q = cirq.GridQubit(0, 0)
    q2 = cirq.GridQubit(0, 1)
    circuit = cirq.Circuit(
        [cirq.H(q), cirq.CNOT(q, q2), cirq.measure(q, key=str(q)), cirq.measure(q2, key=str(q2))]
    )
    simulator = cirq.Simulator()
    res = simulator.run(circuit, repetitions=10000)
    sc, tc = get_signed_count(res.data, [q, q2], [q, q2])
    assert tc == 10000
    assert sc == 10000


def test_get_signed_count_verified():
    q = cirq.GridQubit(0, 0)
    q2 = cirq.GridQubit(0, 1)
    q3 = cirq.GridQubit(0, 2)
    circuit = cirq.Circuit(
        [
            cirq.H(q),
            cirq.CNOT(q, q2),
            cirq.X(q3),
            cirq.measure(q, key=str(q)),
            cirq.measure(q2, key=str(q2)),
            cirq.measure(q3, key=str(q3)),
        ]
    )
    simulator = cirq.Simulator()
    res = simulator.run(circuit, repetitions=10000)
    sc, scvf, tc = get_signed_count_verified(res.data, [q, q2], [q, q2, q3], 1)
    assert tc == 10000
    assert sc == 10000
    assert scvf == 10000


def test_get_signed_count_postselected():
    q = cirq.GridQubit(0, 0)
    q2 = cirq.GridQubit(0, 1)
    circuit = cirq.Circuit(
        [cirq.H(q), cirq.CNOT(q, q2), cirq.measure(q, key=str(q)), cirq.measure(q2, key=str(q2))]
    )
    simulator = cirq.Simulator()
    res = simulator.run(circuit, repetitions=10000)
    sc, tc = get_signed_count_postselected(res.data, [q, q2], [q, q2], 2)
    assert abs(tc - 5000) < 100
    assert abs(sc - 5000) < 100


def test_get_p0m_count_verified():
    q = cirq.GridQubit(0, 0)
    q2 = cirq.GridQubit(0, 1)
    q3 = cirq.GridQubit(0, 2)
    circuit = cirq.Circuit(
        [
            cirq.H(q),
            cirq.CNOT(q, q2),
            cirq.X(q3),
            cirq.measure(q, key=str(q)),
            cirq.measure(q2, key=str(q2)),
            cirq.measure(q3, key=str(q3)),
        ]
    )
    simulator = cirq.Simulator()
    res = simulator.run(circuit, repetitions=10000)
    np, nz, nm = get_p0m_count_verified(res.data, [q, q2], [q, q2, q3], 1)
    assert np == 10000
    assert nz == 0
    assert nm == 0
