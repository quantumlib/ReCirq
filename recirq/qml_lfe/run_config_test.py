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
import pytest
import cirq
from recirq.qml_lfe import run_config
import numpy as np


def test_flatten_circuit():
    q0, q1 = cirq.LineQubit.range(2)
    initial = cirq.Circuit(
        [
            cirq.Moment([cirq.H(q0)]),
            cirq.Moment([cirq.H(q1)]),
        ]
    )
    expected = cirq.Circuit([cirq.Moment([cirq.H(q0), cirq.H(q1)])])
    assert run_config.flatten_circuit(initial) == expected


def test_execute():
    q0, q1 = cirq.LineQubit.range(2)
    circuits = [
        cirq.Circuit(
            cirq.X(q0),
            cirq.H(q1),
            cirq.measure(q0, key="q0"),
            cirq.measure(q1, key="q1"),
        ),
        cirq.Circuit(cirq.H(q0), cirq.H(q1), cirq.measure(q0, key="q0")),
    ]
    results = run_config.execute_batch(circuits, 500, False)
    assert len(results) == 2
    assert results[0].data[["q0", "q1"]].to_numpy().shape == (500, 2)
    np.testing.assert_allclose(
        np.mean(results[0].data[["q0", "q1"]].to_numpy(), axis=0), [1, 0.5], atol=0.1
    )
    assert results[1].data[["q0"]].to_numpy().shape == (500, 1)
    np.testing.assert_allclose(
        np.mean(results[1].data[["q0"]].to_numpy(), axis=0), [0.5], atol=0.1
    )


def test_qubit_pairs():
    all_pairs = run_config.qubit_pairs()
    for pair, next_pair in zip(all_pairs, all_pairs[1:]):
        assert pair[0].is_adjacent(next_pair[0])
        assert pair[0].is_adjacent(pair[1])
