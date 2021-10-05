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
import numpy as np
import cirq

import recirq.quantum_chess.quantum_moves as qm


def test_merge():
    """Tests Merge jump with eqs 11a-f in quantum chess paper."""
    s = cirq.Simulator()
    isqrt2 = 1 / np.sqrt(2)
    s1, s2, t = cirq.LineQubit.range(3)

    # Merge |001> = −i|100>
    c = cirq.Circuit(cirq.X(t), qm.merge_move(s1, s2, t))
    expected = [0, 0, 0, 0, -1j, 0, 0, 0]
    actual = s.simulate(c).state_vector()
    np.testing.assert_almost_equal(actual, expected)

    # Merge |010> = -1/√2 (i|001> −  |010>)
    c = cirq.Circuit(cirq.X(s2), qm.merge_move(s1, s2, t))
    expected = [0, isqrt2 * -1j, isqrt2, 0, 0, 0, 0, 0]
    actual = s.simulate(c).state_vector()
    np.testing.assert_almost_equal(actual, expected)

    # Merge |100> = -1/√2 (i|001> +  |010>)
    c = cirq.Circuit(cirq.X(s1), qm.merge_move(s1, s2, t))
    expected = [0, isqrt2 * -1j, -1 * isqrt2, 0, 0, 0, 0, 0]
    actual = s.simulate(c).state_vector()
    np.testing.assert_almost_equal(actual, expected)

    # Merge |011> = -1/√2 ( |101> + i|110>)
    c = cirq.Circuit(cirq.X(s2), cirq.X(t), qm.merge_move(s1, s2, t))
    expected = [0, 0, 0, 0, 0, -1 * isqrt2, -1j * isqrt2, 0]
    actual = s.simulate(c).state_vector()
    np.testing.assert_almost_equal(actual, expected)

    # Merge |101> = -i/√2 (i|101> +  |110>)
    c = cirq.Circuit(cirq.X(s1), cirq.X(t), qm.merge_move(s1, s2, t))
    expected = [0, 0, 0, 0, 0, isqrt2, -1j * isqrt2, 0]
    actual = s.simulate(c).state_vector()
    np.testing.assert_almost_equal(actual, expected)

    # Merge |110> = −i|011>
    c = cirq.Circuit(cirq.X(s1), cirq.X(s2), qm.merge_move(s1, s2, t))
    expected = [0, 0, 0, -1j, 0, 0, 0, 0]
    actual = s.simulate(c).state_vector()
    np.testing.assert_almost_equal(actual, expected)


def test_merge_slide():
    s = cirq.Simulator()
    s1, s2, t, p1, p2, a = cirq.LineQubit.range(6)
    measure = cirq.Moment(
        cirq.measure(s1, key="s1"), cirq.measure(s2, key="s2"), cirq.measure(t, key="t")
    )

    # If both paths are clear, the first source square should always be empty.
    c = cirq.Circuit(
        cirq.X(p1),
        cirq.X(p2),
        cirq.X(s1),
        qm.merge_slide(s1, t, s2, p1, p2, a),
        measure,
    )
    results = s.run(c, repetitions=100)
    assert all(r == [0] for r in results.measurements["s1"])
