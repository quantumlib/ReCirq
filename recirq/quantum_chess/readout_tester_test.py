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
#
#
import cirq
import cirq_google as cg

import recirq.quantum_chess.readout_tester as readout_tester


def test_readout_tester():
    device = cg.Sycamore23
    sim = cirq.Simulator()

    tester = readout_tester.ReadoutTester(sampler=sim, device=device)

    for is_p11 in [True, False]:
        results = tester.test_qubits(is_p11)
        assert len(results) == len(device.qubits)
        for q in device.qubits:
            assert results[q]
