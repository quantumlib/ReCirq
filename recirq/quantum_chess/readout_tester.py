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
"""
  Simple sanity test to test readout for device qubits.
  This will test P11 and P00 for each qubit on the device.

  See https://www.twitch.tv/anna_chess/video/824369168
  at 1h:17m to see how this can affect live demos.
"""
from typing import Dict

import cirq


class ReadoutTester:
    """Simple sanity test of readout values.

    Initialize with a sampler and device, then use test_qubits
    to test P11 and P00 readout errors just to make sure qubits
    are correctly working.

    Repetitions and the threshold for being correct can also be
    adjusted.
    """

    def __init__(
        self,
        sampler: cirq.Sampler = cirq.Simulator(),
        device: cirq.Device = None,
        repetitions: int = 10000,
        threshold: float = 0.9,
    ):
        self.sampler = sampler
        self.device = device
        self.qubits = device.qubits
        self.repetitions = repetitions
        self.threshold = threshold

    def test_qubits(self, p11=True) -> Dict[cirq.Qid, bool]:
        """Tests qubits to make sure readout is sane.

        With True, P11, Prepare |1>, measure |1> is tested.
        With False, P00, Prepare |0>, measure |0> is tested.

        Returns a dict of qubit to boolean, where True means passed,
        and False means failed.
        """
        qubit_ok = {}
        circuit = cirq.Circuit()
        if p11:
            circuit.append(cirq.X(q) for q in self.qubits)
        circuit.append(cirq.measure(q, key=str(q)) for q in self.qubits)
        results = self.sampler.run(circuit, repetitions=self.repetitions)

        for q in self.qubits:
            count = results.measurements[str(q)].sum()
            if not p11:
                count = self.repetitions - count
            prob = count / self.repetitions
            is_ok = "OK" if prob > self.threshold else "PROBLEM!"
            qubit_ok[q] = True if prob > self.threshold else False
            print(f"Qubit {q}: {count} correct, {prob} {is_ok}")

        return qubit_ok
