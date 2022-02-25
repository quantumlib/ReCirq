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
import os
import numpy as np
from recirq.qml_lfe import learn_states_q


def _predict_exp(data, paulistring):
    """Compute expectation values of paulistring given bitstring data."""
    expectation_value = 0
    for a in data:
        val = 1
        for i, pauli in enumerate(paulistring):
            idx = a[2 * i] * 2 + a[2 * i + 1]
            if pauli == "I":
                continue
            elif pauli == "X":
                ls = [1, 1, -1, -1]
            elif pauli == "Y":
                ls = [-1, 1, 1, -1]
            elif pauli == "Z":
                ls = [1, -1, 1, -1]
            val *= ls[idx]
        expectation_value += val / len(data)
    return expectation_value


def test_twocopy_seperates(tmpdir):
    learn_states_q.run_and_save(
        n=5, n_paulis=10, n_sweeps=250, n_shots=250, save_dir=tmpdir, use_engine=False
    )
    pauli_files = [
        f
        for f in os.listdir(tmpdir)
        if (os.path.isfile(os.path.join(tmpdir, f)) and "basis" not in f)
    ]
    exp_predictions = []
    for fname in pauli_files:
        t = np.load(os.path.join(tmpdir, fname))
        pauli = fname.split("-")[-1][:-4]
        exp_predictions.append(_predict_exp(t, pauli))

    # Strong signal on matching paulis.
    assert np.mean(exp_predictions) >= 0.5
