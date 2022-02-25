import os
import pytest
import tempfile
import numpy as np
from recirq.qml_lfe import learn_states_c


def _predict_exp(data, paulistring):
    """Compute expectation values of paulistring given bitstring data."""
    expectation_value = 0
    for a in data:
        val = 1
        for i, pauli in enumerate(paulistring):
            idx = a[i]
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


def test_shadow_doesnt_seperate(tmpdir):
    learn_states_c.run_and_save(
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

    # No signal under changing bases.
    assert np.mean(exp_predictions) <= 0.5
