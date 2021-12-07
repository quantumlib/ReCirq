import os
import pytest
import tempfile
import numpy as np
from . import learn_states_c


def _predict_exp(data, pauli_list, two_copy):
    ave = 0
    for a in data:
        val = 1
        for i, pauli in enumerate(pauli_list):
            idx = a[i]
            if two_copy:
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
        ave += val / len(data)
    return ave


def test_shadow_doesnt_seperate(tmpdir):
    learn_states_c.run_and_save(
        n=5, n_paulis=10, batch_size=250, n_shots=250, save_dir=tmpdir, use_engine=False
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
        exp_predictions.append(_predict_exp(t, pauli, False))

    # No signal under changing bases.
    assert np.mean(exp_predictions) <= 0.5
