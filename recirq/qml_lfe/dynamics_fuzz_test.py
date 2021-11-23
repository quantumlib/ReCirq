import os
import pytest
import tempfile
import numpy as np


def test_shadows_dont_seperate(tmpdir):
    pyfile = os.path.join(os.path.dirname(__file__), "learn_dynamics_C.py")
    os.system(
        f"python3 {pyfile} --n=6 --depth=5 --n_data=20 --batch_size=20 --save_dir={tmpdir}"
    )

    tsym_datapoints = []
    scramble_datapoints = []
    for i in range(20):
        fname = f"1D-scramble-C-size-6-depth-5-type-1-batch-0-number-{i}.npy"
        t = np.load(os.path.join(tmpdir, fname))
        tsym_datapoints.append([np.mean(t, axis=0), np.std(t, axis=0)])
        fname = f"1D-scramble-C-size-6-depth-5-type-0-batch-0-number-{i}.npy"
        t = np.load(os.path.join(tmpdir, fname))
        scramble_datapoints.append([np.mean(t, axis=0), np.std(t, axis=0)])

    scramble_bitwise_stats = np.mean(scramble_datapoints, axis=0)
    tsym_bitwise_stats = np.mean(tsym_datapoints, axis=0)
    expected_diff = np.zeros_like(tsym_bitwise_stats)
    # Should see no meanginful difference between measurement stats
    # when using shadows.
    np.testing.assert_allclose(
        scramble_bitwise_stats - tsym_bitwise_stats, expected_diff, atol=0.15
    )
