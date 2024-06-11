import h5py
import pytest

from recirq.qcqmc.data import get_integrals_path


@pytest.mark.parametrize("name", ["fh_sto3g", "n2_ccpvtz"])
def test_get_integrals_path(name):
    ipath = get_integrals_path(name)
    assert ipath.exists()
    with h5py.File(ipath) as ifile:
        assert {"ecore", "efci", "h1", "h2"} <= set(ifile.keys())
