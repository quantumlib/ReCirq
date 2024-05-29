import h5py
import pytest

from recirq.qcqmc.data import DEFAULT_BASE_DATA_DIR, get_integrals_path


def test_default_base_data_dir():
    assert ".." not in str(DEFAULT_BASE_DATA_DIR), "should be resolved path"
    print(DEFAULT_BASE_DATA_DIR)
    assert DEFAULT_BASE_DATA_DIR.exists()
    assert DEFAULT_BASE_DATA_DIR.is_dir()
    assert DEFAULT_BASE_DATA_DIR.name == "data"


@pytest.mark.parametrize("name", ["fh_sto3g", "n2_ccpvtz"])
def test_get_integrals_path(name):
    ipath = get_integrals_path(name)
    assert ipath.exists()
    with h5py.File(ipath) as ifile:
        assert {"ecore", "efci", "h1", "h2"} <= set(ifile.keys())
