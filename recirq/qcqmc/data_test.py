# Copyright 2024 Google
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

import h5py
import pytest

from recirq.qcqmc.data import DEFAULT_BASE_DATA_DIR, get_integrals_path


def test_default_base_data_dir():
    assert ".." not in str(DEFAULT_BASE_DATA_DIR), "should be resolved path"
    assert DEFAULT_BASE_DATA_DIR.exists()
    assert DEFAULT_BASE_DATA_DIR.is_dir()
    assert DEFAULT_BASE_DATA_DIR.name == "data"


@pytest.mark.parametrize("name", ["fh_sto3g", "n2_ccpvtz"])
def test_get_integrals_path(name):
    ipath = get_integrals_path(name)
    assert ipath.exists()
    with h5py.File(ipath) as ifile:
        assert {"ecore", "efci", "h1", "h2"} <= set(ifile.keys())
