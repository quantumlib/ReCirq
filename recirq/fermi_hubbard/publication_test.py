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

import os
from recirq.fermi_hubbard.publication import fetch_publication_data


def test_fetch_publication_data():
    base_dir = "fermi_hubbard_data"
    fetch_publication_data(base_dir=base_dir, exclude=["trapping_3u3d"])

    for path in ("gaussians_1u1d_nofloquet", "gaussians_1u1d", "trapping_2u2d"):
        assert os.path.exists(base_dir + os.path.sep + path)

    fetch_publication_data(base_dir=base_dir)
    assert os.path.exists(base_dir + os.path.sep + "trapping_3u3d")
