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

# coverage: ignore
import pytest
from recirq.hfvqe.molecular_data.molecular_data_construction import (
    h_n_linear_molecule)


def test_negative_n_hydrogen_chain():
    # coverage: ignore
    with pytest.raises(ValueError):
        h_n_linear_molecule(1.3, 0)
