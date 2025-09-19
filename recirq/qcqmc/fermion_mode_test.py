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
import cirq
import pytest

from recirq.qcqmc.fermion_mode import FermionicMode


def test_fermionic_mode():
    fm = FermionicMode(5, "a")
    fm2 = cirq.read_json(json_text=cirq.to_json(fm))
    assert fm == fm2

    with pytest.raises(ValueError, match="spin.*"):
        _ = FermionicMode(10, "c")
