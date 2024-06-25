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
"""Specification of a fermion mode."""
import attrs


@attrs.frozen
class FermionicMode:
    """A specification of a fermionic mode.

    Args:
        orb_ind: The spatial orbital index.
        spin: The spin state of the fermion mode (up or down (alpha or beta)).
            Must be either 'a' or 'b'.
    """

    orb_ind: int
    spin: str

    def __attrs_post_init__(self):
        if self.spin not in ["a", "b"]:
            raise ValueError(
                "Spin must be either a or b for spin alpha(up) or beta(down) respectively."
            )

    @classmethod
    def _json_namespace_(cls):
        return "recirq.qcqmc"

    def _json_dict_(self):
        return attrs.asdict(self)

    @property
    def openfermion_standard_index(self) -> int:
        return 2 * self.orb_ind + (0 if self.spin == "a" else 1)
