# Copyright 2023 Google
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


from recirq.seniority_zero.circuits_expt.gate_sets import SeniorityZeroGateSet

def test_make_gate_set():
    gs = SeniorityZeroGateSet(
        cmnot=None,
        cnot=None,
        gsgate=None,
        rzz=None,
        swap_diag=None,
        swap=None,
        xxyy_diag=None,
        xxzz_diag=None,
        yyzz_diag=None,
    )
