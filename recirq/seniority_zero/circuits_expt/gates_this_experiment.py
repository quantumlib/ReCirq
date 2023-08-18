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

"""File to import all gates from to make changing gate sets easier"""

gate_type = 'cz'

# pylint: disable=unused-import
if gate_type == 'sqrt_swap':
    from recirq.seniority_zero.circuits_expt.gates import cmnot_from_sqrt_swap as cmnot
    from recirq.seniority_zero.circuits_expt.gates import cnot_from_sqrt_swap as cnot
    from recirq.seniority_zero.circuits_expt.gates import gsgate_from_sqrt_swap as gsgate
    from recirq.seniority_zero.circuits_expt.gates import rzz_from_sqrt_swap as rzz
    from recirq.seniority_zero.circuits_expt.gates import (
        swap_diagonalization_gate_from_sqrtswap as swap_diag,
    )
    from recirq.seniority_zero.circuits_expt.gates import swap_from_sqrt_swap as swap
    from recirq.seniority_zero.circuits_expt.gates import xxyy_diag_from_sqrt_swap as xxyy_diag
    from recirq.seniority_zero.circuits_expt.gates import xxzz_diag_from_sqrt_swap as xxzz_diag
    from recirq.seniority_zero.circuits_expt.gates import yyzz_diag_from_sqrt_swap as yyzz_diag

elif gate_type == 'cz':
    from recirq.seniority_zero.circuits_expt.gates import cmnot_from_cz as cmnot
    from recirq.seniority_zero.circuits_expt.gates import cnot_from_cz as cnot
    from recirq.seniority_zero.circuits_expt.gates import gsgate_from_cz as gsgate
    from recirq.seniority_zero.circuits_expt.gates import rzz_from_cz as rzz
    from recirq.seniority_zero.circuits_expt.gates import (
        swap_diagonalization_gate_from_cz as swap_diag,
    )
    from recirq.seniority_zero.circuits_expt.gates import swap_from_cz as swap
    from recirq.seniority_zero.circuits_expt.gates import xxyy_diag_from_cz as xxyy_diag
    from recirq.seniority_zero.circuits_expt.gates import xxzz_diag_from_cz as xxzz_diag
    from recirq.seniority_zero.circuits_expt.gates import yyzz_diag_from_cz as yyzz_diag

else:
    raise TypeError('I dont understand the gate type given')
