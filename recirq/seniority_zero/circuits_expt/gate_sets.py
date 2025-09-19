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
from dataclasses import dataclass
from cirq import Gate

from recirq.seniority_zero.circuits_expt.gates import (
    cmnot_from_cz,
    cmnot_from_sqrt_swap,
    cnot_from_cz,
    cnot_from_sqrt_swap,
    gsgate_from_cz,
    gsgate_from_sqrt_swap,
    rzz_from_cz,
    rzz_from_sqrt_swap,
    swap_diagonalization_gate_from_cz,
    swap_diagonalization_gate_from_sqrtswap,
    swap_from_cz,
    swap_from_sqrt_swap,
    xxyy_diag_from_cz,
    xxyy_diag_from_sqrt_swap,
    xxzz_diag_from_cz,
    xxzz_diag_from_sqrt_swap,
    yyzz_diag_from_cz,
    yyzz_diag_from_sqrt_swap,
)


@dataclass
class SeniorityZeroGateSet:
    cmnot: Gate
    cnot: Gate
    gsgate: Gate
    rzz: Gate
    swap_diag: Gate 
    swap: Gate
    xxyy_diag: Gate
    xxzz_diag: Gate
    yyzz_diag: Gate


SQRT_ISWAP_GATESET = SeniorityZeroGateSet(
    cmnot = cmnot_from_sqrt_swap,
    cnot = cnot_from_sqrt_swap,
    gsgate = gsgate_from_sqrt_swap,
    rzz = rzz_from_sqrt_swap,
    swap_diag = swap_diagonalization_gate_from_sqrtswap,
    swap = swap_from_sqrt_swap,
    xxyy_diag = xxyy_diag_from_sqrt_swap,
    yyzz_diag = yyzz_diag_from_sqrt_swap,
    xxzz_diag = xxzz_diag_from_sqrt_swap
)


CZ_GATESET = SeniorityZeroGateSet(
    cmnot = cmnot_from_cz,
    cnot = cnot_from_cz,
    gsgate = gsgate_from_cz,
    rzz = rzz_from_cz,
    swap_diag = swap_diagonalization_gate_from_cz,
    swap = swap_from_cz,
    xxyy_diag = xxyy_diag_from_cz,
    yyzz_diag = yyzz_diag_from_cz,
    xxzz_diag = xxzz_diag_from_cz
)

