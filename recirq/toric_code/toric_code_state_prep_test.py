# Copyright 2022 Google
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

from typing import Tuple

import cirq
import numpy as np
import pytest

from . import toric_code_rectangle as tcr
from . import toric_code_state_prep as tcsp

ROWS = 3
COLS = 2
ORIGIN = cirq.GridQubit(0, 0)
ROW_VECTOR = (1, 1)


def test_middle_out_column_groups_odd():
    code = tcr.ToricCodeRectangle(ORIGIN, ROW_VECTOR, ROWS, 7)
    expected_groups = [{3}, {2, 4}, {1, 5}, {0, 6}]
    assert list(tcsp._middle_out_column_groups(code)) == expected_groups


def test_middle_out_column_groups_even():
    code = tcr.ToricCodeRectangle(ORIGIN, ROW_VECTOR, ROWS, 8)
    expected_groups = [{3, 4}, {2, 5}, {1, 6}, {0, 7}]
    assert list(tcsp._middle_out_column_groups(code)) == expected_groups


@pytest.mark.parametrize("cols", range(1, 10))
def test_cnot_circuit_depth(cols: int):
    code = tcr.ToricCodeRectangle(ORIGIN, ROW_VECTOR, ROWS, cols)
    circuit = tcsp.toric_code_cnot_circuit(code)
    number_of_column_groups = (cols + 1) // 2
    expected_cnot_depth = 2 * number_of_column_groups + 1
    expected_hadamard_depth = 1
    assert len(circuit) == expected_cnot_depth + expected_hadamard_depth


def test_cnot_circuit_hadamards():
    code = tcr.ToricCodeRectangle(ORIGIN, ROW_VECTOR, ROWS, COLS)
    circuit_z = tcsp.toric_code_cnot_circuit(code, x_basis=False)
    circuit_x = tcsp.toric_code_cnot_circuit(code, x_basis=True)
    assert circuit_z == circuit_x[:-1]
    assert circuit_x[-1] == cirq.Moment(cirq.H(q) for q in code.qubits)


def test_cnot_circuit_7_qubit_example():
    """Check that we get the toric code ground state in a 7-qubit case.

    Expect equal amplitude of 0.5 on these bitstrings:

      0 0     1 1     0 0     1 1
    0 0 0   0 1 1   1 1 0   1 0 1
    0 0   , 0 0   , 1 1   , 1 1

    These are read off in binary as 0b0000000, 0b1101100, 0b0011011, 0b1110111.
    """
    small_code = tcr.ToricCodeRectangle(ORIGIN, ROW_VECTOR, 1, 2)
    circuit = tcsp.toric_code_cnot_circuit(small_code)
    result = cirq.Simulator().simulate(circuit)
    state = result.state_vector()
    nonzero_elements = np.nonzero(state)[0]
    assert set(nonzero_elements) == {0b0000000, 0b1101100, 0b0011011, 0b1110111}
    assert np.allclose(state[nonzero_elements], [0.5] * 4)


def test_cnot_circuit_12_qubit_example():
    """Check that we get the toric code ground state in a 12-qubit case.

    Expect equal amplitude of 0.25 on these bitstrings:

      0 0       0 0       0 0       0 0
    0 0 0 0   0 0 1 1   0 0 0 0   0 0 1 1
    0 0 0 0   0 0 1 1   0 1 1 0   0 1 0 1
      0 0   ,   0 0   ,   1 1   ,   1 1   ,

      1 1       1 1       1 1       1 1
    0 1 1 0   0 1 0 1   0 1 1 0   0 1 0 1
    0 0 0 0   0 0 1 1   0 1 1 0   0 1 0 1
      0 0   ,   0 0   ,   1 1   ,   1 1   ,

      0 0       0 0       0 0       0 0
    1 1 0 0   1 1 1 1   1 1 0 0   1 1 1 1
    1 1 0 0   1 1 1 1   1 0 1 0   1 0 0 1
      0 0   ,   0 0   ,   1 1   ,   1 1   ,

      1 1       1 1       1 1       1 1
    1 0 1 0   1 0 0 1   1 0 1 0   1 0 0 1
    1 1 0 0   1 1 1 1   1 0 1 0   1 0 0 1
      0 0   ,   0 0   ,   1 1   ,   1 1

    These are read off in binary as 0b000000000000, 0b000011001100, 0b000000011011, etc.
    """
    code = tcr.ToricCodeRectangle(ORIGIN, ROW_VECTOR, 2, 2)
    circuit = tcsp.toric_code_cnot_circuit(code)
    result = cirq.Simulator().simulate(circuit)
    state = result.state_vector()
    nonzero_elements = np.nonzero(state)[0]
    assert set(nonzero_elements) == {
        0b000000000000,
        0b000011001100,
        0b000000011011,
        0b000011010111,
        0b110110000000,
        0b110101001100,
        0b110110011011,
        0b110101010111,
        0b001100110000,
        0b001111111100,
        0b001100101011,
        0b001111100111,
        0b111010110000,
        0b111001111100,
        0b111010101011,
        0b111001100111,
    }
    assert np.allclose(state[nonzero_elements], [0.25] * 16)
