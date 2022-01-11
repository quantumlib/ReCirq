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

ROWS = 3
COLS = 2
ORIGIN = cirq.GridQubit(0, 0)
ROW_VECTOR = (1, 1)


def q(i: int, j: int) -> cirq.GridQubit:
    return cirq.GridQubit(i, j)


@pytest.mark.parametrize(
    "displacement", [(0, 0), (10, 10), (-4, 5), (-1, -12), (3, -1)]
)
def test_q_displaced(displacement: Tuple[int, int]):
    qubit = cirq.GridQubit(-2, 12)
    displaced_qubit = tcr.q_displaced(qubit, np.array(displacement))
    expected = qubit + displacement
    assert displaced_qubit == expected


def test_captain_qubits():
    code = tcr.ToricCodeRectangle(ORIGIN, ROW_VECTOR, ROWS, COLS)
    captains = code.captain_qubits
    assert len(captains) == ROWS * COLS
    assert ORIGIN in captains
    assert captains == {q(0, 0), q(-1, 1), q(1, 1), q(0, 2), q(2, 2), q(1, 3)}


def test_qubits():
    code = tcr.ToricCodeRectangle(ORIGIN, ROW_VECTOR, ROWS, COLS)
    qubits = code.qubits
    assert len(qubits) == 2 * ROWS * COLS + ROWS + COLS
    assert ORIGIN in qubits
    assert qubits == {
        q(-1, 1),
        q(-1, 2),
        q(0, 0),
        q(0, 1),
        q(0, 2),
        q(0, 3),
        q(1, 0),
        q(1, 1),
        q(1, 2),
        q(1, 3),
        q(1, 4),
        q(2, 1),
        q(2, 2),
        q(2, 3),
        q(2, 4),
        q(3, 2),
        q(3, 3),
    }


def test_qubits_reading_order():
    code = tcr.ToricCodeRectangle(ORIGIN, ROW_VECTOR, ROWS, COLS)
    assert code.qubits_reading_order == (
        q(0, 0),  # Captains
        q(-1, 1),
        q(1, 0),  # Between
        q(0, 1),
        q(-1, 2),
        q(1, 1),  # Captains
        q(0, 2),
        q(2, 1),  # Between
        q(1, 2),
        q(0, 3),
        q(2, 2),  # Captains
        q(1, 3),
        q(3, 2),  # Between
        q(2, 3),
        q(1, 4),
        q(3, 3),  # "False" captains
        q(2, 4),
    )


def test_x_plaquette_indices():
    code = tcr.ToricCodeRectangle(ORIGIN, ROW_VECTOR, ROWS, COLS)
    indices = {x for x in code.x_plaquette_indices()}
    assert len(indices) == ROWS * COLS
    for row, col in indices:
        assert 0 <= row < ROWS
        assert 0 <= col < COLS


def test_z_plaquette_indices():
    code = tcr.ToricCodeRectangle(ORIGIN, ROW_VECTOR, ROWS, COLS)
    indices = {x for x in code.z_plaquette_indices()}
    assert len(indices) == (ROWS + 1) * (COLS + 1)
    for row, col in indices:
        assert 0 <= row < ROWS + 1
        assert 0 <= col < COLS + 1


def test_is_corner():
    code = tcr.ToricCodeRectangle(ORIGIN, ROW_VECTOR, ROWS, COLS)
    expected_corners = {
        (0, 0),
        (0, COLS),
        (ROWS, 0),
        (ROWS, COLS),
    }
    corners = set(rc for rc in code.z_plaquette_indices() if code.is_corner(*rc))
    assert corners == expected_corners


def test_is_edge():
    code = tcr.ToricCodeRectangle(ORIGIN, ROW_VECTOR, ROWS, COLS)
    edges = set(rc for rc in code.z_plaquette_indices() if code.is_edge(*rc))
    assert len(edges) == 2 * (code.rows + code.cols) - 4
    assert (0, 0) not in edges
    assert (1, 0) in edges


@pytest.mark.parametrize("plaquette", [(0, 0), (10, 10), (-4, 5), (-1, -12), (3, -1)])
def test_x_plaquette_to_captain(plaquette: Tuple[int, int]):
    code = tcr.ToricCodeRectangle(ORIGIN, ROW_VECTOR, ROWS, COLS)
    captain = code.x_plaquette_to_captain(*plaquette)
    assert plaquette == code.captain_to_x_plaquette(captain)


def test_x_plaquette_to_qubits():
    code = tcr.ToricCodeRectangle(ORIGIN, ROW_VECTOR, ROWS, COLS)
    row = 1
    col = 1
    qubits = code.x_plaquette_to_qubits(row, col)
    assert len(qubits) == 4
    assert code.x_plaquette_to_captain(row, col) in qubits
    assert qubits == {q(0, 2), q(0, 3), q(1, 2), q(1, 3)}


def test_z_plaquette_to_qubits():
    code = tcr.ToricCodeRectangle(ORIGIN, ROW_VECTOR, ROWS, COLS)
    # Central plaquette, four qubits
    row = 1
    col = 1
    qubits = code.z_plaquette_to_qubits(row, col)
    assert len(qubits) == 4
    assert code.x_plaquette_to_captain(row, col) in qubits
    assert qubits == {q(0, 1), q(0, 2), q(1, 1), q(1, 2)}

    # Left side, three qubits
    row = 1
    col = 0
    qubits = code.z_plaquette_to_qubits(row, col)
    assert len(qubits) == 3
    assert code.x_plaquette_to_captain(row, col) in qubits
    assert qubits == {q(1, 0), q(1, 1), q(2, 1)}

    # Right side, three qubits, x captain not present
    row = 2
    col = 2
    qubits = code.z_plaquette_to_qubits(row, col)
    assert len(qubits) == 3
    assert code.x_plaquette_to_captain(row, col) not in qubits
    assert qubits == {q(0, 3), q(1, 3), q(1, 4)}

    # Upper right corner, two qubits, x captain not present
    row = 0
    col = 2
    qubits = code.z_plaquette_to_qubits(row, col)
    assert len(qubits) == 2
    assert code.x_plaquette_to_captain(row, col) not in qubits
    assert qubits == {q(-1, 1), q(-1, 2)}

    # Lower left corner, two qubits
    row = 3
    col = 0
    qubits = code.z_plaquette_to_qubits(row, col)
    assert len(qubits) == 2
    assert code.x_plaquette_to_captain(row, col) in qubits
    assert qubits == {q(3, 2), q(3, 3)}


def test_x_plaquette_to_qubit_idxs():
    code = tcr.ToricCodeRectangle(ORIGIN, ROW_VECTOR, ROWS, COLS)
    row = 1
    col = 1
    idxs = code.x_plaquette_to_qubit_idxs(row, col)
    assert len(idxs) == 4
    assert idxs == {4, 5, 8, 9}


def test_z_plaquette_to_qubit_idxs():
    code = tcr.ToricCodeRectangle(ORIGIN, ROW_VECTOR, ROWS, COLS)
    row = 1
    col = 1
    idxs = code.z_plaquette_to_qubit_idxs(row, col)
    assert len(idxs) == 4
    assert idxs == {3, 4, 7, 8}


@pytest.mark.parametrize("captain", [q(0, 0), q(-1, 1), q(1, 1), q(0, 2)])
def test_captain_to_x_plaquette(captain: cirq.GridQubit):
    code = tcr.ToricCodeRectangle(ORIGIN, ROW_VECTOR, ROWS, COLS)
    plaquette = code.captain_to_x_plaquette(captain)
    assert captain == code.x_plaquette_to_captain(*plaquette)


def test_invalid_captain_to_x_plaquette():
    code = tcr.ToricCodeRectangle(ORIGIN, ROW_VECTOR, ROWS, COLS)
    non_captain = ORIGIN + (0, 1)
    with pytest.raises(ValueError):
        _ = code.captain_to_x_plaquette(non_captain)


def test_q_directions():
    code = tcr.ToricCodeRectangle(ORIGIN, ROW_VECTOR, ROWS, COLS)
    # Left side
    captain = q(1, 1)
    assert code.q_lower_left(captain) == q(2, 1)
    assert code.q_lower_right(captain) == q(1, 2)
    assert code.q_down(captain) == q(2, 2)
    assert code.q_upper_left(captain) == q(1, 0)
    assert code.q_across_left(captain) == q(2, 0)
    assert code.q_lower_outside(captain) == code.q_lower_left(captain)
    assert code.q_lower_inside(captain) == code.q_lower_right(captain)

    # Right side
    captain = q(0, 2)
    assert code.q_lower_left(captain) == q(1, 2)
    assert code.q_lower_right(captain) == q(0, 3)
    assert code.q_down(captain) == q(1, 3)
    assert code.q_upper_left(captain) == q(0, 1)
    assert code.q_across_left(captain) == q(1, 1)
    assert code.q_lower_outside(captain) == code.q_lower_right(captain)
    assert code.q_lower_inside(captain) == code.q_lower_left(captain)
