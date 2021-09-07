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
"""Quantum Circuits for common quantum chess moves."""
from typing import List

import cirq


def normal_move(s: cirq.Qid, t: cirq.Qid):
    """A normal move in quantum chess.

    This function takes two qubits and returns a generator
    that performs a normal move that moves a piece from the
    source square to the target square.

    Args:
      s: source qubit (square where piece starts)
      t: target qubit (square to move piece to)
    """
    yield cirq.ISWAP(s, t) ** 0.5
    yield cirq.ISWAP(s, t) ** 0.5


def split_move(s: cirq.Qid, t1: cirq.Qid, t2: cirq.Qid):
    """A Split move in quantum chess.

    This function takes three qubits and returns a generator
    that performs a split move from the source qubit to the
    two target qubits.

    Args:
      s: source qubit (square where piece starts)
      t1: target qubit (first square to move piece to)
      t2: target qubit (second square to move piece to)
    """
    yield cirq.ISWAP(s, t1) ** 0.5
    yield cirq.ISWAP(s, t2) ** 0.5
    yield cirq.ISWAP(s, t2) ** 0.5


def merge_move(s1: cirq.Qid, s2: cirq.Qid, t: cirq.Qid):
    """A Merge move in quantum chess.

    This function takes three qubits and returns a generator
    that performs a merge move from two source qubits to the
    target qubit.

    Args:
      s1: source qubit (first square where a piece starts from)
      s2: source qubit (second square where a piece starts from)
      t: target qubit (square to move a piece to)
    """
    yield cirq.ISWAP(s1, t) ** -0.5
    yield cirq.ISWAP(s1, t) ** -0.5
    yield cirq.ISWAP(s2, t) ** -0.5


def slide_move(
    s: cirq.Qid, t: cirq.Qid, path: List[cirq.Qid], ancilla: cirq.Qid = None
):
    """A Slide move in quantum chess.

    This function takes three qubits and returns a generator
    that performs a slide move.  This will move a piece
    from the source to the target controlled by the path qubit.
    The move will only occur if the path qubit is turned on.

    Args:
      s: source qubit (square where piece starts)
      t: target qubit (square to move piece to)
      path: qubit blocked the path
      ancilla: ancilla needed if len(path) > 1
    """
    if len(path) == 0:
        return normal_move(s, t)
    if len(path) == 1:
        p = path[0]
        yield cirq.X(p)
        yield cirq.ISWAP(s, t).controlled_by(p)
        yield cirq.X(p)
        return
    if ancilla is None:
        raise ValueError("Must specify ancilla for path greater than 1.")
    for p in path:
        yield cirq.X(p)
    yield cirq.X(ancilla).controlled_by(*path)
    yield cirq.ISWAP(s, t).controlled_by(ancilla)
    # Ancilla must be cleaned up, lest it affect the phases of the board state
    yield cirq.X(ancilla).controlled_by(*path)
    for p in path:
        yield cirq.X(p)


def place_piece(s: cirq.Qid):
    """Place a piece on the board.

    This is not actually a move.  However, since qubits in Cirq
    default to starting in the |0> state, we will need to activate
    the qubit by applying an X gate to initialize the position.
    """
    yield cirq.X(s)


def controlled_operation(gate, qubits, path_qubits, anti_qubits):
    """Apply gate on qubits, controlled by path_qubits and
    anti-controlled by anti_qubits.
    """
    for p in anti_qubits:
        yield cirq.X(p)
    yield gate.on(*qubits).controlled_by(*path_qubits, *anti_qubits)
    for p in anti_qubits:
        yield cirq.X(p)


def queenside_castle(squbit, rook_squbit, tqubit, rook_tqubit, b_qubit):
    """Performs queenside castle, anti-controlled by b_qubit."""
    yield cirq.X(b_qubit)
    yield cirq.ISWAP(rook_squbit, rook_tqubit).controlled_by(b_qubit)
    yield cirq.ISWAP(squbit, tqubit).controlled_by(b_qubit)
    yield cirq.X(b_qubit)


def split_slide(squbit, tqubit, tqubit2, path1, path2, ancilla):
    """Performs a split slide from squbit to two target qubits.

    2 path qubits are needed as controls.  An additional ancilla
    is needed to transform the double controlled iSWAPs into
    single controlled iSWAPs.
    """
    # Set up the ancilla which will act as a control
    # This essentially takes ISWAP.controlled_by(path1, path2)
    # and turns it into ISWAP.controlled_by(ancilla)
    yield cirq.X(ancilla).controlled_by(path2, path1)

    yield (cirq.ISWAP(squbit, tqubit) ** 0.5).controlled_by(ancilla)
    yield (cirq.ISWAP(squbit, tqubit2)).controlled_by(ancilla)

    # Now switch to anti-control of path2
    yield cirq.X(ancilla).controlled_by(path1)

    yield cirq.ISWAP(squbit, tqubit).controlled_by(ancilla)

    # Switch to control of path2 but anti-control of path1
    yield cirq.X(ancilla).controlled_by(path1)

    # In order to prevent path2 from needing connectivity to
    # the ancilla qubit, swap path1 and path2.
    # Then do a CNOT on ancilla with the swapped "path2"
    yield cirq.SWAP(path1, path2)
    yield cirq.X(ancilla).controlled_by(path1)

    yield cirq.ISWAP(squbit, tqubit2).controlled_by(ancilla)

    # Zero out ancillas
    yield cirq.X(ancilla).controlled_by(path1)
    yield cirq.SWAP(path1, path2)
    yield cirq.X(ancilla).controlled_by(path2, path1)


def merge_slide(squbit, tqubit, squbit2, path1, path2, ancilla):
    yield cirq.X(ancilla).controlled_by(path1, path2)

    yield (cirq.ISWAP(squbit, tqubit) ** -1.0).controlled_by(ancilla)
    yield (cirq.ISWAP(squbit2, tqubit) ** -0.5).controlled_by(ancilla)

    # Now switch to anti-control of path1
    yield cirq.X(ancilla).controlled_by(path2)
    yield (cirq.ISWAP(squbit2, tqubit) ** -1).controlled_by(ancilla)
    # Switch to control of path1 but anti-control of path2
    yield cirq.X(ancilla).controlled_by(path2)

    # In order to prevent path2 from needing connectivity to
    # the ancilla qubit, swap path1 and path2.
    # Then do a CNOT on ancilla with the swapped "path2"
    yield cirq.SWAP(path2, path1)
    yield cirq.X(ancilla).controlled_by(path2)
    yield (cirq.ISWAP(squbit, tqubit) ** -1).controlled_by(ancilla)

    # Zero out ancillas
    yield cirq.X(ancilla).controlled_by(path2)
    yield cirq.SWAP(path2, path1)
    yield cirq.X(ancilla).controlled_by(path1, path2)


def en_passant(squbit, tqubit, epqubit, path, c):
    yield cirq.X(path).controlled_by(squbit, epqubit)
    yield cirq.ISWAP(epqubit, c).controlled_by(path)
    yield cirq.ISWAP(squbit, tqubit).controlled_by(path)


def capture_ep(squbit, tqubit, epqubit, path, c, c2):
    yield cirq.CNOT(epqubit, path)
    yield cirq.CNOT(tqubit, path)
    yield cirq.ISWAP(epqubit, c).controlled_by(path)
    yield cirq.ISWAP(tqubit, c2).controlled_by(path)
    yield cirq.ISWAP(squbit, tqubit).controlled_by(path)
