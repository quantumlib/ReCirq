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
"""
Utilities for converting to and from bit boards.
"""
from typing import List

import cirq

import recirq.quantum_chess.move as move


def nth_bit_of(n: int, bit_board: int) -> bool:
    """Returns the n-th bit of a 64-bit chess bitstring"""
    return (bit_board >> n) % 2 == 1


def set_nth_bit(n: int, bit_board: int, val: bool) -> int:
    """Sets the nth bit of the bitstring to a specific value."""
    return bit_board - (nth_bit_of(n, bit_board) << n) + (int(val) << n)


def bit_to_qubit(n: int) -> cirq.Qid:
    """Turns a number into a cirq Qubit."""
    return cirq.NamedQubit(bit_to_square(n))


def num_ones(n: int) -> int:
    """Number of ones in the binary representation of n."""
    count = 0
    while n > 0:
        if n % 2 == 1:
            count += 1
        n = n // 2
    return count


def bit_ones(n: int) -> List[int]:
    """Indices of ones in the binary representation of n."""
    indices = []
    index = 0
    while n > 0:
        if n % 2 == 1:
            indices.append(index)
        n >>= 1
        index += 1
    return indices


def qubit_to_bit(q: cirq.LineQubit) -> int:
    """Retrieves the number from a cirq Qubit's name.

    Does not work for ancilla Qubits.
    """
    return square_to_bit(q.name)


def xy_to_bit(x: int, y: int) -> int:
    """Transform x/y coordinates into a bitboard bit number."""
    return y * 8 + x


def square_to_bit(sq: str) -> int:
    """Transform algebraic square notation into a bitboard bit number."""
    return move.y_of(sq) * 8 + move.x_of(sq)


def bit_to_square(bit: int) -> str:
    """Transform a bitboard bit number into algebraic square notation."""
    return move.to_square(bit % 8, bit // 8)


def squares_to_bitboard(squares: List[str]) -> int:
    """Transform a list of algebraic squares into a 64-bit board bitstring."""
    bitboard = 0
    for sq in squares:
        bitboard += 1 << square_to_bit(sq)
    return bitboard


def bitboard_to_squares(bitboard: int) -> List[str]:
    """Transform a 64-bit bitstring into a list of algebraic squares."""
    squares = []
    for n in range(64):
        if nth_bit_of(n, bitboard):
            squares.append(bit_to_square(n))
    return squares
