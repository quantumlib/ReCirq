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
import cirq

import recirq.quantum_chess.bit_utils as u


def test_nth_bit_of():
    assert u.nth_bit_of(0, 7)
    assert u.nth_bit_of(1, 7)
    assert u.nth_bit_of(2, 7)
    assert not u.nth_bit_of(3, 7)
    assert not u.nth_bit_of(0, 6)
    assert u.nth_bit_of(1, 6)
    assert u.nth_bit_of(2, 6)
    assert not u.nth_bit_of(3, 6)


def test_set_nth_bit():
    assert u.set_nth_bit(0, 6, True) == 7
    assert u.set_nth_bit(0, 6, False) == 6
    assert u.set_nth_bit(2, 7, True) == 7
    assert u.set_nth_bit(2, 7, False) == 3


def test_bit_to_qubit():
    assert u.bit_to_qubit(2) == cirq.NamedQubit("c1")
    assert u.bit_to_qubit(63) == cirq.NamedQubit("h8")


def test_num_ones():
    assert u.num_ones(int("1000101", 2)) == 3
    assert u.num_ones(int("0000000", 2)) == 0
    assert u.num_ones(int("0111111", 2)) == 6


def test_bit_ones():
    assert u.bit_ones(int("1000101", 2)) == [0, 2, 6]
    assert u.bit_ones(int("0000000", 2)) == []
    assert u.bit_ones(int("0111111", 2)) == [0, 1, 2, 3, 4, 5]


def test_qubit_to_bit():
    assert u.qubit_to_bit(cirq.NamedQubit("c1")) == 2
    assert u.qubit_to_bit(cirq.NamedQubit("h8")) == 63


def test_square_to_bit():
    assert u.square_to_bit("a1") == 0
    assert u.square_to_bit("a2") == 8
    assert u.square_to_bit("b2") == 9
    assert u.square_to_bit("b1") == 1
    assert u.square_to_bit("h8") == 63


def test_squares_to_bitboard():
    assert u.squares_to_bitboard(["a1", "a2"]) == 257
    assert u.squares_to_bitboard(["a1", "c1"]) == 5


def test_bitboard_to_squares():
    assert u.bitboard_to_squares(257) == ["a1", "a2"]
    assert u.bitboard_to_squares(5) == ["a1", "c1"]
