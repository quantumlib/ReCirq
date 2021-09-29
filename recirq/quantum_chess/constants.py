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

PIECES = {
    0: "?",  # Shown if there is nonzero probability where a piece was not expected
    1: "P",
    -1: "p",
    2: "N",
    -2: "n",
    3: "B",
    -3: "b",
    4: "R",
    -4: "r",
    5: "Q",
    -5: "q",
    6: "K",
    -6: "k",
}

REV_PIECES = {
    " ": 0,
    ".": 0,
    "P": 1,
    "p": -1,
    "N": 2,
    "n": -2,
    "B": 3,
    "b": -3,
    "R": 4,
    "r": -4,
    "Q": 5,
    "q": -5,
    "K": 6,
    "k": -6,
}

EMPTY = 0
PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING = 6

a1 = cirq.NamedQubit("a1")
a2 = cirq.NamedQubit("a2")
a3 = cirq.NamedQubit("a3")
a4 = cirq.NamedQubit("a4")
a5 = cirq.NamedQubit("a5")
a6 = cirq.NamedQubit("a6")
a7 = cirq.NamedQubit("a7")
a8 = cirq.NamedQubit("a8")
b1 = cirq.NamedQubit("b1")
b2 = cirq.NamedQubit("b2")
b3 = cirq.NamedQubit("b3")
b4 = cirq.NamedQubit("b4")
b5 = cirq.NamedQubit("b5")
b6 = cirq.NamedQubit("b6")
b7 = cirq.NamedQubit("b7")
b8 = cirq.NamedQubit("b8")
c1 = cirq.NamedQubit("c1")
c2 = cirq.NamedQubit("c2")
c3 = cirq.NamedQubit("c3")
c4 = cirq.NamedQubit("c4")
c5 = cirq.NamedQubit("c5")
c6 = cirq.NamedQubit("c6")
c7 = cirq.NamedQubit("c7")
c8 = cirq.NamedQubit("c8")
d1 = cirq.NamedQubit("d1")
d2 = cirq.NamedQubit("d2")
d3 = cirq.NamedQubit("d3")
d4 = cirq.NamedQubit("d4")
d5 = cirq.NamedQubit("d5")
d6 = cirq.NamedQubit("d6")
d7 = cirq.NamedQubit("d7")
d8 = cirq.NamedQubit("d8")
e1 = cirq.NamedQubit("e1")
e2 = cirq.NamedQubit("e2")
e3 = cirq.NamedQubit("e3")
e4 = cirq.NamedQubit("e4")
e5 = cirq.NamedQubit("e5")
e6 = cirq.NamedQubit("e6")
e7 = cirq.NamedQubit("e7")
e8 = cirq.NamedQubit("e8")
f1 = cirq.NamedQubit("f1")
f2 = cirq.NamedQubit("f2")
f3 = cirq.NamedQubit("f3")
f4 = cirq.NamedQubit("f4")
f5 = cirq.NamedQubit("f5")
f6 = cirq.NamedQubit("f6")
f7 = cirq.NamedQubit("f7")
f8 = cirq.NamedQubit("f8")
g1 = cirq.NamedQubit("g1")
g2 = cirq.NamedQubit("g2")
g3 = cirq.NamedQubit("g3")
g4 = cirq.NamedQubit("g4")
g5 = cirq.NamedQubit("g5")
g6 = cirq.NamedQubit("g6")
g7 = cirq.NamedQubit("g7")
g8 = cirq.NamedQubit("g8")
h1 = cirq.NamedQubit("h1")
h2 = cirq.NamedQubit("h2")
h3 = cirq.NamedQubit("h3")
h4 = cirq.NamedQubit("h4")
h5 = cirq.NamedQubit("h5")
h6 = cirq.NamedQubit("h6")
h7 = cirq.NamedQubit("h7")
h8 = cirq.NamedQubit("h8")
