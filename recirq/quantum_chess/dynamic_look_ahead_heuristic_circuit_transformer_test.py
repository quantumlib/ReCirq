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
import math
import pytest
from collections import deque

import cirq

import recirq.quantum_chess.circuit_transformer as ct

a0 = cirq.NamedQubit('a0')
a1 = cirq.NamedQubit('a1')
a2 = cirq.NamedQubit('a2')
a3 = cirq.NamedQubit('a3')
a4 = cirq.NamedQubit('a4')
a5 = cirq.NamedQubit('a5')
a6 = cirq.NamedQubit('a6')
a7 = cirq.NamedQubit('a7')

logical = {
    a0: [(a2, 0), (a1, 1), (a3, 6)],
    a1: [(a0, 1), (a2, 3)],
    a2: [(a0, 0), (a1, 3), (a3, 5)],
    a3: [(a2, 5), (a0, 6)],
}

logical2 = {
    a0: [(a2, 0), (a1, 1), (a3, 6)],
    a1: [(a0, 1), (a2, 3), (a7, 8)],
    a2: [(a0, 0), (a1, 3), (a3, 5)],
    a3: [(a2, 5), (a0, 6)],
    a4: [(a5, 0), (a6, 7)],
    a5: [(a4, 0)],
    a6: [(a7, 0), (a4, 7)],
    a7: [(a6, 0), (a1, 8)],
}

zero_zero = cirq.GridQubit(0, 0)
zero_one = cirq.GridQubit(0, 1)
zero_two = cirq.GridQubit(0, 2)
zero_three = cirq.GridQubit(0, 3)
zero_four = cirq.GridQubit(0, 4)
zero_five = cirq.GridQubit(0, 5)
zero_six = cirq.GridQubit(0, 6)
zero_seven = cirq.GridQubit(0, 7)
zero_eight = cirq.GridQubit(0, 8)
zero_nine = cirq.GridQubit(0, 9)
zero_ten = cirq.GridQubit(0, 10)
one_zero = cirq.GridQubit(1, 0)
one_one = cirq.GridQubit(1, 1)
one_two = cirq.GridQubit(1, 2)
one_three = cirq.GridQubit(1, 3)
one_four = cirq.GridQubit(1, 4)
one_five = cirq.GridQubit(1, 5)
one_six = cirq.GridQubit(1, 6)
one_seven = cirq.GridQubit(1, 7)
one_eight = cirq.GridQubit(1, 8)
one_nine = cirq.GridQubit(1, 9)
one_ten = cirq.GridQubit(1, 10)

physical = {
    zero_zero: [zero_one, one_zero],
    zero_one: [zero_zero, one_one, zero_two],
    zero_two: [zero_one, one_two, zero_three],
    zero_three: [zero_two, one_three, zero_four],
    zero_four: [zero_three, one_four, zero_five],
    zero_five: [zero_four, one_five, zero_six],
    zero_six: [zero_five, one_six, zero_seven],
    zero_seven: [zero_six, one_seven, zero_eight],
    zero_eight: [zero_seven, one_eight, zero_nine],
    zero_nine: [zero_eight, one_nine, zero_ten],
    zero_ten: [zero_nine, one_ten],
    one_zero: [one_one, zero_zero],
    one_one: [one_zero, zero_one, one_two],
    one_two: [one_one, zero_two, one_three],
    one_three: [one_two, zero_three, one_four],
    one_four: [one_three, zero_four, one_five],
    one_five: [one_four, zero_five, one_six],
    one_six: [one_five, zero_six, one_seven],
    one_seven: [one_six, zero_seven, one_eight],
    one_eight: [one_seven, zero_eight, one_nine],
    one_nine: [one_eight, zero_nine, one_ten],
    one_ten: [one_nine, zero_ten],
}

def test_build_physical_qubits_graph():
    t = ct.DynamicLookAheadHeuristicCircuitTransformer(cirq.google.Foxtail)
    g = t.build_physical_qubits_graph()
    assert len(g) == len(physical)
    for q in physical:
        assert set(physical[q]) == set(g[q])

def test_get_least_connected_qubit():
    t = ct.DynamicLookAheadHeuristicCircuitTransformer(cirq.google.Foxtail)
    g = {
        0: [1],
        1: [0, 2],
        2: [1],
        3: [4],
        4: [3],
    }
    assert t.get_least_connected_qubit(g, deque([0, 1, 2])) == 0
    assert t.get_least_connected_qubit(g, deque([3, 4])) == 3

def test_build_logical_qubits_graph():
    t = ct.DynamicLookAheadHeuristicCircuitTransformer(cirq.google.Foxtail)
    # One connected component.
    c = cirq.Circuit(
        cirq.ISWAP(a2, a0),
        cirq.ISWAP(a0, a1),
        cirq.ISWAP(a0, a2),
        cirq.ISWAP(a2, a1),
        cirq.ISWAP(a1, a2),
        cirq.ISWAP(a2, a3),
        cirq.ISWAP(a0, a3),
    )
    assert t.build_logical_qubits_graph(c) == logical
    # Three connected components.
    c2 = cirq.Circuit(
        cirq.ISWAP(a2, a0),
        cirq.ISWAP(a0, a1),
        cirq.ISWAP(a0, a2),
        cirq.ISWAP(a2, a1),
        cirq.ISWAP(a1, a2),
        cirq.ISWAP(a2, a3),
        cirq.ISWAP(a0, a3),
        cirq.ISWAP(a4, a5),
        cirq.ISWAP(a6, a7),
    )
    assert t.build_logical_qubits_graph(c2) == logical2
    c3 = cirq.Circuit(cirq.X(a1), cirq.X(a2), cirq.X(a3))
    assert t.build_logical_qubits_graph(c3) == {
        a1: [(a3, 2)],
        a2: [(a3, 1)],
        a3: [(a2, 1), (a1, 2)],
    }

def test_graph_center():
    t = ct.DynamicLookAheadHeuristicCircuitTransformer(cirq.google.Foxtail)
    assert t.find_graph_center(physical) == zero_five
    assert t.find_graph_center(logical) == a0

def test_traverse():
    t = ct.DynamicLookAheadHeuristicCircuitTransformer(cirq.google.Foxtail)
    assert t.traverse(logical, a0) == deque([a0, a2, a1, a3])

def test_find_reference_qubits():
    t = ct.DynamicLookAheadHeuristicCircuitTransformer(cirq.google.Foxtail)
    mapping = {
        a0: zero_five,
    }
    assert t.find_reference_qubits(mapping, logical, a2) == [zero_five]

def test_find_candidate_qubits():
    t = ct.DynamicLookAheadHeuristicCircuitTransformer(cirq.google.Foxtail)
    # First level has free qubits.
    mapping = {
        zero_five: "mapped",
    }
    assert t.find_candidate_qubits(mapping, physical, zero_five) == [
        cirq.GridQubit(0, 4),
        cirq.GridQubit(1, 5),
        cirq.GridQubit(0, 6),
    ]
    # Second level has free qubits.
    mapping = {
        zero_five: "mapped",
        zero_four: "mapped",
        zero_six: "mapped",
        one_five: "mapped",
    }
    assert t.find_candidate_qubits(mapping, physical, zero_five) == [
        cirq.GridQubit(0, 3),
        cirq.GridQubit(1, 4),
        cirq.GridQubit(1, 6),
        cirq.GridQubit(0, 7),
    ]
    # Third level has free qubits.
    mapping = {
        zero_three: "mapped",
        zero_four: "mapped",
        zero_five: "mapped",
        zero_six: "mapped",
        zero_seven: "mapped",
        one_four: "mapped",
        one_five: "mapped",
        one_six: "mapped",
    }
    assert t.find_candidate_qubits(mapping, physical, zero_five) == [
        cirq.GridQubit(0, 2),
        cirq.GridQubit(1, 3),
        cirq.GridQubit(1, 7),
        cirq.GridQubit(0, 8),
    ]

def test_find_shortest_path_distance():
    t = ct.DynamicLookAheadHeuristicCircuitTransformer(cirq.google.Foxtail)
    g = {
        0: [1, 7],
        1: [0, 2, 7],
        2: [1, 3, 5, 8],
        3: [2, 4, 5],
        4: [3, 5],
        5: [2, 3, 4, 6],
        6: [5, 7],
        7: [0, 1, 6, 8],
        8: [2, 7],
    }
    assert t.find_shortest_path_distance(g, 0, 5) == 3

def test_calculate_initial_mapping():
    t = ct.DynamicLookAheadHeuristicCircuitTransformer(cirq.google.Foxtail)
    c = cirq.Circuit(
        cirq.ISWAP(a2, a0),
        cirq.ISWAP(a0, a1),
        cirq.ISWAP(a0, a2),
        cirq.ISWAP(a2, a1),
        cirq.ISWAP(a1, a2),
        cirq.ISWAP(a2, a3),
        cirq.ISWAP(a0, a3),
    )
    print(t.transform(c))
