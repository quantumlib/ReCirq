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

grid_qubits = dict(
    (f'{row}_{col}', cirq.GridQubit(row, col))
    for row in range(2) for col in range(11)
)


def test_build_physical_qubits_graph():
    t = ct.DynamicLookAheadHeuristicCircuitTransformer(cirq.google.Foxtail)
    g = t.build_physical_qubits_graph()
    expected = {
        grid_qubits['0_0']: [
            grid_qubits['0_1'],
            grid_qubits['1_0']
        ],
        grid_qubits['0_1']: [
            grid_qubits['0_0'],
            grid_qubits['1_1'],
            grid_qubits['0_2'],
        ],
        grid_qubits['0_2']: [
            grid_qubits['0_1'],
            grid_qubits['1_2'],
            grid_qubits['0_3'],
        ],
        grid_qubits['0_3']: [
            grid_qubits['0_2'],
            grid_qubits['1_3'],
            grid_qubits['0_4'],
        ],
        grid_qubits['0_4']: [
            grid_qubits['0_3'],
            grid_qubits['1_4'],
            grid_qubits['0_5'],
        ],
        grid_qubits['0_5']: [
            grid_qubits['0_4'],
            grid_qubits['1_5'],
            grid_qubits['0_6'],
        ],
        grid_qubits['0_6']: [
            grid_qubits['0_5'],
            grid_qubits['1_6'],
            grid_qubits['0_7'],
        ],
        grid_qubits['0_7']: [
            grid_qubits['0_6'],
            grid_qubits['1_7'],
            grid_qubits['0_8'],
        ],
        grid_qubits['0_8']: [
            grid_qubits['0_7'],
            grid_qubits['1_8'],
            grid_qubits['0_9'],
        ],
        grid_qubits['0_9']: [
            grid_qubits['0_8'],
            grid_qubits['1_9'],
            grid_qubits['0_10'],
        ],
        grid_qubits['0_10']: [
            grid_qubits['0_9'],
            grid_qubits['1_10'],
        ],
        grid_qubits['1_0']: [
            grid_qubits['1_1'],
            grid_qubits['0_0'],
        ],
        grid_qubits['1_1']: [
            grid_qubits['1_0'],
            grid_qubits['0_1'],
            grid_qubits['1_2'],
        ],
        grid_qubits['1_2']: [
            grid_qubits['1_1'],
            grid_qubits['0_2'],
            grid_qubits['1_3'],
        ],
        grid_qubits['1_3']: [
            grid_qubits['1_2'],
            grid_qubits['0_3'],
            grid_qubits['1_4'],
        ],
        grid_qubits['1_4']: [
            grid_qubits['1_3'],
            grid_qubits['0_4'],
            grid_qubits['1_5'],
        ],
        grid_qubits['1_5']: [
            grid_qubits['1_4'],
            grid_qubits['0_5'],
            grid_qubits['1_6'],
        ],
        grid_qubits['1_6']: [
            grid_qubits['1_5'],
            grid_qubits['0_6'],
            grid_qubits['1_7'],
        ],
        grid_qubits['1_7']: [
            grid_qubits['1_6'],
            grid_qubits['0_7'],
            grid_qubits['1_8'],
        ],
        grid_qubits['1_8']: [
            grid_qubits['1_7'],
            grid_qubits['0_8'],
            grid_qubits['1_9'],
        ],
        grid_qubits['1_9']: [
            grid_qubits['1_8'],
            grid_qubits['0_9'],
            grid_qubits['1_10'],
        ],
        grid_qubits['1_10']: [
            grid_qubits['1_9'],
            grid_qubits['0_10'],
        ],
    }
    assert len(g) == len(expected)
    for q in expected:
        assert set(g[q]) == set(expected[q])


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
        cirq.ISWAP(a1, a2),
        cirq.ISWAP(a2, a3),
    )
    assert t.build_logical_qubits_graph(c) == {
        a0: [(a2, 0), (a1, 1)],
        a1: [(a0, 1), (a2, 3)],
        a2: [(a0, 0), (a1, 3), (a3, 4)],
        a3: [(a2, 4)],
    }
    # Three connected components with one-qubit and two-qubit gates.
    c = cirq.Circuit(
        cirq.ISWAP(a2, a0),
        cirq.ISWAP(a0, a1),
        cirq.ISWAP(a0, a2),
        cirq.ISWAP(a1, a2),
        cirq.ISWAP(a2, a3),
        cirq.ISWAP(a4, a5),
        cirq.ISWAP(a6, a7),
    )
    assert t.build_logical_qubits_graph(c) == {
        a0: [(a2, 0), (a1, 1)],
        a1: [(a0, 1), (a2, 3)],
        a2: [(a0, 0), (a1, 3), (a3, 4)],
        a3: [(a2, 4), (a7, 6)],
        a4: [(a5, 0), (a6, 5)],
        a5: [(a4, 0)],
        a6: [(a7, 0), (a4, 5)],
        a7: [(a6, 0), (a3, 6)],
    }
    # Three connected components with only one-qubit gates.
    c = cirq.Circuit(cirq.X(a1), cirq.X(a2), cirq.X(a3))
    assert t.build_logical_qubits_graph(c) == {
        a1: [(a3, 2)],
        a2: [(a3, 1)],
        a3: [(a2, 1), (a1, 2)],
    }


def test_graph_center():
    t = ct.DynamicLookAheadHeuristicCircuitTransformer(cirq.google.Foxtail)
    g = {
        0: [1, 4],
        1: [0, 2, 4],
        2: [1, 4, 5],
        3: [4],
        4: [0, 1, 2, 3, 5],
        5: [2, 4],
    }
    assert t.find_graph_center(g) == 4
    g = {
        0: [(1,), (4,)],
        1: [(0,), (2,), (4,)],
        2: [(1,), (4,), (5,)],
        3: [(4,)],
        4: [(0,), (1,), (2,), (3,), (5,)],
        5: [(2,), (4,)],
    }
    assert t.find_graph_center(g) == 4


def test_traverse():
    t = ct.DynamicLookAheadHeuristicCircuitTransformer(cirq.google.Foxtail)
    g = {
        0: [(2, 0), (1, 1), (3, 6)],
        1: [(0, 1), (2, 3)],
        2: [(0, 0), (1, 3), (3, 5)],
        3: [(2, 5), (0, 6)],
    }
    assert t.traverse(g, 0) == deque([0, 2, 1, 3])
    assert t.traverse(g, 1) == deque([1, 0, 2, 3])
    assert t.traverse(g, 2) == deque([2, 0, 1, 3])
    assert t.traverse(g, 3) == deque([3, 2, 0, 1])


def test_find_reference_qubits():
    t = ct.DynamicLookAheadHeuristicCircuitTransformer(cirq.google.Foxtail)
    g = {
        a0: [(a2, 0), (a1, 1)],
        a1: [(a0, 1), (a2, 3)],
        a2: [(a0, 0), (a1, 3), (a3, 5)],
        a3: [(a2, 5)],
    }
    mapping = {
        a0: grid_qubits['0_5'],
    }
    assert set(t.find_reference_qubits(mapping, g, a2)) == {
        grid_qubits['0_5'],
    }
    mapping = {
        a0: grid_qubits['0_5'],
        a2: grid_qubits['1_5'],
    }
    assert set(t.find_reference_qubits(mapping, g, a1)) == {
        grid_qubits['0_5'],
        grid_qubits['1_5'],
    }


def test_find_candidate_qubits():
    t = ct.DynamicLookAheadHeuristicCircuitTransformer(cirq.google.Foxtail)
    g = t.build_physical_qubits_graph()
    # First level has free qubits.
    mapped = {
        grid_qubits['0_5'],
    }
    assert set(t.find_candidate_qubits(mapped, g, grid_qubits['0_5'])) == {
        cirq.GridQubit(0, 4),
        cirq.GridQubit(1, 5),
        cirq.GridQubit(0, 6),
    }
    # Second level has free qubits.
    mapped = {
        grid_qubits['0_4'],
        grid_qubits['0_5'],
        grid_qubits['0_6'],
        grid_qubits['1_5'],
    }
    assert set(t.find_candidate_qubits(mapped, g, grid_qubits['0_5'])) == {
        cirq.GridQubit(0, 3),
        cirq.GridQubit(1, 4),
        cirq.GridQubit(1, 6),
        cirq.GridQubit(0, 7),
    }
    # Third level has free qubits.
    mapped = {
        grid_qubits['0_3'],
        grid_qubits['0_4'],
        grid_qubits['0_5'],
        grid_qubits['0_6'],
        grid_qubits['0_7'],
        grid_qubits['1_4'],
        grid_qubits['1_5'],
        grid_qubits['1_6'],
    }
    assert set(t.find_candidate_qubits(mapped, g, grid_qubits['0_5'])) == {
        cirq.GridQubit(0, 2),
        cirq.GridQubit(1, 3),
        cirq.GridQubit(1, 7),
        cirq.GridQubit(0, 8),
    }


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
    assert t.find_shortest_path_distance(g, 1, 8) == 2
    assert t.find_shortest_path_distance(g, 4, 7) == 3
