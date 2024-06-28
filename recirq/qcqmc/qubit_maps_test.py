# Copyright 2024 Google
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
"""Various mappings between fermions and qubits."""

import cirq
import pytest

from recirq.qcqmc import qubit_maps


def test_get_qubits_a_b():
    n_orb = 4
    qubits = qubit_maps.get_qubits_a_b(n_orb=n_orb)
    assert len(qubits) == 8
    assert qubits == (
        cirq.GridQubit(0, 0),
        cirq.GridQubit(0, 1),
        cirq.GridQubit(0, 2),
        cirq.GridQubit(0, 3),
        cirq.GridQubit(1, 0),
        cirq.GridQubit(1, 1),
        cirq.GridQubit(1, 2),
        cirq.GridQubit(1, 3),
    )


def test_get_qubits_a_b_reversed():
    n_orb = 4
    qubits = qubit_maps.get_qubits_a_b_reversed(n_orb=n_orb)
    assert len(qubits) == 8
    assert qubits == (
        cirq.GridQubit(0, 0),
        cirq.GridQubit(0, 1),
        cirq.GridQubit(0, 2),
        cirq.GridQubit(0, 3),
        cirq.GridQubit(1, 3),
        cirq.GridQubit(1, 2),
        cirq.GridQubit(1, 1),
        cirq.GridQubit(1, 0),
    )


def test_get_4_qubit_fermion_qubit_map():
    fermion_qubit_map = qubit_maps.get_4_qubit_fermion_qubit_map()
    assert fermion_qubit_map == {
        2: cirq.GridQubit(0, 0),
        3: cirq.GridQubit(1, 0),
        0: cirq.GridQubit(0, 1),
        1: cirq.GridQubit(1, 1),
    }


def test_get_8_qubit_fermion_qubit_map():
    fermion_qubit_map = qubit_maps.get_8_qubit_fermion_qubit_map()
    assert fermion_qubit_map == {
        2: cirq.GridQubit(0, 0),
        3: cirq.GridQubit(1, 0),
        4: cirq.GridQubit(0, 1),
        5: cirq.GridQubit(1, 1),
        0: cirq.GridQubit(0, 2),
        1: cirq.GridQubit(1, 2),
        6: cirq.GridQubit(0, 3),
        7: cirq.GridQubit(1, 3),
    }


def test_get_12_qubit_fermion_qubit_map():
    fermion_qubit_map = qubit_maps.get_12_qubit_fermion_qubit_map()
    assert fermion_qubit_map == {
        4: cirq.GridQubit(0, 0),
        5: cirq.GridQubit(1, 0),
        6: cirq.GridQubit(0, 1),
        7: cirq.GridQubit(1, 1),
        2: cirq.GridQubit(0, 2),
        3: cirq.GridQubit(1, 2),
        8: cirq.GridQubit(0, 3),
        9: cirq.GridQubit(1, 3),
        0: cirq.GridQubit(0, 4),
        1: cirq.GridQubit(1, 4),
        10: cirq.GridQubit(0, 5),
        11: cirq.GridQubit(1, 5),
    }


def test_get_16_qubit_fermion_qubit_map():
    fermion_qubit_map = qubit_maps.get_16_qubit_fermion_qubit_map()
    assert fermion_qubit_map == {
        6: cirq.GridQubit(0, 0),
        7: cirq.GridQubit(1, 0),
        8: cirq.GridQubit(0, 1),
        9: cirq.GridQubit(1, 1),
        4: cirq.GridQubit(0, 2),
        5: cirq.GridQubit(1, 2),
        10: cirq.GridQubit(0, 3),
        11: cirq.GridQubit(1, 3),
        2: cirq.GridQubit(0, 4),
        3: cirq.GridQubit(1, 4),
        12: cirq.GridQubit(0, 5),
        13: cirq.GridQubit(1, 5),
        0: cirq.GridQubit(0, 6),
        1: cirq.GridQubit(1, 6),
        14: cirq.GridQubit(0, 7),
        15: cirq.GridQubit(1, 7),
    }


@pytest.mark.parametrize("n_qubits", (4, 8, 12, 16))
def test_get_fermion_qubit_map_pp_plus(n_qubits):
    fermion_qubit_map = qubit_maps.get_fermion_qubit_map_pp_plus(n_qubits=n_qubits)
    assert len(fermion_qubit_map) == n_qubits


@pytest.mark.parametrize("n_qubits", (4, 8, 12, 16))
def test_get_mode_qubit_map_pp_plus(n_qubits):
    mode_qubit_map = qubit_maps.get_mode_qubit_map_pp_plus(n_qubits=n_qubits)
    assert len(mode_qubit_map) == n_qubits
