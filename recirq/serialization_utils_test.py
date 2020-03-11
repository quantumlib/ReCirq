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

import recirq

import abc
import inspect
import io
import itertools
import textwrap

import networkx as nx
import numpy as np
import pytest

import cirq


def test_bits_roundtrip():
    bitstring = np.asarray([0, 1, 0, 1, 1, 1, 1, 0, 0, 1])
    b = recirq.BitArray(bitstring)

    buffer = io.StringIO()
    cirq.to_json(b, buffer)

    buffer.seek(0)
    text = buffer.read()
    assert text == """{
  "cirq_type": "recirq.BitArray",
  "shape": [
    10
  ],
  "packedbits": "5e40"
}"""

    buffer.seek(0)
    b2 = recirq.read_json(buffer)
    assert b == b2


def test_bits_roundtrip_big():
    bits = np.random.choice([0, 1], size=(30_000, 53))
    b = recirq.BitArray(bits)
    buffer = io.StringIO()
    cirq.to_json(b, buffer)
    buffer.seek(0)
    b2 = recirq.read_json(buffer)
    assert b == b2

    bits = np.random.choice([0, 1], size=(3000, 11, 53))
    b = recirq.BitArray(bits)
    buffer = io.StringIO()
    cirq.to_json(b, buffer)
    buffer.seek(0)
    b2 = recirq.read_json(buffer)
    assert b == b2


def test_bitstrings_roundtrip_big():
    bitstrings = np.random.choice([0, 1], size=(30_000, 53))
    ba = recirq.BitArray(bitstrings)

    buffer = io.StringIO()
    cirq.to_json(ba, buffer)
    buffer.seek(0)
    ba2 = recirq.read_json(buffer)
    assert ba == ba2


def test_numpy_roundtrip(tmpdir):
    re = np.random.uniform(0, 1, 100)
    im = np.random.uniform(0, 1, 100)
    a = re + 1.j * im
    a = np.reshape(a, (10, 10))
    ba = recirq.NumpyArray(a)

    fn = f'{tmpdir}/hello.json'
    cirq.to_json(ba, fn)
    ba2 = recirq.read_json(fn)

    assert ba == ba2


def test_str_and_repr():
    bits = np.array([0, 1, 0, 1])
    assert str(recirq.BitArray(bits)) == 'recirq.BitArray([0 1 0 1])'
    assert repr(recirq.BitArray(bits)) == 'recirq.BitArray(array([0, 1, 0, 1]))'

    nums = np.array([1, 2, 3, 4])
    assert str(recirq.NumpyArray(nums)) == 'recirq.NumpyArray([1 2 3 4])'
    assert repr(recirq.NumpyArray(nums)) == 'recirq.NumpyArray(array([1, 2, 3, 4]))'
