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

import dataclasses
import datetime
import glob
import io
import itertools
import os
import sys
from typing import Dict, Any, Type, Union, List

import numpy as np

from cirq import protocols

if sys.version_info > (3, 8):
    from typing import Protocol
else:
    Protocol = object


class Task(Protocol):
    fn: str


def exists(task: Task, base_dir: str):
    fn = f'{base_dir}/{task.fn}.json'
    return os.path.exists(fn)


def save(task: Task, data: Dict[str, Any], base_dir: str, mode='x'):
    with_meta = {
        'timestamp': datetime.datetime.now().isoformat(),
        'task': task,
    }
    with_meta.update(data)

    fn = f'{base_dir}/{task.fn}.json'
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with open(fn, mode) as f:
        protocols.to_json(with_meta, f)
    return fn


class Registry:
    """Mutable registry of cirq_type class name to class object to
    assist in deserialization.
    """

    def __init__(self):
        self._mapping = dict()

    def register(self, cirq_type: str, cls):
        if not cirq_type.startswith('recirq.'):
            raise ValueError("Please only use recirq.Registry for objects "
                             "in the recirq namespace.")

        self._mapping[cirq_type] = cls

    def get(self, cirq_type: str):
        return self._mapping.get(cirq_type, None)


# Singleton pattern:
Registry = Registry()


class NumpyArray:
    """Support for compact serialization of a numpy array.

    Instead of transforming data to a list-of-lists, this hex-encodes
    a binary representation of the numpy array.
    """

    def __init__(self, a: np.ndarray):
        self.a = a

    def _json_dict_(self):
        buffer = io.BytesIO()
        np.save(buffer, self.a, allow_pickle=False)
        buffer.seek(0)
        d = {
            'cirq_type': 'recirq.' + self.__class__.__name__,
            'npy': buffer.read().hex(),
        }
        buffer.close()
        return d

    @classmethod
    def _from_json_dict_(cls, npy: str, **kwargs):
        buffer = io.BytesIO()
        buffer.write(bytes.fromhex(npy))
        buffer.seek(0)
        a = np.load(buffer, allow_pickle=False)
        buffer.close()
        return cls(a)

    def __str__(self):
        return f'recirq.NumpyArray({self.a})'

    def __repr__(self):
        return f'recirq.NumpyArray({repr(self.a)})'

    def __eq__(self, other):
        return np.array_equal(self.a, other.a)


class BitArray:
    """A serializable wrapper for arrays specifically of bits.

    This is very similar to ``NumpyArray``, except it first "packs"
    bits into uint8's, potentially saving a factor of eight in storage
    size. The resulting binary buffer is still hex encoded into
    a JSON string.
    """

    def __init__(self, bits: np.ndarray):
        self.bits = bits

    def _json_dict_(self):
        packed_bits = np.packbits(self.bits)
        assert packed_bits.dtype == np.uint8, packed_bits.dtype
        return {
            'cirq_type': 'recirq.' + self.__class__.__name__,
            'shape': self.bits.shape,
            'packedbits': packed_bits.tobytes().hex(),
        }

    @classmethod
    def _from_json_dict_(cls, shape: List[int], packedbits: str, **kwargs):
        # Hex -> bytes -> packed array -> padded array -> final array
        bits_bytes = bytes.fromhex(packedbits)
        bits = np.frombuffer(bits_bytes, dtype=np.uint8)
        bits = np.unpackbits(bits)
        bits = bits[:np.prod(shape)].reshape(shape)
        return cls(bits)

    def __str__(self):
        return f'recirq.BitArray({self.bits})'

    def __repr__(self):
        return f'recirq.BitArray({repr(self.bits)})'

    def __eq__(self, other):
        return np.array_equal(self.bits, other.bits)


def _recirq_class_resolver(cirq_type: str) -> Union[None, Type]:
    from recirq import BitArray, NumpyArray
    return {
        'recirq.BitArray': BitArray,
        'recirq.NumpyArray': NumpyArray,
    }.get(cirq_type)


DEFAULT_RESOLVERS = [_recirq_class_resolver, Registry.get] \
                    + protocols.DEFAULT_RESOLVERS


def read_json(file_or_fn=None, *, json_text=None, resolvers=None):
    """Read a JSON file that optionally contains cirq objects.

    This differs from cirq.read_json by its default argument for
    `resolvers`. Use this function if your JSON document contains
    recirq objects.
    """
    if resolvers is None:
        resolvers = DEFAULT_RESOLVERS

    return protocols.read_json(file_or_fn=file_or_fn,
                               json_text=json_text,
                               resolvers=resolvers)


def iterload_records(dataset_id: str, base_dir: str):
    """Helper function to iteratively load records saved while
    following the data collection idioms.

    The yielded records will be dictionaries exactly as structured in the
    JSON document.
    """
    for fn in glob.iglob(f'{base_dir}/{dataset_id}/**/*.json', recursive=True):
        yield read_json(fn)


def load_records(dataset_id: str, base_dir: str):
    """Helper function to load records saved while
    following the data collection idioms.

    The returned records will be dictionaries exactly as structured in the
    JSON document.
    """
    return list(iterload_records(dataset_id, base_dir))


def flatten_dataclass_into_record(record, key):
    """Helper function to 'flatten' a dataclass.

    This is useful for flattening a JSON hierarchy for construction
    of a Pandas DataFrame.
    """
    dc_dict = dataclasses.asdict(record[key])
    record.update(**dc_dict)
    if key not in dc_dict:
        # Otherwise, we've already overwritten the original key
        del record[key]


def roundrobin(*iterables):
    """Iterate through `iterables` in a 'roundrobin' fashion.

    Taken from
    https://docs.python.org/3.7/library/itertools.html#itertools-recipes
    """
    num_active = len(iterables)
    nexts = itertools.cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = itertools.cycle(itertools.islice(nexts, num_active))
