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

import abc
import hashlib
import pathlib
from pathlib import Path

import attrs
import numpy as np

DEFAULT_BASE_DATA_DIR = (Path(__file__).parent / "data").resolve()


def get_integrals_path(
    name: str, *, base_data_dir: pathlib.Path = DEFAULT_BASE_DATA_DIR
) -> Path:
    """Find integral data file by name.

    Args:
        name: The molecule name.
        base_data_dir: The base directory where the data folder should live.
    """
    return base_data_dir / "integrals" / name / "hamiltonian.chk"


_MOLECULE_INFO = {"fh_sto3g"}


# should go in Dedicated module
@attrs.frozen
class Params(abc.ABC):
    """Store named parameters which will potentiall be read from a file.

    Args:
        name: The name of the parameters
    """

    name: str

    def __eq__(self, other):
        """A helper method to compare two dataclasses which might contain arrays."""
        if self is other:
            return True

        if self.__class__ is not other.__class__:
            return NotImplemented

        return all(
            array_compatible_eq(thing1, thing2)
            for thing1, thing2 in zip(
                dataclasses.astuple(self), dataclasses.astuple(other)
            )
        )

    @property
    def path_string(self) -> str:
        raise NotImplementedError()

    @property
    def base_path(self) -> pathlib.Path:
        return pathlib.Path(self.path_string + "_" + self.hash_key)

    @property
    def hash_key(self) -> str:
        """Gets the hash key for a set of params."""
        return hashlib.md5(str(self).encode("utf-8")).hexdigest()[0:16]

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        # This custom repr lets us add fields with default values without changing
        # the repr. This, in turn, lets us use the hash_key reliably even when
        # we add new fields with a default value.
        fields = dataclasses.fields(self)
        # adjusted_fields = [f for f in fields if getattr(self, f.name) != f.default]
        adjusted_fields = [
            f
            for f in fields
            if not array_compatible_eq(getattr(self, f.name), f.default)
        ]

        return (
            self.__class__.__qualname__
            + "("
            + ", ".join([f"{f.name}={getattr(self, f.name)}" for f in adjusted_fields])
            + ")"
        )

    @classmethod
    def _json_namespace_(cls):
        return "recirq.qcqmc"


# Should go in dedicated module
@attrs.frozen
class Data(abc.ABC):
    """A container for different forms of data used in qcqmc (hamiltonian, trial wavefunction, ...).

    Args:
        params: The name of the parameters
    """

    params: Params

    def __eq__(self, other):
        """A helper method to compare two dataclasses which might contain arrays."""
        if self is other:
            return True

        if self.__class__ is not other.__class__:
            return NotImplemented

        return all(
            array_compatible_eq(thing1, thing2)
            for thing1, thing2 in zip(attrs.astuple(self), attrs.astuple(other))
        )

    @classmethod
    def _json_namespace_(cls):
        return "recirq.qcqmc"


def array_compatible_eq(thing1, thing2):
    """A check for equality which can handle arrays."""
    if thing1 is thing2:
        return True

    # Here we handle dicts because they might have arrays in them.
    if isinstance(thing1, dict) and isinstance(thing2, dict):
        return all(
            array_compatible_eq(k_1, k_2) and array_compatible_eq(v_1, v_2)
            for (k_1, v_1), (k_2, v_2) in zip(thing1.items(), thing2.items())
        )
    if isinstance(thing1, np.ndarray) and isinstance(thing2, np.ndarray):
        return np.array_equal(thing1, thing2)
    if isinstance(thing1, np.ndarray) + isinstance(thing2, np.ndarray) == 1:
        return False
    try:
        return thing1 == thing2
    except TypeError:
        return NotImplemented
