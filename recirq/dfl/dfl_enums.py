# Copyright 2025 Google
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

"""Enums defining the core gates, states, and bases for
the Disorder-Free Localization (DFL) experiment.
"""

import enum

class TwoQubitGate(enum.Enum):
    """Available two-qubit gate types for the Trotter simulation.

    CZ refers to the two-qubit 'cz' gate
    CPHASE refers to the 'cphase' operation
    """
    CZ = 'cz'
    CPHASE = 'cphase'


class InitialState(enum.Enum):
    """Available initial quantum states for the DFL experiment.

    SINGLE_SECTOR refers to the gauge invariant 'single_sector' state
    SUPERPOSITION refers to the 'superposition' state of disorder configurations.
    """
    SINGLE_SECTOR = 'single_sector'
    SUPERPOSITION = 'superposition'


class Basis(enum.Enum):
    """Available bases for the DFL experiment.

    LGT refers to the local lattice gauge theory ('lgt') basis.
    DUAL refers to the 'dual' basis.
    """
    LGT = 'lgt'
    DUAL = 'dual'
