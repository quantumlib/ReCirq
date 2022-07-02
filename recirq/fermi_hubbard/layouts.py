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
"""Embeddings of Fermi-Hubbard problem on a chip."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type, Union

import cirq

GridQubitPairs = List[Tuple[cirq.GridQubit, cirq.GridQubit]]


@dataclass(init=False, unsafe_hash=True)
class LineLayout:
    """Mapping of problem qubits to two parallel lines on a grid.

    This mapping encodes fermions under Jordan-Wigner transformations to qubits
    on a chip. This is the most favourable connectivity, where all the
    next-neighbour and on-site interactions can be realized without any swap
    between fermions.

    For example, the eight-site embedding on a grid looks like:

    1↑──2↑──3↑──4↑──5↑──6↑──7↑──8↑
    ┃   ┃   ┃   ┃   ┃   ┃   ┃   ┃
    1↓──2↓──3↓──4↓──5↓──6↓──7↓──8↓

    where thin lines symbolize next-neighbour interactions and bold lines the
    on-site interactions.

    When instance of this layout is used in FermiHubbardParameters, the
    create_circuits function constructs a cirq.Circuit mapped to given qubits
    which realize a Trotterized evolution.

    Attributes:
        size: Number of sites for each chain.
        origin: The starting qubit coordinates on a 2D grid. The up chain will
            start from this qubit and the down chain will start at the qubit
            which is 90 degrees clockwise from the line direction.
        rotation: Counter-clockwise rotation of the (1, 0) vector which
            indicates the direction where line should be traced. Only multiplies
            of 90 are allowed.
    """

    size: int
    origin: Tuple[int, int]
    rotation: int

    def __init__(self, *,
                 size: int,
                 origin: Tuple[int, int] = (0, 0),
                 rotation: int = 0) -> None:
        a, b = origin
        self.origin = a, b
        self.size = size
        self.rotation = rotation

        self._initialize_layout()

    def _initialize_layout(self) -> None:
        up_qubits, down_qubits = _find_line_qubits(self.size,
                                                   self.origin,
                                                   self.rotation)

        up_even_pairs, up_odd_pairs = _get_even_odd_pairs(up_qubits)
        down_even_pairs, down_odd_pairs = _get_even_odd_pairs(down_qubits)

        self._hop_even_pairs = down_even_pairs + up_even_pairs
        self._hop_odd_pairs = down_odd_pairs + up_odd_pairs

        self._up_qubits = up_qubits
        self._down_qubits = down_qubits

    @classmethod
    def cirq_resolvers(cls) -> Dict[str, Optional[Type]]:
        return {cls.__name__: cls}

    @property
    def up_qubits(self) -> List[cirq.GridQubit]:
        return self._up_qubits

    @property
    def down_qubits(self) -> List[cirq.GridQubit]:
        return self._down_qubits

    @property
    def all_qubits(self) -> List[cirq.GridQubit]:
        return self.up_qubits + self.down_qubits

    @property
    def up_even_pairs(self) -> GridQubitPairs:
        return _get_even_pairs(self.up_qubits)

    @property
    def down_even_pairs(self) -> GridQubitPairs:
        return _get_even_pairs(self.down_qubits)

    @property
    def up_odd_pairs(self) -> GridQubitPairs:
        return _get_odd_pairs(self.up_qubits)

    @property
    def down_odd_pairs(self) -> GridQubitPairs:
        return _get_odd_pairs(self.down_qubits)

    @property
    def interaction_pairs(self) -> GridQubitPairs:
        return list(zip(self.up_qubits, self._down_qubits))

    def default_layout(self) -> 'LineLayout':
        return LineLayout(size=self.size)

    def text_diagram(self, draw_grid_coords: bool = True) -> str:
        return _draw_chains(self.up_qubits,
                            self.down_qubits,
                            self.interaction_pairs,
                            draw_grid_coords)

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)


@dataclass(init=False, unsafe_hash=True)
class ZigZagLayout:
    """Mapping of problem qubits to two zig-zag lines on a grid.

    This mapping encodes fermions under Jordan-Wigner transformations to qubits
    on a chip in a more compact way compared to LineLayout. Because of that, not
    all on-site interactions can be realized simultaneously and a Fermionic
    swaps are necessary to do them. This influences a cost of a single Trotter
    step because the interaction terms are realized in two moments, and an
    additional swapping layer is added.

    For example, the eight-site embedding on a grid looks like:

    1↓━━1↑
    │   │
    2↓━━2↑──3↑
    │       │
    3↓──4↓━━4↑──5↑
        │       │
        5↓──6↓━━6↑──7↑
            │       │
            7↓──8↓━━8↑

    where thin lines symbolize next-neighbour interactions and bold lines the
    on-site interactions.

    When instance of this layout is used in FermiHubbardParameters, the
    create_circuits function constructs a cirq.Circuit mapped to given qubits
    which realize a Trotterized evolution.

    Attributes:
        size: Number of sites for each chain.
        origin: The starting qubit coordinates on a 2D grid where zig-zag is
            traced from.
        rotation: Counter-clockwise rotation of the (1, 0) vector which
            indicates the direction where zig-zag should be traced. Only
            multiplies of 90 are allowed.
        flipped: If true, zig-zag is flipped along the axis perpendicular to the
            traced direction.
        exchange_chains: If true, qubits on up an down chain are exchanged with
            each other.
        reverse_chains: If true, qubits on each chain are reversed.
    """

    size: int
    origin: Tuple[int, int]
    rotation: int
    flipped: bool
    exchange_chains: bool
    reverse_chains: bool

    def __init__(self, *,
                 size: int,
                 origin: Tuple[int, int] = (0, 0),
                 rotation: int = 0,
                 flipped: bool = False,
                 exchange_chains: bool = False,
                 reverse_chains: bool = False) -> None:
        self.size = size
        a, b = origin
        self.origin = (a, b)
        self.rotation = rotation
        self.flipped = flipped
        self.exchange_chains = exchange_chains
        self.reverse_chains = reverse_chains

        self._initialize_layout()

    def _initialize_layout(self) -> None:

        up_qubits, down_qubits = _find_zigzag_qubits(self.size,
                                                     self.origin,
                                                     self.rotation,
                                                     self.flipped,
                                                     self.exchange_chains)

        self._int_even_pairs = [(down_qubits[i], up_qubits[i])
                                for i in range(1, self.size, 2)]
        self._int_even_other_qubits = down_qubits[0::2] + up_qubits[0::2]

        self._int_odd_pairs = ([(down_qubits[0], up_qubits[0])] +
                               [(down_qubits[i], up_qubits[i])
                                for i in range(1, self.size - 2, 2)])
        self._int_odd_other_qubits = (down_qubits[2:-2:2] + down_qubits[-2:] +
                                      up_qubits[2:-2:2] + up_qubits[-2:])

        self._up_even_pairs = _get_even_pairs(up_qubits)
        self._up_odd_pairs = _get_odd_pairs(up_qubits)

        self._down_even_pairs = _get_even_pairs(down_qubits)
        self._down_odd_pairs = _get_odd_pairs(down_qubits)

        if self.reverse_chains:
            up_qubits = list(reversed(up_qubits))
            down_qubits = list(reversed(down_qubits))
        else:
            up_qubits = up_qubits
            down_qubits = down_qubits

        self._up_qubits = up_qubits
        self._down_qubits = down_qubits

    @classmethod
    def cirq_resolvers(cls) -> Dict[str, Optional[Type]]:
        return {cls.__name__: cls}

    @property
    def up_qubits(self) -> List[cirq.GridQubit]:
        return self._up_qubits

    @property
    def down_qubits(self) -> List[cirq.GridQubit]:
        return self._down_qubits

    @property
    def all_qubits(self) -> List[cirq.GridQubit]:
        return self.up_qubits + self.down_qubits

    @property
    def up_even_pairs(self) -> GridQubitPairs:
        return self._up_even_pairs

    @property
    def down_even_pairs(self) -> GridQubitPairs:
        return self._down_even_pairs

    @property
    def up_odd_pairs(self) -> GridQubitPairs:
        return self._up_odd_pairs

    @property
    def down_odd_pairs(self) -> GridQubitPairs:
        return self._down_odd_pairs

    @property
    def interaction_even_pairs(self) -> GridQubitPairs:
        return self._int_even_pairs

    @property
    def interaction_even_other_qubits(self) -> List[cirq.GridQubit]:
        return self._int_even_other_qubits

    @property
    def interaction_odd_pairs(self) -> GridQubitPairs:
        return self._int_odd_pairs

    @property
    def interaction_odd_other_qubits(self) -> List[cirq.GridQubit]:
        return self._int_odd_other_qubits

    def default_layout(self) -> 'ZigZagLayout':
        return ZigZagLayout(size=self.size)

    def text_diagram(self, draw_grid_coords: bool = True) -> str:
        return _draw_chains(
            self.up_qubits,
            self.down_qubits,
            self.interaction_even_pairs + self.interaction_odd_pairs,
            draw_grid_coords)

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)


def _find_line_qubits(
        size: int,
        origin: Tuple[int, int],
        rotation: int
) -> Tuple[List[cirq.GridQubit], List[cirq.GridQubit]]:
    if rotation % 90 != 0:
        raise ValueError('Layout rotation must be a multiple of 90 degrees')

    def generate(row, col, drow, dcol) -> List[cirq.GridQubit]:
        return [cirq.GridQubit(row + i * drow, col + i * dcol)
                for i in range(size)]

    rotations = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    drow, dcol = rotations[(rotation % 360) // 90]

    up_row, up_col = origin
    up_qubits = generate(up_row, up_col, drow, dcol)

    down_drow, down_dcol = rotations[((rotation + 270) % 360) // 90]
    down_row, down_col = up_row + down_drow, up_col + down_dcol
    down_qubits = generate(down_row, down_col, drow, dcol)

    return up_qubits, down_qubits


def _find_zigzag_qubits(
        size: int,
        origin: Tuple[int, int],
        rotation: int,
        flipped: bool,
        exchange_chains: bool
) -> Tuple[List[cirq.GridQubit], List[cirq.GridQubit]]:

    if size % 2 == 1:
        raise ValueError(
            'Odd number of sites is not supported for ZigZagLayout')

    if rotation % 90 != 0:
        raise ValueError(
            'ZigZagLayout rotation must be a multiple of 90 degrees')

    # Compute qubits coordinates pinned to (0, 0)
    ref_up = 0, 1
    up_coords = [(ref_up[0] + (i + 1) // 2, ref_up[1] + i // 2)
                 for i in range(size - 1)]

    ref_down = 1, 0
    down_coords = [(ref_down[0] + (i + 1) // 2, ref_down[1] + i // 2)
                   for i in range(size - 1)]

    if flipped:
        up_coords = reversed([(0, 0)] + up_coords)
        down_coords = reversed(down_coords + [(size // 2, size // 2)])
    else:
        up_coords = up_coords + [(size // 2, size // 2)]
        down_coords = [(0, 0)] + down_coords

    # Rotate qubit coordinates if necessary
    rotation = (rotation // 90) % 4
    if rotation == 1:
        up_coords = [(y, -x) for x, y in up_coords]
        down_coords = [(y, -x) for x, y in down_coords]
    elif rotation == 2:
        up_coords = [(-x, -y) for x, y in up_coords]
        down_coords = [(-x, -y) for x, y in down_coords]
    elif rotation == 3:
        up_coords = [(-y, x) for x, y in up_coords]
        down_coords = [(-y, x) for x, y in down_coords]

    # Translate qubit coordinates to pin at origin
    up_coords = [(x + origin[0], y + origin[1])
                 for x, y in up_coords]
    down_coords = [(x + origin[0], y + origin[1])
                   for x, y in down_coords]

    up_qubits = [cirq.GridQubit(*coord) for coord in up_coords]
    down_qubits = [cirq.GridQubit(*coord) for coord in down_coords]

    if exchange_chains:
        up_qubits, down_qubits = down_qubits, up_qubits

    return up_qubits, down_qubits


def _get_even_odd_pairs(qubits: List[cirq.GridQubit],
                        ) -> Tuple[List[Tuple[cirq.GridQubit, cirq.GridQubit]],
                                   List[Tuple[cirq.GridQubit, cirq.GridQubit]]]:
    even_pairs = [(qubits[i + 1], qubits[i])
                  for i in range(0, len(qubits) - 1, 2)]
    odd_pairs = [(qubits[i + 1], qubits[i])
                 for i in range(1, len(qubits) - 1, 2)]
    return even_pairs, odd_pairs


def _get_even_pairs(qubits: List[cirq.GridQubit]
                    ) -> List[Tuple[cirq.GridQubit, cirq.GridQubit]]:
    return [(qubits[i + 1], qubits[i]) for i in range(0, len(qubits) - 1, 2)]


def _get_odd_pairs(qubits: List[cirq.GridQubit]
                   ) -> List[Tuple[cirq.GridQubit, cirq.GridQubit]]:
    return [(qubits[i + 1], qubits[i]) for i in range(1, len(qubits) - 1, 2)]


def _draw_chains(up_qubits: List[cirq.GridQubit],
                 down_qubits: List[cirq.GridQubit],
                 interactions: List[Tuple[cirq.GridQubit, cirq.GridQubit]],
                 draw_grid_coords: bool) -> str:

    def qubit_coords(qubit: cirq.GridQubit) -> Tuple[int, int]:
        return qubit.col - min_col, qubit.row - min_row

    diagram = cirq.TextDiagramDrawer()

    min_col = min(qubit.col for qubit in up_qubits + down_qubits)
    min_row = min(qubit.row for qubit in up_qubits + down_qubits)

    last_col, last_row = None, None

    for index, qubit in enumerate(up_qubits):
        col, row = qubit_coords(qubit)
        label = f'{index + 1}↑'
        if draw_grid_coords:
            label += f' {str(qubit)}'
        diagram.write(col, row, label)
        if index:
            diagram.grid_line(last_col, last_row, col, row)
        last_col, last_row = col, row

    for index, qubit in enumerate(down_qubits):
        col, row = qubit_coords(qubit)
        label = f'{index + 1}↓'
        if draw_grid_coords:
            label += f' {str(qubit)}'
        diagram.write(col, row, label)
        if index:
            diagram.grid_line(last_col, last_row, col, row)
        last_col, last_row = col, row

    for a, b in interactions:
        diagram.grid_line(*qubit_coords(a), *qubit_coords(b), emphasize=True)

    return diagram.render(horizontal_spacing=3 if draw_grid_coords else 2,
                          vertical_spacing=2 if draw_grid_coords else 1,
                          use_unicode_characters=True)


QubitsLayout = Union[LineLayout, ZigZagLayout]
"""Arbitrary qubits layout"""
