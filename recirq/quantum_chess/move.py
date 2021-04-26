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
import recirq.quantum_chess.enums as enums

_ORD_A = ord('a')


def to_rank(x: int) -> str:
    """Returns the algebraic notation rank from the x coordinate."""
    return chr(_ORD_A + x)


def to_square(x: int, y: int) -> str:
    """Returns the algebraic notation of a square."""
    return chr(_ORD_A + x) + str(y + 1)


def x_of(square: str) -> int:
    """Returns x coordinate of an algebraic notation square (e.g. 'f4')."""
    return ord(square[0]) - _ORD_A


def y_of(square: str) -> int:
    """Returns y coordinate of an algebraic notation square (e.g. 'f4')."""
    return int(square[1]) - 1


class Move:
    """Container class that has the source and target of a quantum chess move.

    If the move is a split move, it will have a target2.  If a merge move,
    it will have a source2 attribute.

    For moves that are input from the quantum chess board API, this will
    have a move type and variant that determines what kind of move this is
    (capture, exclusion, etc).
    """
    def __init__(self,
                 source: str,
                 target: str,
                 *,
                 source2: str = None,
                 target2: str = None,
                 move_type: enums.MoveType = None,
                 move_variant: enums.MoveVariant = None,
                 measurement: bool = None):
        self.source = source
        self.source2 = source2
        self.target = target
        self.target2 = target2
        self.move_type = move_type
        self.move_variant = move_variant
        self.measurement = measurement

    def __eq__(self, other):
        if isinstance(other, Move):
            return (self.source == other.source and self.target == other.target
                    and self.target2 == other.target2
                    and self.move_type == other.move_type
                    and self.move_variant == other.move_variant
                    and self.measurement == other.measurement)
        return False

    @classmethod
    def from_string(cls, str_to_parse: str):
        """Creates a move from a string shorthand for tests.


        Format=source,target,target2,source2[.measurement]:type:variant
        with commas omitted.

        if target2 is specified, then source2 should
        be '--'

        Examples:
           'a1a2:JUMP:BASIC'
           'b1a3c3:SPLIT_JUMP:BASIC'
           'b1a3c3.m0:SPLIT_JUMP:BASIC'
           'a3b1--c3:MERGE_JUMP:BASIC'
        """
        fields = str_to_parse.split(':')
        if len(fields) != 3:
            raise ValueError(f'Invalid move string {str_to_parse}')
        source = fields[0][0:2]
        target = fields[0][2:4]

        move_and_measurement = fields[0].split('.', maxsplit=1)
        measurement = None
        if len(move_and_measurement) == 2:
            _, m_str = move_and_measurement
            if m_str != 'm0' and m_str != 'm1':
                raise ValueError(f'Invalid measurement string {m_str}')
            measurement = (m_str == 'm1')

        move_type = enums.MoveType[fields[1]]
        move_variant = enums.MoveVariant[fields[2]]
        if len(fields[0]) <= 4:
            return cls(source,
                       target,
                       move_type=move_type,
                       move_variant=move_variant,
                       measurement=measurement)
        if len(fields[0]) <= 6:
            return cls(source,
                       target,
                       target2=fields[0][4:6],
                       move_type=move_type,
                       move_variant=move_variant,
                       measurement=measurement)
        return cls(source,
                   target,
                   source2=fields[0][6:8],
                   move_type=move_type,
                   move_variant=move_variant,
                   measurement=measurement)

    def is_split_move(self) -> bool:
        return self.target2 is not None

    def is_merge_move(self) -> bool:
        return self.source2 is not None

    def has_measurement(self) -> bool:
        return self.measurement is not None

    def __str__(self):
        movestr = self.source + self.target
        if self.is_split_move():
            movestr = self.source + '^' + self.target + self.target2
        if self.is_merge_move():
            movestr = self.source + self.source2 + '^' + self.target

        if self.has_measurement():
            return movestr + ('.m1' if self.measurement else '.m0')
        else:
            return movestr
