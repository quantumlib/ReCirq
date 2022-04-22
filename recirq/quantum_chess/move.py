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
from recirq.quantum_chess import constants
import recirq.quantum_chess.enums as enums
from typing import Optional

_ORD_A = ord("a")


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

    def __init__(
        self,
        source: str,
        target: str,
        *,
        source2: Optional[str] = None,
        target2: Optional[str] = None,
        move_type: Optional[enums.MoveType] = None,
        move_variant: Optional[enums.MoveVariant] = None,
        promotion_piece: Optional[int] = None,
        measurement: Optional[int] = None,
    ):
        self.source = source
        self.source2 = source2
        self.target = target
        self.target2 = target2
        self.move_type = move_type
        self.move_variant = move_variant
        self.promotion_piece = promotion_piece
        self.measurement = measurement

    def __eq__(self, other):
        if isinstance(other, Move):
            return (
                self.source == other.source
                and self.source2 == other.source2
                and self.target == other.target
                and self.target2 == other.target2
                and self.move_type == other.move_type
                and self.move_variant == other.move_variant
                and self.promotion_piece == other.promotion_piece
                and self.measurement == other.measurement
            )
        return False

    @classmethod
    def from_string(cls, str_to_parse: str):
        """Creates a move from a string shorthand for tests.


        Format=sources_and_targets[.measurement]:type:variant

        where sources_and_targets could be:
            a pair of 2-character square strings concatenated together
            source^t1t2 for split moves with 2 targets
            s1s2^target for merge moves with 2 sources

        Examples:
           'a1a2:JUMP:BASIC'
           'b1^a3c3:SPLIT_JUMP:BASIC'
           'b1^a3c3.m0:SPLIT_JUMP:BASIC'
           'b1^a3c3.m1:SPLIT_JUMP:BASIC'
           'a3b1^c3:MERGE_JUMP:BASIC'
        """
        fields = str_to_parse.split(":")
        if len(fields) != 3:
            raise ValueError(f"Invalid move string {str_to_parse}")

        move_and_measurement = fields[0].split(".", maxsplit=1)
        source_target = move_and_measurement[0]
        measurement = None
        if len(move_and_measurement) == 2:
            _, m_str = move_and_measurement
            if m_str[0] != "m":
                raise ValueError(f"Invalid measurement string {m_str}")
            measurement = int(m_str[1:])

        sources = None
        targets = None
        promotion_piece = None

        if "^" in source_target:
            sources_str, targets_str = source_target.split("^", maxsplit=1)
            sources = [sources_str[i : i + 2] for i in range(0, len(sources_str), 2)]
            targets = [targets_str[i : i + 2] for i in range(0, len(targets_str), 2)]
        else:
            if len(source_target) == 5:
                promotion_piece = constants.REV_PIECES[source_target[4]]
            elif len(source_target) != 4:
                raise ValueError(f"Invalid sources/targets string {source_target}")
            sources = [source_target[0:2]]
            targets = [source_target[2:4]]

        move_type = enums.MoveType[fields[1]]
        move_variant = enums.MoveVariant[fields[2]]
        if len(sources) == 1 and len(targets) == 1:
            return cls(
                sources[0],
                targets[0],
                move_type=move_type,
                move_variant=move_variant,
                promotion_piece=promotion_piece,
                measurement=measurement,
            )
        if len(sources) == 1 and len(targets) == 2:
            return cls(
                sources[0],
                targets[0],
                target2=targets[1],
                move_type=move_type,
                move_variant=move_variant,
                measurement=measurement,
            )
        if len(sources) == 2 and len(targets) == 1:
            return cls(
                sources[0],
                targets[0],
                source2=sources[1],
                move_type=move_type,
                move_variant=move_variant,
                measurement=measurement,
            )
        raise ValueError(
            f"Wrong number of sources {sources} or targets {targets} for {str_to_parse}"
        )

    def is_split_move(self) -> bool:
        return self.target2 is not None

    def is_merge_move(self) -> bool:
        return self.source2 is not None

    def has_measurement(self) -> bool:
        return self.measurement is not None

    def to_string(self, include_type=False) -> str:
        """
        Constructs the string representation of this move object.

        By default, only returns the move source(s), target(s), and measurement
        if present.

        Args:
          include_type: also include the move type/variant in the string
        """
        movestr = self.source + self.target
        if self.is_split_move():
            movestr = self.source + "^" + self.target + self.target2
        if self.is_merge_move():
            movestr = self.source + self.source2 + "^" + self.target
        if self.promotion_piece is not None:
            movestr += constants.PIECES[self.promotion_piece]
        if self.has_measurement():
            movestr += ".m" + str(self.measurement)

        if include_type and self.move_type is not None:
            movestr += ":" + self.move_type.name
        if include_type and self.move_variant is not None:
            movestr += ":" + self.move_variant.name

        return movestr

    def __str__(self):
        return self.to_string()
