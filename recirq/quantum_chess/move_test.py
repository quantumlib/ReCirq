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
from recirq.quantum_chess.move import Move
import recirq.quantum_chess.enums as enums


def test_equality():
    assert Move("a1", "b4") == Move("a1", "b4")
    assert Move(
        source="g3", source2="c3", target="e4", move_type=enums.MoveType.MERGE_JUMP
    ) == Move(
        source="g3", source2="c3", target="e4", move_type=enums.MoveType.MERGE_JUMP
    )
    assert Move(
        source="g3", source2="c3", target="e4", move_type=enums.MoveType.MERGE_JUMP
    ) != Move(
        source="g3", source2="d6", target="e4", move_type=enums.MoveType.MERGE_JUMP
    )
    assert Move("a1", "b4") != Move("a1", "b5")
    assert Move("a1", "b4") != "a1b4"

    assert Move("a1", "b4", measurement=1) == Move("a1", "b4", measurement=1)
    assert Move("a1", "b4", measurement=1) != Move("a1", "b4", measurement=0)
    assert Move("a1", "b4", measurement=1) != Move("a1", "b4")

    assert Move("a1", "b4", move_type=enums.MoveType.JUMP) == Move(
        "a1", "b4", move_type=enums.MoveType.JUMP
    )
    assert Move("a1", "b4", move_type=enums.MoveType.JUMP) != Move(
        "a1", "b4", move_type=enums.MoveType.SLIDE
    )

    assert Move("e2", "e1", promotion_piece=-constants.QUEEN) == Move(
        "e2", "e1", promotion_piece=-constants.QUEEN
    )
    assert Move("e2", "e1", promotion_piece=-constants.QUEEN) != Move(
        "e2", "e1", promotion_piece=-constants.KNIGHT
    )


def test_from_string():
    assert Move.from_string("a1b4:JUMP:BASIC") == Move(
        "a1", "b4", move_type=enums.MoveType.JUMP, move_variant=enums.MoveVariant.BASIC
    )
    assert Move.from_string("a1b4.m0:JUMP:BASIC") == Move(
        "a1",
        "b4",
        move_type=enums.MoveType.JUMP,
        move_variant=enums.MoveVariant.BASIC,
        measurement=0,
    )
    assert Move.from_string("a1b4.m1:JUMP:BASIC") == Move(
        "a1",
        "b4",
        move_type=enums.MoveType.JUMP,
        move_variant=enums.MoveVariant.BASIC,
        measurement=1,
    )
    assert Move.from_string("a1b4.m1:JUMP:BASIC") == Move(
        "a1",
        "b4",
        move_type=enums.MoveType.JUMP,
        move_variant=enums.MoveVariant.BASIC,
        measurement=1,
    )
    assert Move.from_string("c8^g4f5.m0:SPLIT_SLIDE:BASIC") == Move(
        "c8",
        "g4",
        target2="f5",
        move_type=enums.MoveType.SPLIT_SLIDE,
        move_variant=enums.MoveVariant.BASIC,
        measurement=0,
    )

    assert Move.from_string("a1^a4d1:SPLIT_SLIDE:BASIC") == Move(
        "a1",
        "a4",
        target2="d1",
        move_type=enums.MoveType.SPLIT_SLIDE,
        move_variant=enums.MoveVariant.BASIC,
    )
    assert Move.from_string("c3^c4c5:SPLIT_JUMP:BASIC") == Move(
        "c3",
        "c4",
        target2="c5",
        move_type=enums.MoveType.SPLIT_JUMP,
        move_variant=enums.MoveVariant.BASIC,
    )
    assert Move.from_string("a4d4^d1:MERGE_SLIDE:BASIC") == Move(
        "a4",
        "d1",
        source2="d4",
        move_type=enums.MoveType.MERGE_SLIDE,
        move_variant=enums.MoveVariant.BASIC,
    )
    assert Move.from_string("e2f1q:PAWN_CAPTURE:CAPTURE") == Move(
        "e2",
        "f1",
        move_type=enums.MoveType.PAWN_CAPTURE,
        move_variant=enums.MoveVariant.CAPTURE,
        promotion_piece=-constants.QUEEN,
    )


def test_to_string():
    assert str(Move("a1", "b4")) == "a1b4"
    assert str(Move("a1", "b4", target2="c4")) == "a1^b4c4"
    assert str(Move("a1", "b4", source2="b1")) == "a1b1^b4"
    assert str(Move("a1", "b4", measurement=1)) == "a1b4.m1"
    assert str(Move("a1", "b4", source2="b1", measurement=0)) == "a1b1^b4.m0"
    assert (
        str(Move("c7", "c8", measurement=0, promotion_piece=constants.QUEEN))
        == "c7c8Q.m0"
    )
    assert (
        Move(
            "c8",
            "g4",
            target2="f5",
            move_type=enums.MoveType.SPLIT_SLIDE,
            move_variant=enums.MoveVariant.BASIC,
            measurement=0,
        ).to_string(include_type=True)
        == "c8^g4f5.m0:SPLIT_SLIDE:BASIC"
    )


def test_move_round_trip():
    moves = (
        Move(
            "a1",
            "b4",
            move_type=enums.MoveType.JUMP,
            move_variant=enums.MoveVariant.BASIC,
        ),
        Move(
            "a1",
            "b4",
            move_type=enums.MoveType.JUMP,
            move_variant=enums.MoveVariant.BASIC,
            measurement=0,
        ),
        Move(
            "a1",
            "b4",
            move_type=enums.MoveType.JUMP,
            move_variant=enums.MoveVariant.BASIC,
            measurement=1,
        ),
        Move(
            "a1",
            "b4",
            move_type=enums.MoveType.JUMP,
            move_variant=enums.MoveVariant.BASIC,
            measurement=1,
        ),
        Move(
            "c8",
            "g4",
            target2="f5",
            move_type=enums.MoveType.SPLIT_SLIDE,
            move_variant=enums.MoveVariant.BASIC,
            measurement=0,
        ),
    )
    for m in moves:
        assert m == Move.from_string(m.to_string(include_type=True))


def test_string_round_trip():
    move_strings = (
        "a1b4:JUMP:BASIC",
        "a1b4.m0:JUMP:BASIC",
        "a1b4.m1:JUMP:BASIC",
        "a1b4.m1:JUMP:BASIC",
        "c8^g4f5.m0:SPLIT_SLIDE:BASIC",
    )
    for s in move_strings:
        assert s == Move.from_string(s).to_string(include_type=True)
