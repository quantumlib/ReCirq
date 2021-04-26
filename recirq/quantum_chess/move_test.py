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
import recirq.quantum_chess.move as move
import recirq.quantum_chess.enums as enums


def test_equality():
    assert move.Move('a1', 'b4') == move.Move('a1', 'b4')
    assert move.Move('a1', 'b4') != move.Move('a1', 'b5')
    assert move.Move('a1', 'b4') != "a1b4"

    assert move.Move('a1', 'b4',
                     measurement=True) == move.Move('a1',
                                                    'b4',
                                                    measurement=True)
    assert move.Move('a1', 'b4', measurement=True) != move.Move(
        'a1', 'b4', measurement=False)
    assert move.Move('a1', 'b4', measurement=True) != move.Move('a1', 'b4')

    assert move.Move('a1', 'b4', move_type=enums.MoveType.JUMP) == move.Move(
        'a1', 'b4', move_type=enums.MoveType.JUMP)
    assert move.Move('a1', 'b4', move_type=enums.MoveType.JUMP) != move.Move(
        'a1', 'b4', move_type=enums.MoveType.SLIDE)


def test_from_string():
    assert move.Move.from_string('a1b4:JUMP:BASIC') == move.Move(
        'a1',
        'b4',
        move_type=enums.MoveType.JUMP,
        move_variant=enums.MoveVariant.BASIC)
    assert move.Move.from_string('a1b4.m0:JUMP:BASIC') == move.Move(
        'a1',
        'b4',
        move_type=enums.MoveType.JUMP,
        move_variant=enums.MoveVariant.BASIC,
        measurement=False)
    assert move.Move.from_string('a1b4.m1:JUMP:BASIC') == move.Move(
        'a1',
        'b4',
        move_type=enums.MoveType.JUMP,
        move_variant=enums.MoveVariant.BASIC,
        measurement=True)


def test_to_string():
    assert str(move.Move('a1', 'b4')) == 'a1b4'
    assert str(move.Move('a1', 'b4', target2='c4')) == 'a1^b4c4'
    assert str(move.Move('a1', 'b4', source2='b1')) == 'a1b1^b4'
    assert str(move.Move('a1', 'b4', measurement=True)) == 'a1b4.m1'
    assert str(move.Move('a1', 'b4', source2='b1',
                         measurement=False)) == 'a1b1^b4.m0'
