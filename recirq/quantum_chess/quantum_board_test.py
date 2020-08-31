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
import pytest
import cirq
import cirq.contrib.noise_models as ccn

import recirq.engine_utils as utils
import recirq.quantum_chess.enums as enums
import recirq.quantum_chess.move as move
import recirq.quantum_chess.quantum_board as qb
import recirq.quantum_chess.bit_utils as u
from recirq.quantum_chess.test_utils import (
    assert_sample_distribution,
    assert_samples_in,
    assert_this_or_that,
    assert_prob_about,
    assert_fifty_fifty,
)

NOISY_SAMPLER = cirq.DensityMatrixSimulator(noise=ccn.DepolarizingNoiseModel(
    depol_prob=0.004), seed=1234)

BIG_CIRQ_BOARDS = (
    qb.CirqBoard(0, error_mitigation=enums.ErrorMitigation.Error),
    qb.CirqBoard(0,
                 device=utils.get_device_obj_by_name('Syc54-noiseless'),
                 error_mitigation=enums.ErrorMitigation.Error),
)

ALL_CIRQ_BOARDS = BIG_CIRQ_BOARDS + (
    qb.CirqBoard(0,
                 device=utils.get_device_obj_by_name('Syc23-noiseless'),
                 error_mitigation=enums.ErrorMitigation.Error),
    qb.CirqBoard(0,
                 sampler=utils.get_sampler_by_name('Syc23-simulator'),
                 device=utils.get_device_obj_by_name('Syc23-simulator'),
                 error_mitigation=enums.ErrorMitigation.Correct,
                 noise_mitigation=0.10),
)


@pytest.mark.parametrize('board', ALL_CIRQ_BOARDS)
def test_initial_state(board):
    """Tests basic functionality of boards and setting an initial state."""
    b = board.with_state(u.squares_to_bitboard(['a1', 'b1', 'c1']))
    samples = b.sample(100)
    assert len(samples) == 100
    for x in samples:
        assert x == 7
    probs = b.get_probability_distribution(100)
    assert len(probs) == 64
    full_squares = b.get_full_squares_bitboard()
    empty_squares = b.get_empty_squares_bitboard()
    for bit in range(3):
        assert qb.nth_bit_of(bit, full_squares)
        assert not qb.nth_bit_of(bit, empty_squares)
        assert probs[bit] == 1.0
    for bit in range(3, 64):
        assert not qb.nth_bit_of(bit, full_squares)
        assert qb.nth_bit_of(bit, empty_squares)
        assert probs[bit] == 0.0


@pytest.mark.parametrize('board', ALL_CIRQ_BOARDS)
def test_classical_jump_move(board):
    """Tests a jump move in a classical position."""
    b = board.with_state(u.squares_to_bitboard(['a1', 'c1']))
    m = move.Move('a1',
                  'b1',
                  move_type=enums.MoveType.JUMP,
                  move_variant=enums.MoveVariant.BASIC)
    b.do_move(m)
    assert_samples_in(b, [u.squares_to_bitboard(['b1', 'c1'])])


@pytest.mark.parametrize('move_type,board', (
    *[(enums.MoveType.SPLIT_JUMP, b) for b in ALL_CIRQ_BOARDS],
    *[(enums.MoveType.SPLIT_SLIDE, b) for b in ALL_CIRQ_BOARDS],
))
def test_split_move(move_type, board):
    b = board.with_state(u.squares_to_bitboard(['a1']))
    b.do_move(
        move.Move('a1',
                  'a3',
                  target2='c1',
                  move_type=move_type,
                  move_variant=enums.MoveVariant.BASIC))
    samples = b.sample(100)
    assert_this_or_that(samples, u.squares_to_bitboard(['a3']),
                        u.squares_to_bitboard(['c1']))
    probs = b.get_probability_distribution(5000)
    assert_fifty_fifty(probs, qb.square_to_bit('a3'))
    assert_fifty_fifty(probs, qb.square_to_bit('c1'))

    # Test doing a jump after a split move
    m = move.Move('c1',
                  'd1',
                  move_type=enums.MoveType.JUMP,
                  move_variant=enums.MoveVariant.BASIC)
    did_it_move = b.do_move(m)
    samples = b.sample(100)
    assert_this_or_that(samples, u.squares_to_bitboard(['a3']),
                        u.squares_to_bitboard(['d1']))
    probs = b.get_probability_distribution(5000)
    assert_fifty_fifty(probs, u.square_to_bit('a3'))
    assert_fifty_fifty(probs, u.square_to_bit('d1'))


@pytest.mark.parametrize('board', ALL_CIRQ_BOARDS)
def test_split_and_use_same_square(board):
    b = board.with_state(u.squares_to_bitboard(['a1']))
    assert b.perform_moves(
        'a1a2b1:SPLIT_JUMP:BASIC',
        'b1b2:JUMP:BASIC',
        'b2a2:JUMP:BASIC',
    )
    probs = b.get_probability_distribution(5000)
    assert_sample_distribution(b, {
        u.squares_to_bitboard(['a2']): 1 / 2,
        u.squares_to_bitboard(['b2']): 1 / 2
    })


@pytest.mark.parametrize('board', ALL_CIRQ_BOARDS)
def test_exclusion(board):
    """Splits piece b1 to c1 and d1 then tries a excluded move from a1 to c1."""
    b = board.with_state(u.squares_to_bitboard(['a1', 'b1']))
    b.do_move(
        move.Move('b1',
                  'c1',
                  target2='d1',
                  move_type=enums.MoveType.SPLIT_JUMP,
                  move_variant=enums.MoveVariant.BASIC))
    did_it_move = b.do_move(
        move.Move('a1',
                  'c1',
                  move_type=enums.MoveType.JUMP,
                  move_variant=enums.MoveVariant.EXCLUDED))
    samples = b.sample(20)
    if did_it_move:
        expected = u.squares_to_bitboard(['c1', 'd1'])
        assert all(sample == expected for sample in samples)
    else:
        expected = u.squares_to_bitboard(['a1', 'c1'])
        assert all(sample == expected for sample in samples)


@pytest.mark.parametrize('board', ALL_CIRQ_BOARDS)
def test_capture(board):
    """Splits piece from b1 to c1 and d1 then attempts a capture on a1."""
    b = board.with_state(u.squares_to_bitboard(['a1', 'b1']))
    b.do_move(
        move.Move('b1',
                  'c1',
                  target2='d1',
                  move_type=enums.MoveType.SPLIT_JUMP,
                  move_variant=enums.MoveVariant.BASIC))
    did_it_move = b.do_move(
        move.Move('c1',
                  'a1',
                  move_type=enums.MoveType.JUMP,
                  move_variant=enums.MoveVariant.CAPTURE))
    if did_it_move:
        expected = u.squares_to_bitboard(['a1'])
    else:
        expected = u.squares_to_bitboard(['a1', 'd1'])
    assert_samples_in(b, [expected])


@pytest.mark.parametrize('board', ALL_CIRQ_BOARDS)
def test_merge_move(board):
    """Splits piece on a1 to b1 and c1 and then merges back to a1."""
    b = board.with_state(u.squares_to_bitboard(['a1']))
    b.do_move(
        move.Move('a1',
                  'b1',
                  target2='c1',
                  move_type=enums.MoveType.SPLIT_JUMP,
                  move_variant=enums.MoveVariant.BASIC))
    b.do_move(
        move.Move('b1',
                  'a1',
                  source2='c1',
                  move_type=enums.MoveType.MERGE_JUMP,
                  move_variant=enums.MoveVariant.BASIC))
    assert_samples_in(b, [u.squares_to_bitboard(['a1'])])
    assert b.get_full_squares_bitboard() == 1


@pytest.mark.parametrize('board', ALL_CIRQ_BOARDS)
def test_simple_slide_move(board):
    """Tests a basic slide that is totally unblocked."""
    b = board.with_state(u.squares_to_bitboard(['a1']))
    b.do_move(
        move.Move('a1',
                  'd1',
                  move_type=enums.MoveType.SLIDE,
                  move_variant=enums.MoveVariant.BASIC))
    samples = b.sample(10)
    assert all(sample == u.squares_to_bitboard(['d1']) for sample in samples)


@pytest.mark.parametrize('board', ALL_CIRQ_BOARDS)
def test_blocked_slide_move(board):
    """Tests a basic slide that is blocked.

    Slide from a1 to d1 is blocked by a piece on b1.
    """
    b = board.with_state(u.squares_to_bitboard(['a1', 'b1']))
    m = move.Move('a1',
                  'd1',
                  move_type=enums.MoveType.SLIDE,
                  move_variant=enums.MoveVariant.BASIC)
    b.do_move(m)
    samples = b.sample(10)
    expected = u.squares_to_bitboard(['a1', 'b1'])
    assert all(sample == expected for sample in samples)


@pytest.mark.parametrize('board', ALL_CIRQ_BOARDS)
def test_blocked_slide_clear(board):
    """Blocked slide with 100% success

    Tests a piece "blocked" by itself.

    Position: Ra1. Moves: Ra1^a3a4 Ra3a8
    """
    b = board.with_state(u.squares_to_bitboard(['a1']))
    assert b.perform_moves(
        'a1a3a4:SPLIT_SLIDE:BASIC',
        'a3a8:SLIDE:BASIC',
    )
    print(b.circuit)
    samples = b.sample(100)
    possibilities = [
        u.squares_to_bitboard(['a4']),
        u.squares_to_bitboard(['a8']),
    ]
    assert all(sample in possibilities for sample in samples)


@pytest.mark.parametrize('board', ALL_CIRQ_BOARDS)
def test_blocked_slide_blocked(board):
    """Blocked slide with 0% success

    Position: re5, rf5. Moves: rf5d5
    """
    b = board.with_state(u.squares_to_bitboard(['e5', 'f5']))
    m = move.Move('f5',
                  'd5',
                  move_type=enums.MoveType.SLIDE,
                  move_variant=enums.MoveVariant.BASIC)
    b.do_move(m)
    samples = b.sample(100)
    expected = u.squares_to_bitboard(['e5', 'f5'])
    assert all(sample == expected for sample in samples)


def test_blocked_slide_capture_through():
    success = 0
    for trials in range(100):
        b = qb.CirqBoard(u.squares_to_bitboard(['a8', 'c6']))
        b.do_move(
            move.Move('c6',
                      'b8',
                      target2='d8',
                      move_type=enums.MoveType.SPLIT_JUMP,
                      move_variant=enums.MoveVariant.BASIC))
        did_it_move = b.do_move(
            move.Move('a8',
                      'd8',
                      move_type=enums.MoveType.SLIDE,
                      move_variant=enums.MoveVariant.CAPTURE))
        if did_it_move:
            success += 1
    assert success > 25
    assert success < 75


# Works on all boards but is really slow
@pytest.mark.parametrize('board', BIG_CIRQ_BOARDS)
def test_superposition_slide_move(board):
    """Tests a basic slide through a superposition.

    Sets up superposition of c1 and f1 with a slide from a1 to d1.

    Valid end state should be a1 and c1 (blocked), state = 5, or
    d1 and f1 (unblocked), state = 40.
    """
    b = board.with_state(u.squares_to_bitboard(['a1', 'e1']))
    b.do_move(
        move.Move('e1',
                  'f1',
                  target2='c1',
                  move_type=enums.MoveType.SPLIT_JUMP,
                  move_variant=enums.MoveVariant.BASIC))
    b.do_move(
        move.Move('a1',
                  'd1',
                  move_type=enums.MoveType.SLIDE,
                  move_variant=enums.MoveVariant.BASIC))
    blocked = u.squares_to_bitboard(['a1', 'c1'])
    moved = u.squares_to_bitboard(['d1', 'f1'])
    assert_samples_in(b, [blocked, moved])
    probs = b.get_probability_distribution(5000)
    assert_fifty_fifty(probs, 0)
    assert_fifty_fifty(probs, 2)
    assert_fifty_fifty(probs, 3)
    assert_fifty_fifty(probs, 5)


@pytest.mark.parametrize('board', BIG_CIRQ_BOARDS)
def test_superposition_slide_move2(board):
    """Tests a basic slide through a superposition of two pieces.

    Splits b3 and c3 to b2/b1 and c2/c1 then slides a1 to d1.
    """
    b = board.with_state(u.squares_to_bitboard(['a1', 'b3', 'c3']))
    assert b.perform_moves(
        'b3b2b1:SPLIT_JUMP:BASIC',
        'c3c2c1:SPLIT_JUMP:BASIC',
        'a1e1:SLIDE:BASIC',
    )
    possibilities = [
        u.squares_to_bitboard(['a1', 'b1', 'c1']),
        u.squares_to_bitboard(['a1', 'b1', 'c2']),
        u.squares_to_bitboard(['a1', 'b2', 'c1']),
        u.squares_to_bitboard(['e1', 'b2', 'c2'])
    ]
    samples = b.sample(100)
    assert (all(sample in possibilities for sample in samples))
    probs = b.get_probability_distribution(10000)
    assert_fifty_fifty(probs, u.square_to_bit('b2'))
    assert_fifty_fifty(probs, u.square_to_bit('b1'))
    assert_fifty_fifty(probs, u.square_to_bit('c2'))
    assert_fifty_fifty(probs, u.square_to_bit('c1'))
    assert_prob_about(probs, u.square_to_bit('a1'), 0.75)
    assert_prob_about(probs, u.square_to_bit('e1'), 0.25)


@pytest.mark.parametrize('board', BIG_CIRQ_BOARDS)
def test_excluded_slide(board):
    """Test excluded slide.

    Slides from a1 to c1.  b1 will block path in superposition
    and c1 will be blocked/excluded in superposition.
    """
    b = board.with_state(u.squares_to_bitboard(['a1', 'b2', 'c2']))
    did_it_move = b.perform_moves(
        'b2b1a2:SPLIT_JUMP:BASIC',
        'c2c1d2:SPLIT_JUMP:BASIC',
        'a1c1:SLIDE:EXCLUDED',
    )
    samples = b.sample(100)

    if did_it_move:
        possibilities = [
            u.squares_to_bitboard(['c1', 'a2', 'd2']),
            u.squares_to_bitboard(['a1', 'b1', 'd2']),
        ]
        assert (all(sample in possibilities for sample in samples))
    else:
        possibilities = [
            u.squares_to_bitboard(['a1', 'b1', 'c1']),
            u.squares_to_bitboard(['a1', 'a2', 'c1']),
        ]
        assert (all(sample in possibilities for sample in samples))


@pytest.mark.parametrize('board', BIG_CIRQ_BOARDS)
def test_capture_slide(board):
    """Tests a capture slide move.

    Splits from a1 to b1/c1 then tries to capture a piece on c3.
    Will test most cases, since c1, c2, and c3 will all be in
    superposition.
    """
    b = board.with_state(u.squares_to_bitboard(['a1', 'a2', 'a3']))
    did_it_move = b.perform_moves(
        'a1b1c1:SPLIT_JUMP:BASIC',
        'a2b2c2:SPLIT_JUMP:BASIC',
        'a3b3c3:SPLIT_JUMP:BASIC',
        'c1c3:SLIDE:CAPTURE',
    )
    samples = b.sample(1000)
    if did_it_move:
        possibilities = [
            u.squares_to_bitboard(['c3', 'b2']),
            u.squares_to_bitboard(['c3', 'b2', 'b3']),
        ]
        assert (all(sample in possibilities for sample in samples))
    else:
        possibilities = [
            u.squares_to_bitboard(['c1', 'c2', 'b3']),
            u.squares_to_bitboard(['c1', 'c2', 'c3']),
            u.squares_to_bitboard(['b1', 'b2', 'b3']),
            u.squares_to_bitboard(['b1', 'b2', 'c3']),
            u.squares_to_bitboard(['b1', 'c2', 'b3']),
            u.squares_to_bitboard(['b1', 'c2', 'c3']),
        ]
        assert (all(sample in possibilities for sample in samples))


@pytest.mark.parametrize('board', BIG_CIRQ_BOARDS)
def test_split_one_slide(board):
    """Tests a split slide with one blocked path.

    a1 will split to a3 and c1 with square a2 blocked in superposition.
    """
    b = board.with_state(u.squares_to_bitboard(['a1', 'b2']))
    assert b.perform_moves(
        'b2a2c2:SPLIT_JUMP:BASIC',
        'a1a3c1:SPLIT_SLIDE:BASIC',
    )
    samples = b.sample(100)
    possibilities = [
        u.squares_to_bitboard(['c1', 'a2']),
        u.squares_to_bitboard(['a3', 'c2']),
        u.squares_to_bitboard(['c1', 'c2']),
    ]
    assert (all(sample in possibilities for sample in samples))
    probs = b.get_probability_distribution(10000)
    assert_fifty_fifty(probs, qb.square_to_bit('a2'))
    assert_fifty_fifty(probs, qb.square_to_bit('c2'))
    assert_prob_about(probs, qb.square_to_bit('a3'), 0.25)
    assert_prob_about(probs, qb.square_to_bit('c1'), 0.75)


@pytest.mark.parametrize('board', BIG_CIRQ_BOARDS)
def test_split_both_sides(board):
    """ Tests a split slide move where both paths of the slide
    are blocked in superposition.

    The piece will split from a1 to a5/e1.
    The squares c1 and d1 will block one side of the path in superposition.
    The square a3 will be blocked on the other path.

    This will create a lop-sided distribution to test multi-square paths.
    """
    b = board.with_state(u.squares_to_bitboard(['a1', 'b3', 'c3', 'd3']))
    assert b.perform_moves(
        'b3a3b4:SPLIT_JUMP:BASIC',
        'c3c2c1:SPLIT_JUMP:BASIC',
        'd3d2d1:SPLIT_JUMP:BASIC',
        'a1a5e1:SPLIT_SLIDE:BASIC',
    )
    samples = b.sample(1000)
    assert len(samples) == 1000
    possibilities = [
        #                 a1/a5/e1 a3/b4 c1/c2 d1/d2
        u.squares_to_bitboard(['a1', 'a3', 'c1', 'd1']),
        u.squares_to_bitboard(['a1', 'a3', 'c1', 'd2']),
        u.squares_to_bitboard(['a1', 'a3', 'c2', 'd1']),
        u.squares_to_bitboard(['e1', 'a3', 'c2', 'd2']),
        u.squares_to_bitboard(['a5', 'b4', 'c1', 'd1']),
        u.squares_to_bitboard(['a5', 'b4', 'c1', 'd2']),
        u.squares_to_bitboard(['a5', 'b4', 'c2', 'd1']),
        u.squares_to_bitboard(['a5', 'b4', 'c2', 'd2']),
        u.squares_to_bitboard(['e1', 'b4', 'c2', 'd2']),
    ]
    assert (all(sample in possibilities for sample in samples))
    probs = b.get_probability_distribution(25000)
    assert_fifty_fifty(probs, qb.square_to_bit('a3'))
    assert_fifty_fifty(probs, qb.square_to_bit('b4'))
    assert_fifty_fifty(probs, qb.square_to_bit('c1'))
    assert_fifty_fifty(probs, qb.square_to_bit('c2'))
    assert_fifty_fifty(probs, qb.square_to_bit('d1'))
    assert_fifty_fifty(probs, qb.square_to_bit('d2'))
    assert_prob_about(probs, qb.square_to_bit('a1'), 0.375)
    assert_prob_about(probs, qb.square_to_bit('e1'), 0.1875)
    assert_prob_about(probs, qb.square_to_bit('a5'), 0.4375)


@pytest.mark.parametrize('board', BIG_CIRQ_BOARDS)
def test_merge_slide_one_side(board):
    """Tests merge slide.

    Splits a1 to a4 and d1 and then merges to d4.
    The square c4 will block one path of the merge in superposition.
    """
    b = board.with_state(u.squares_to_bitboard(['a1', 'c3']))
    assert b.perform_moves(
        'a1a4d1:SPLIT_SLIDE:BASIC',
        'c3c4c5:SPLIT_JUMP:BASIC',
        'a4d4--d1:MERGE_SLIDE:BASIC',
    )
    possibilities = [
        u.squares_to_bitboard(['a4', 'c4']),
        u.squares_to_bitboard(['d4', 'c4']),
        u.squares_to_bitboard(['d4', 'c5']),
    ]
    assert_samples_in(b, possibilities)
    probs = b.get_probability_distribution(20000)
    assert_fifty_fifty(probs, qb.square_to_bit('c4'))
    assert_fifty_fifty(probs, qb.square_to_bit('c5'))
    assert_prob_about(probs, qb.square_to_bit('a4'), 0.25)
    assert_prob_about(probs, qb.square_to_bit('d4'), 0.75)


@pytest.mark.parametrize('board', BIG_CIRQ_BOARDS)
def test_merge_slide_both_side(board):
    """Tests a merge slide where both paths are blocked.

    Splits a1 to a4/d1 and merges back to d4. c4 and d3 will
    block one square on each path.
    """
    b = board.with_state(u.squares_to_bitboard(['a1', 'c2', 'c3']))
    assert b.perform_moves(
        'a1a4d1:SPLIT_SLIDE:BASIC',
        'c3c4c5:SPLIT_JUMP:BASIC',
        'c2d2e2:SPLIT_JUMP:BASIC',
        'a4d4--d1:MERGE_SLIDE:BASIC',
    )
    possibilities = [
        u.squares_to_bitboard(['a4', 'd2', 'c4']),
        u.squares_to_bitboard(['d1', 'd2', 'c4']),
        u.squares_to_bitboard(['a4', 'e2', 'c4']),
        u.squares_to_bitboard(['d4', 'e2', 'c4']),
        u.squares_to_bitboard(['d1', 'd2', 'c5']),
        u.squares_to_bitboard(['d4', 'd2', 'c5']),
        u.squares_to_bitboard(['d4', 'e2', 'c5']),
    ]
    assert_samples_in(b, possibilities)
    probs = b.get_probability_distribution(20000)
    assert_fifty_fifty(probs, qb.square_to_bit('c4'))
    assert_fifty_fifty(probs, qb.square_to_bit('c5'))
    assert_fifty_fifty(probs, qb.square_to_bit('d2'))
    assert_fifty_fifty(probs, qb.square_to_bit('e2'))
    assert_fifty_fifty(probs, qb.square_to_bit('d4'))
    assert_prob_about(probs, qb.square_to_bit('a4'), 0.25)
    assert_prob_about(probs, qb.square_to_bit('d1'), 0.25)


@pytest.mark.parametrize('board', ALL_CIRQ_BOARDS)
def test_unentangled_pawn_capture(board):
    """Classical pawn capture."""
    b = board.with_state(u.squares_to_bitboard(['a4', 'b3', 'c3']))
    m = move.Move('b3',
                  'a4',
                  move_type=enums.MoveType.PAWN_CAPTURE,
                  move_variant=enums.MoveVariant.BASIC)
    assert b.do_move(m)
    assert_samples_in(b, [u.squares_to_bitboard(['a4', 'c3'])])


@pytest.mark.parametrize('board', BIG_CIRQ_BOARDS)
def test_pawn_capture(board):
    """Tests pawn capture with entanglement.

    Rook on a3 => a4/a5 and rook on c3 => c4/c5.
    Pawn on b3 attempts to capture both rooks.
    The first capture should put the pawn in super-position,
    and the second should force a measurement.
    """
    b = board.with_state(u.squares_to_bitboard(['a3', 'b3', 'c3']))
    # Capture and put the pawn in superposition
    assert b.perform_moves('a3a4a5:SPLIT_JUMP:BASIC', 'b3a4:PAWN_CAPTURE:BASIC')
    possibilities = [
        u.squares_to_bitboard(['a5', 'b3', 'c3']),
        u.squares_to_bitboard(['a4', 'c3']),
    ]
    assert_samples_in(b, possibilities)
    probs = b.get_probability_distribution(5000)
    assert_fifty_fifty(probs, qb.square_to_bit('a5'))
    assert_fifty_fifty(probs, qb.square_to_bit('a4'))
    assert_fifty_fifty(probs, qb.square_to_bit('b3'))

    did_it_move = b.perform_moves('c3c4c5:SPLIT_JUMP:BASIC',
                                  'b3c4:PAWN_CAPTURE:BASIC')
    if did_it_move:
        possibilities = [
            u.squares_to_bitboard(['a5', 'b3', 'c5']),
            u.squares_to_bitboard(['a5', 'c4']),
        ]
    else:
        possibilities = [
            u.squares_to_bitboard(['a4', 'c4']),
            u.squares_to_bitboard(['a4', 'c5']),
        ]
    assert_samples_in(b, possibilities)


@pytest.mark.parametrize('initial_board,source,target', (
    (u.squares_to_bitboard(['e1', 'f1', 'h1']), 'e1', 'g1'),
    (u.squares_to_bitboard(['e1', 'g1', 'h1']), 'e1', 'g1'),
    (u.squares_to_bitboard(['e1', 'f1', 'g1', 'h1']), 'e1', 'g1'),
    (u.squares_to_bitboard(['e8', 'f8', 'h8']), 'e8', 'g8'),
    (u.squares_to_bitboard(['e8', 'g8', 'h8']), 'e8', 'g8'),
    (u.squares_to_bitboard(['e8', 'f8', 'g8', 'h8']), 'e8', 'g8'),
    (u.squares_to_bitboard(['e1', 'c1', 'a1']), 'e1', 'c1'),
    (u.squares_to_bitboard(['e1', 'd1', 'a1']), 'e1', 'c1'),
    (u.squares_to_bitboard(['e1', 'c1', 'd1', 'a1']), 'e1', 'c1'),
))
def test_illegal_castle(initial_board, source, target):
    """Tests various combinations of illegal capture.

    Args:
        initial_board: bitboard to set up
        source: king to move (should be e1 or e8)
        target: square to move king to.
    """
    b = qb.CirqBoard(initial_board)
    if target in ['g1', 'g8']:
        move_type = move_type = enums.MoveType.KS_CASTLE
    else:
        move_type = move_type = enums.MoveType.QS_CASTLE
    m = move.Move(source,
                  target,
                  move_type=move_type,
                  move_variant=enums.MoveVariant.BASIC)
    assert not b.do_move(m)
    assert_samples_in(b, [initial_board])


@pytest.mark.parametrize('board', ALL_CIRQ_BOARDS)
def test_unblocked_black_castle(board):
    """Tests classical kingside black castling move."""
    b = board.with_state(u.squares_to_bitboard(['e8', 'h8']))
    m = move.Move('e8',
                  'g8',
                  move_type=enums.MoveType.KS_CASTLE,
                  move_variant=enums.MoveVariant.BASIC)
    assert b.do_move(m)
    assert_samples_in(b, [u.squares_to_bitboard(['f8', 'g8'])])


@pytest.mark.parametrize('board', ALL_CIRQ_BOARDS)
def test_unblocked_white_castle(board):
    """Tests classical kingside white castling move."""
    b = board.with_state(u.squares_to_bitboard(['e1', 'h1']))
    m = move.Move('e1',
                  'g1',
                  move_type=enums.MoveType.KS_CASTLE,
                  move_variant=enums.MoveVariant.BASIC)
    assert b.do_move(m)
    assert_samples_in(b, [u.squares_to_bitboard(['f1', 'g1'])])


@pytest.mark.parametrize('board', ALL_CIRQ_BOARDS)
def test_unblocked_whitequeenside_castle(board):
    """Tests classical queenside white castling move."""
    b = board.with_state(u.squares_to_bitboard(['e1', 'a1']))
    m = move.Move('e1',
                  'c1',
                  move_type=enums.MoveType.QS_CASTLE,
                  move_variant=enums.MoveVariant.BASIC)
    assert b.do_move(m)
    assert_samples_in(b, [u.squares_to_bitboard(['d1', 'c1'])])


@pytest.mark.parametrize('board', ALL_CIRQ_BOARDS)
def test_unblocked_blackqueenside_castle(board):
    """Tests classical queenside black castling move."""
    b = board.with_state(u.squares_to_bitboard(['e8', 'a8']))
    m = move.Move('e8',
                  'c8',
                  move_type=enums.MoveType.QS_CASTLE,
                  move_variant=enums.MoveVariant.BASIC)
    assert b.do_move(m)
    assert_samples_in(b, [u.squares_to_bitboard(['d8', 'c8'])])


@pytest.mark.parametrize('board', BIG_CIRQ_BOARDS)
def test_2block_ks_castle(board):
    """Kingside castling move blocked by 2 pieces in superposition."""
    b = board.with_state(u.squares_to_bitboard(['e8', 'f6', 'g6', 'h8']))
    did_it_move = b.perform_moves(
        'f6f7f8:SPLIT_JUMP:BASIC',
        'g6g7g8:SPLIT_JUMP:BASIC',
        'e8g8:KS_CASTLE:BASIC',
    )
    if did_it_move:
        possibilities = [u.squares_to_bitboard(['g8', 'f7', 'g7', 'f8'])]
    else:
        possibilities = [
            u.squares_to_bitboard(['e8', 'f8', 'g7', 'h8']),
            u.squares_to_bitboard(['e8', 'f7', 'g8', 'h8']),
            u.squares_to_bitboard(['e8', 'f8', 'g8', 'h8']),
        ]
    assert_samples_in(b, possibilities)


@pytest.mark.parametrize('board', ALL_CIRQ_BOARDS)
def test_1block_ks_castle(board):
    """Kingside castling move blocked by 1 piece in superposition."""
    b = board.with_state(u.squares_to_bitboard(['e1', 'f3', 'h1']))
    b.do_move(
        move.Move('f3',
                  'f2',
                  target2='f1',
                  move_type=enums.MoveType.SPLIT_JUMP,
                  move_variant=enums.MoveVariant.BASIC))
    did_it_move = b.do_move(
        move.Move('e1',
                  'g1',
                  move_type=enums.MoveType.KS_CASTLE,
                  move_variant=enums.MoveVariant.BASIC))
    if did_it_move:
        expected = u.squares_to_bitboard(['f1', 'f2', 'g1'])
    else:
        expected = u.squares_to_bitboard(['e1', 'f1', 'h1'])
    assert_samples_in(b, [expected])


@pytest.mark.parametrize('board', ALL_CIRQ_BOARDS)
def test_entangled_qs_castle(board):
    """Queenside castling move with b-square blocked by superposition.

    This should entangle the castling rook/king with the piece.
    """
    b = board.with_state(u.squares_to_bitboard(['e1', 'b3', 'a1']))
    b.do_move(
        move.Move('b3',
                  'b2',
                  target2='b1',
                  move_type=enums.MoveType.SPLIT_SLIDE,
                  move_variant=enums.MoveVariant.BASIC))
    assert b.do_move(
        move.Move('e1',
                  'c1',
                  move_type=enums.MoveType.QS_CASTLE,
                  move_variant=enums.MoveVariant.BASIC))
    possibilities = [
        u.squares_to_bitboard(['a1', 'b1', 'e1']),
        u.squares_to_bitboard(['c1', 'b2', 'd1'])
    ]
    assert_samples_in(b, possibilities)


@pytest.mark.parametrize('board', BIG_CIRQ_BOARDS)
def test_entangled_qs_castle2(board):
    """Queenside castling move with all intervening squares blocked."""
    b = board.with_state(u.squares_to_bitboard(['e1', 'b3', 'c3', 'd3', 'a1']))
    did_it_move = b.perform_moves(
        'b3b2b1:SPLIT_JUMP:BASIC',
        'c3c2c1:SPLIT_JUMP:BASIC',
        'd3d2d1:SPLIT_JUMP:BASIC',
        'e1c1:QS_CASTLE:BASIC',
    )
    if did_it_move:
        possibilities = [
            u.squares_to_bitboard(['a1', 'b1', 'e1', 'c2', 'd2']),
            u.squares_to_bitboard(['c1', 'b2', 'd1', 'c2', 'd2']),
        ]
    else:
        possibilities = [
            u.squares_to_bitboard(['a1', 'b1', 'e1', 'c1', 'd2']),
            u.squares_to_bitboard(['a1', 'b2', 'e1', 'c1', 'd2']),
            u.squares_to_bitboard(['a1', 'b1', 'e1', 'c2', 'd1']),
            u.squares_to_bitboard(['a1', 'b2', 'e1', 'c2', 'd1']),
            u.squares_to_bitboard(['a1', 'b1', 'e1', 'c1', 'd1']),
            u.squares_to_bitboard(['a1', 'b2', 'e1', 'c1', 'd1']),
        ]
    assert_samples_in(b, possibilities)


@pytest.mark.parametrize('board', ALL_CIRQ_BOARDS)
def test_classical_ep(board):
    """Fully classical en passant."""
    b = board.with_state(u.squares_to_bitboard(['e5', 'd5']))
    m = move.Move('e5',
                  'd6',
                  move_type=enums.MoveType.PAWN_EP,
                  move_variant=enums.MoveVariant.BASIC)
    b.do_move(m)
    assert_samples_in(b, [u.squares_to_bitboard(['d6'])])


# Seems to be problematic on real device
def test_capture_ep():
    """Tests capture en passant.

    Splits b8 to a6/c6.  Capture a6 with b5 pawn to put the source pawn
    into superposition.  Then move c6 out of the way.

    Finally, move c7 to c5 and perform en passant on b6.
    """
    b = qb.CirqBoard(u.squares_to_bitboard(['b8', 'b5', 'c7']))
    did_it_move = b.perform_moves(
        'b8a6c6:SPLIT_JUMP:BASIC',
        'b5a6:PAWN_CAPTURE:BASIC',
        'c7c5:PAWN_TWO_STEP:BASIC',
        'b5c6:PAWN_EP:CAPTURE',
    )
    if did_it_move:
        expected = u.squares_to_bitboard(['c6', 'c7'])
    else:
        expected = u.squares_to_bitboard(['a6', 'c5'])
    assert_samples_in(b, [expected])


def test_capture_ep2():
    """Tests capture en passant.

    Splits b8 to a6/c6.  Capture a6 with b5 pawn to put the source pawn
    into superposition.

    Finally, move c7 to c5 and perform en passant on b6.  This should
    either capture the knight or the pawn.
    """
    b = qb.CirqBoard(u.squares_to_bitboard(['b8', 'b5', 'c7']))
    did_it_move = b.perform_moves(
        'b8a6c6:SPLIT_JUMP:BASIC',
        'c7c5:PAWN_TWO_STEP:BASIC',
        'b5c6:PAWN_EP:CAPTURE',
    )
    assert did_it_move
    possibilities = [
        u.squares_to_bitboard(['c6', 'c7']),
        u.squares_to_bitboard(['c6', 'a6']),
    ]
    assert_samples_in(b, possibilities)


def test_blocked_ep():
    """Tests blocked en passant.

    Splits c4 to b6 and d6.  Moves a pawn through the piece on d6.
    Attempts to en passant the d-pawn using blocked e.p.
    """
    b = qb.CirqBoard(u.squares_to_bitboard(['c4', 'c5', 'd7']))
    did_it_move = b.perform_moves(
        'c4d6b6:SPLIT_JUMP:BASIC',
        'd7d5:PAWN_TWO_STEP:BASIC',
        'c5d6:PAWN_EP:EXCLUDED',
    )
    if did_it_move:
        possibilities = [
            u.squares_to_bitboard(['b6', 'd6']),
        ]
    else:
        possibilities = [
            u.squares_to_bitboard(['c5', 'd6', 'd7']),
        ]
    assert_samples_in(b, possibilities)


def test_basic_ep():
    b = qb.CirqBoard(u.squares_to_bitboard(['b8', 'b5', 'c7']))
    assert b.perform_moves(
        'b8a6c6:SPLIT_JUMP:BASIC',
        'b5a6:PAWN_CAPTURE:BASIC',
        'c6b8:JUMP:BASIC',
        'c7c5:PAWN_TWO_STEP:BASIC',
        'b5c6:PAWN_EP:BASIC',
    )
    possibilities = [
        u.squares_to_bitboard(['b8', 'c6']),
        u.squares_to_bitboard(['a6', 'c5']),
    ]
    assert_samples_in(b, possibilities)
