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
import os
import random

import cirq
import numpy as np
import pytest

import recirq.engine_utils as utils
import recirq.quantum_chess.bit_utils as u
import recirq.quantum_chess.enums as enums
import recirq.quantum_chess.move as move
import recirq.quantum_chess.quantum_board as qb
from recirq.quantum_chess.test_utils import (
    assert_sample_distribution,
    assert_samples_in,
    assert_this_or_that,
    assert_prob_about,
    assert_fifty_fifty,
)
from recirq.quantum_chess.bit_utils import bit_to_qubit, square_to_bit, nth_bit_of
from recirq.quantum_chess.caching_utils import CacheKey

# The number of samples needed to avoid caching previous repetitions.
# Choosing this to be above the sample size of is_classical(), which is 1000, will avoid caching.
CACHE_INVALIDATION_REPS = 1001


def get_seed():
    seed = os.environ.get("RECIRQ_CHESS_TEST_SEED")
    if seed:
        seed = int(seed)
    else:
        seed = random.randrange(2**32)
    print("Using seed", seed)
    return seed


# Boards


def syc23_noisy(state):
    return qb.CirqBoard(
        state,
        sampler=cirq.DensityMatrixSimulator(
            noise=cirq.ConstantQubitNoiseModel(
                qubit_noise_gate=cirq.DepolarizingChannel(0.005)
            ),
            seed=get_seed(),
        ),
        device=utils.get_device_obj_by_name("Syc23-simulator"),
        error_mitigation=enums.ErrorMitigation.Correct,
        noise_mitigation=0.10,
    )


def syc23_noiseless(state):
    np.random.seed(get_seed())
    return qb.CirqBoard(
        state,
        device=utils.get_device_obj_by_name("Syc23-noiseless"),
        error_mitigation=enums.ErrorMitigation.Error,
    )


def syc54_noiseless(state):
    np.random.seed(get_seed())
    return qb.CirqBoard(
        state,
        device=utils.get_device_obj_by_name("Syc54-noiseless"),
        error_mitigation=enums.ErrorMitigation.Error,
    )


def simulator(state):
    np.random.seed(get_seed())
    return qb.CirqBoard(state, error_mitigation=enums.ErrorMitigation.Error)


BIG_CIRQ_BOARDS = (
    simulator,
    syc54_noiseless,
)

ALL_CIRQ_BOARDS = BIG_CIRQ_BOARDS + (
    syc23_noiseless,
    syc23_noisy,
)


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_initial_state(board):
    """Tests basic functionality of boards and setting an initial state."""
    b = board(u.squares_to_bitboard(["a1", "b1", "c1"]))
    samples = b.sample(100)
    assert len(samples) == 100
    for x in samples:
        assert x == 7
    probs = b.get_probability_distribution(100)
    assert len(probs) == 64
    board_probs = b.get_board_probability_distribution(100)
    assert len(board_probs) == 1
    assert board_probs[7] == 1.0
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


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_classical_jump_move(board):
    """Tests a jump move in a classical position."""
    b = board(u.squares_to_bitboard(["a1", "c1"]))
    m = move.Move(
        "a1", "b1", move_type=enums.MoveType.JUMP, move_variant=enums.MoveVariant.BASIC
    )
    b.do_move(m)
    assert_samples_in(b, [u.squares_to_bitboard(["b1", "c1"])])


def test_path_qubits():
    """Source and target should be in the same line, otherwise ValueError should be returned."""
    b = qb.CirqBoard(u.squares_to_bitboard(["a1", "b3", "c4", "d5", "e6", "f7"]))
    assert b.path_qubits("b3", "f7") == [
        bit_to_qubit(square_to_bit("c4")),
        bit_to_qubit(square_to_bit("d5")),
        bit_to_qubit(square_to_bit("e6")),
    ]
    with pytest.raises(ValueError):
        b.path_qubits("a1", "b3")
    with pytest.raises(ValueError):
        b.path_qubits("c4", "a1")


@pytest.mark.parametrize(
    "move_type,board",
    (
        *[(enums.MoveType.SPLIT_JUMP, b) for b in BIG_CIRQ_BOARDS],
        *[(enums.MoveType.SPLIT_SLIDE, b) for b in BIG_CIRQ_BOARDS],
    ),
)
def test_split_move(move_type, board):
    b = board(u.squares_to_bitboard(["a1"]))
    b.do_move(
        move.Move(
            "a1",
            "a3",
            target2="c1",
            move_type=move_type,
            move_variant=enums.MoveVariant.BASIC,
        )
    )
    samples = b.sample(100)
    assert_this_or_that(
        samples, u.squares_to_bitboard(["a3"]), u.squares_to_bitboard(["c1"])
    )
    probs = b.get_probability_distribution(5000)
    assert_fifty_fifty(probs, qb.square_to_bit("a3"))
    assert_fifty_fifty(probs, qb.square_to_bit("c1"))
    board_probs = b.get_board_probability_distribution(5000)
    assert len(board_probs) == 2
    assert_fifty_fifty(board_probs, u.squares_to_bitboard(["a3"]))
    assert_fifty_fifty(board_probs, u.squares_to_bitboard(["c1"]))

    # Test doing a jump after a split move
    m = move.Move(
        "c1", "d1", move_type=enums.MoveType.JUMP, move_variant=enums.MoveVariant.BASIC
    )
    assert b.do_move(m)
    samples = b.sample(100)
    assert_this_or_that(
        samples, u.squares_to_bitboard(["a3"]), u.squares_to_bitboard(["d1"])
    )
    probs = b.get_probability_distribution(5000)
    assert_fifty_fifty(probs, u.square_to_bit("a3"))
    assert_fifty_fifty(probs, u.square_to_bit("d1"))
    board_probs = b.get_board_probability_distribution(5000)
    assert len(board_probs) == 2
    assert_fifty_fifty(board_probs, u.squares_to_bitboard(["a3"]))
    assert_fifty_fifty(board_probs, u.squares_to_bitboard(["d1"]))


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_split_and_use_same_square(board):
    b = board(u.squares_to_bitboard(["a1"]))
    assert b.perform_moves(
        "a1^a2b1:SPLIT_JUMP:BASIC",
        "b1b2:JUMP:BASIC",
        "b2a2:JUMP:BASIC",
    )
    assert_sample_distribution(
        b, {u.squares_to_bitboard(["a2"]): 1 / 2, u.squares_to_bitboard(["b2"]): 1 / 2}
    )
    board_probs = b.get_board_probability_distribution(5000)
    assert len(board_probs) == 2
    assert_fifty_fifty(board_probs, u.squares_to_bitboard(["a2"]))
    assert_fifty_fifty(board_probs, u.squares_to_bitboard(["b2"]))


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_overlapping_splits(board):
    b = board(u.squares_to_bitboard(["e1", "g1"]))
    assert b.perform_moves(
        "e1^d3f3:SPLIT_JUMP:BASIC",
        "g1^h3f3:SPLIT_JUMP:BASIC",
    )
    assert_sample_distribution(
        b,
        {
            u.squares_to_bitboard(["d3", "f3"]): 1 / 4,
            u.squares_to_bitboard(["d3", "h3"]): 1 / 4,
            u.squares_to_bitboard(["f3", "g1"]): 1 / 4,
            u.squares_to_bitboard(["h3", "g1"]): 1 / 4,
        },
    )


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_exclusion(board):
    """Splits piece b1 to c1 and d1 then tries a excluded move from a1 to c1."""
    b = board(u.squares_to_bitboard(["a1", "b1"]))
    b.do_move(
        move.Move(
            "b1",
            "c1",
            target2="d1",
            move_type=enums.MoveType.SPLIT_JUMP,
            move_variant=enums.MoveVariant.BASIC,
        )
    )
    did_it_move = b.do_move(
        move.Move(
            "a1",
            "c1",
            move_type=enums.MoveType.JUMP,
            move_variant=enums.MoveVariant.EXCLUDED,
        )
    )
    samples = b.sample(20)
    if did_it_move:
        expected = u.squares_to_bitboard(["c1", "d1"])
        assert all(sample == expected for sample in samples)
    else:
        expected = u.squares_to_bitboard(["a1", "c1"])
        assert all(sample == expected for sample in samples)


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_capture(board):
    """Splits piece from b1 to c1 and d1 then attempts a capture on a1."""
    b = board(u.squares_to_bitboard(["a1", "b1"]))
    b.do_move(
        move.Move(
            "b1",
            "c1",
            target2="d1",
            move_type=enums.MoveType.SPLIT_JUMP,
            move_variant=enums.MoveVariant.BASIC,
        )
    )
    did_it_move = b.do_move(
        move.Move(
            "c1",
            "a1",
            move_type=enums.MoveType.JUMP,
            move_variant=enums.MoveVariant.CAPTURE,
        )
    )
    if did_it_move:
        expected = u.squares_to_bitboard(["a1"])
    else:
        expected = u.squares_to_bitboard(["a1", "d1"])
    assert_samples_in(b, [expected])


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_merge_move(board):
    """Splits piece on a1 to b1 and c1 and then merges back to a1."""
    b = board(u.squares_to_bitboard(["a1"]))
    b.do_move(
        move.Move(
            "a1",
            "b1",
            target2="c1",
            move_type=enums.MoveType.SPLIT_JUMP,
            move_variant=enums.MoveVariant.BASIC,
        )
    )
    b.do_move(
        move.Move(
            "b1",
            "a1",
            source2="c1",
            move_type=enums.MoveType.MERGE_JUMP,
            move_variant=enums.MoveVariant.BASIC,
        )
    )
    assert_samples_in(b, [u.squares_to_bitboard(["a1"])])
    assert b.get_full_squares_bitboard() == 1


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_simple_slide_move(board):
    """Tests a basic slide that is totally unblocked."""
    b = board(u.squares_to_bitboard(["a1"]))
    b.do_move(
        move.Move(
            "a1",
            "d1",
            move_type=enums.MoveType.SLIDE,
            move_variant=enums.MoveVariant.BASIC,
        )
    )
    samples = b.sample(10)
    assert all(sample == u.squares_to_bitboard(["d1"]) for sample in samples)


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_blocked_slide_move(board):
    """Tests a basic slide that is blocked.

    Slide from a1 to d1 is blocked by a piece on b1.
    """
    b = board(u.squares_to_bitboard(["a1", "b1"]))
    m = move.Move(
        "a1", "d1", move_type=enums.MoveType.SLIDE, move_variant=enums.MoveVariant.BASIC
    )
    b.do_move(m)
    samples = b.sample(10)
    expected = u.squares_to_bitboard(["a1", "b1"])
    assert all(sample == expected for sample in samples)


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_blocked_slide_clear(board):
    """Blocked slide with 100% success

    Tests a piece "blocked" by itself.

    Position: Ra1. Moves: Ra1^a3a4 Ra3a8
    """
    b = board(u.squares_to_bitboard(["a1"]))
    assert b.perform_moves(
        "a1^a3a4:SPLIT_SLIDE:BASIC",
        "a3a8:SLIDE:BASIC",
    )
    samples = b.sample(100)
    possibilities = [
        u.squares_to_bitboard(["a4"]),
        u.squares_to_bitboard(["a8"]),
    ]
    assert all(sample in possibilities for sample in samples)


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_blocked_slide_blocked(board):
    """Blocked slide with 0% success

    Position: re5, rf5. Moves: rf5d5
    """
    b = board(u.squares_to_bitboard(["e5", "f5"]))
    m = move.Move(
        "f5", "d5", move_type=enums.MoveType.SLIDE, move_variant=enums.MoveVariant.BASIC
    )
    b.do_move(m)
    samples = b.sample(100)
    expected = u.squares_to_bitboard(["e5", "f5"])
    assert all(sample == expected for sample in samples)


def test_blocked_slide_capture_through():
    success = 0
    b = simulator(0)
    for _ in range(100):
        b.with_state(u.squares_to_bitboard(["a8", "c6"]))
        b.do_move(
            move.Move(
                "c6",
                "b8",
                target2="d8",
                move_type=enums.MoveType.SPLIT_JUMP,
                move_variant=enums.MoveVariant.BASIC,
            )
        )
        did_it_move = b.do_move(
            move.Move(
                "a8",
                "d8",
                move_type=enums.MoveType.SLIDE,
                move_variant=enums.MoveVariant.CAPTURE,
            )
        )
        if did_it_move:
            success += 1
    assert success > 25
    assert success < 75


def test_blocked_slide_capture_force_success():
    b = simulator(0)
    b.with_state(u.squares_to_bitboard(["f7", "f2", "g1"]))
    did_it_move = b.perform_moves("g1^f3h3:SPLIT_JUMP:BASIC", "f7f2.m1:SLIDE:CAPTURE")
    assert did_it_move
    samples = b.sample(100)
    for sample in samples:
        assert sample == u.squares_to_bitboard(["f2", "h3"])


def test_blocked_slide_capture_force_failure():
    b = simulator(0)
    b.with_state(u.squares_to_bitboard(["f7", "f2", "g1"]))
    did_it_move = b.perform_moves("g1^f3h3:SPLIT_JUMP:BASIC", "f7f2.m0:SLIDE:CAPTURE")
    assert not did_it_move
    samples = b.sample(100)
    for sample in samples:
        assert sample == u.squares_to_bitboard(["f7", "f3", "f2"])


# Works on all boards but is really slow
@pytest.mark.parametrize("board", BIG_CIRQ_BOARDS)
def test_superposition_slide_move(board):
    """Tests a basic slide through a superposition.

    Sets up superposition of c1 and f1 with a slide from a1 to d1.

    Valid end state should be a1 and c1 (blocked), state = 5, or
    d1 and f1 (unblocked), state = 40.
    """
    b = board(u.squares_to_bitboard(["a1", "e1"]))
    b.do_move(
        move.Move(
            "e1",
            "f1",
            target2="c1",
            move_type=enums.MoveType.SPLIT_JUMP,
            move_variant=enums.MoveVariant.BASIC,
        )
    )
    b.do_move(
        move.Move(
            "a1",
            "d1",
            move_type=enums.MoveType.SLIDE,
            move_variant=enums.MoveVariant.BASIC,
        )
    )
    blocked = u.squares_to_bitboard(["a1", "c1"])
    moved = u.squares_to_bitboard(["d1", "f1"])
    assert_samples_in(b, [blocked, moved])
    probs = b.get_probability_distribution(5000)
    assert_fifty_fifty(probs, 0)
    assert_fifty_fifty(probs, 2)
    assert_fifty_fifty(probs, 3)
    assert_fifty_fifty(probs, 5)
    board_probs = b.get_board_probability_distribution(5000)
    assert len(board_probs) == 2
    assert_fifty_fifty(board_probs, blocked)
    assert_fifty_fifty(board_probs, moved)


def test_superposition_slide_move2():
    """Tests a basic slide through a superposition of two pieces.

    Splits b3 and c3 to b2/b1 and c2/c1 then slides a1 to d1.
    """
    b = simulator(u.squares_to_bitboard(["a1", "b3", "c3"]))
    assert b.perform_moves(
        "b3^b2b1:SPLIT_JUMP:BASIC",
        "c3^c2c1:SPLIT_JUMP:BASIC",
        "a1e1:SLIDE:BASIC",
    )
    possibilities = [
        u.squares_to_bitboard(["a1", "b1", "c1"]),
        u.squares_to_bitboard(["a1", "b1", "c2"]),
        u.squares_to_bitboard(["a1", "b2", "c1"]),
        u.squares_to_bitboard(["e1", "b2", "c2"]),
    ]
    samples = b.sample(100)
    assert all(sample in possibilities for sample in samples)
    probs = b.get_probability_distribution(10000)
    assert_fifty_fifty(probs, u.square_to_bit("b2"))
    assert_fifty_fifty(probs, u.square_to_bit("b1"))
    assert_fifty_fifty(probs, u.square_to_bit("c2"))
    assert_fifty_fifty(probs, u.square_to_bit("c1"))
    assert_prob_about(probs, u.square_to_bit("a1"), 0.75)
    assert_prob_about(probs, u.square_to_bit("e1"), 0.25)
    board_probs = b.get_board_probability_distribution(10000)
    assert len(board_probs) == len(possibilities)
    for possibility in possibilities:
        assert_prob_about(board_probs, possibility, 0.25)


def test_slide_with_two_path_qubits_coherence():
    """Tests that a path ancilla does not mess up split/merge coherence.

    Position: Qd1, Bf1, Ng1.
    See https://github.com/quantumlib/ReCirq/issues/193.
    """
    b = simulator(u.squares_to_bitboard(["d1", "f1", "g1"]))
    assert b.perform_moves(
        "g1^h3f3:SPLIT_JUMP:BASIC",
        "f1^e2b5:SPLIT_SLIDE:BASIC",
        "d1h5:SLIDE:BASIC",
        "d1h5:SLIDE:BASIC",
        "d1h5:SLIDE:BASIC",
        "d1h5:SLIDE:BASIC",
        "h3f3^g1:MERGE_JUMP:BASIC",
    )
    assert_sample_distribution(
        b,
        {
            u.squares_to_bitboard(["d1", "b5", "g1"]): 1 / 2,
            u.squares_to_bitboard(["d1", "e2", "g1"]): 1 / 2,
        },
    )


def test_split_slide_merge_slide_coherence():
    b = simulator(u.squares_to_bitboard(["b4", "d3"]))
    assert b.perform_moves(
        "d3^c5e5:SPLIT_JUMP:BASIC",
        "b4^b8e7:SPLIT_SLIDE:BASIC",
        "b8e7^b4:MERGE_SLIDE:BASIC",
        "c5e5^d3:MERGE_JUMP:BASIC",
    )
    assert_samples_in(b, [u.squares_to_bitboard(["b4", "d3"])])


def test_split_slide_zero_one():
    b = simulator(u.squares_to_bitboard(["a1", "d3"]))
    assert b.perform_moves(
        "d3^c3d1:SPLIT_SLIDE:BASIC",
        "a1^a4f1:SPLIT_SLIDE:BASIC",
    )
    assert_sample_distribution(
        b,
        {
            u.squares_to_bitboard(["d1", "a4"]): 1 / 2,
            u.squares_to_bitboard(["c3", "f1"]): 1 / 4,
            u.squares_to_bitboard(["c3", "a4"]): 1 / 4,
        },
    )


def test_split_slide_zero_multiple():
    b = simulator(u.squares_to_bitboard(["a1", "d3", "f2"]))
    assert b.perform_moves(
        "d3^c3d1:SPLIT_SLIDE:BASIC",
        "f2^e2f1:SPLIT_SLIDE:BASIC",
        "a1^a5g1:SPLIT_SLIDE:BASIC",
    )
    assert_sample_distribution(
        b,
        {
            u.squares_to_bitboard(["c3", "e2", "a5"]): 1 / 8,
            u.squares_to_bitboard(["c3", "e2", "g1"]): 1 / 8,
            u.squares_to_bitboard(["c3", "f1", "a5"]): 1 / 4,
            u.squares_to_bitboard(["d1", "e2", "a5"]): 1 / 4,
            u.squares_to_bitboard(["d1", "f1", "a5"]): 1 / 4,
        },
    )


def test_split_slide_multiple_zero():
    b = simulator(u.squares_to_bitboard(["a1", "d3", "f2"]))
    assert b.perform_moves(
        "d3^c3d1:SPLIT_SLIDE:BASIC",
        "f2^e2f1:SPLIT_SLIDE:BASIC",
        "a1^g1a5:SPLIT_SLIDE:BASIC",
    )
    assert_sample_distribution(
        b,
        {
            u.squares_to_bitboard(["c3", "e2", "a5"]): 1 / 8,
            u.squares_to_bitboard(["c3", "e2", "g1"]): 1 / 8,
            u.squares_to_bitboard(["c3", "f1", "a5"]): 1 / 4,
            u.squares_to_bitboard(["d1", "e2", "a5"]): 1 / 4,
            u.squares_to_bitboard(["d1", "f1", "a5"]): 1 / 4,
        },
    )


def test_split_slide_one_one_same_qubit():
    b = simulator(u.squares_to_bitboard(["a1", "d3"]))
    assert b.perform_moves(
        "d3^c3d1:SPLIT_SLIDE:BASIC",
        "a1^e1f1:SPLIT_SLIDE:BASIC",
    )
    assert_sample_distribution(
        b,
        {
            u.squares_to_bitboard(["d1", "a1"]): 1 / 2,
            u.squares_to_bitboard(["c3", "e1"]): 1 / 4,
            u.squares_to_bitboard(["c3", "f1"]): 1 / 4,
        },
    )


def test_split_slide_one_one_diff_qubits():
    b = simulator(u.squares_to_bitboard(["c1", "d3"]))
    assert b.perform_moves(
        "d3^c3d1:SPLIT_SLIDE:BASIC",
        "c1^c5f1:SPLIT_SLIDE:BASIC",
    )
    assert_sample_distribution(
        b,
        {
            u.squares_to_bitboard(["f1", "c3"]): 1 / 2,
            u.squares_to_bitboard(["c5", "d1"]): 1 / 2,
        },
    )


def test_split_slide_one_multiple():
    b = simulator(u.squares_to_bitboard(["a1", "d3", "f2"]))
    assert b.perform_moves(
        "d3^c3d1:SPLIT_SLIDE:BASIC",
        "f2^e2f1:SPLIT_SLIDE:BASIC",
        "a1^e1g1:SPLIT_SLIDE:BASIC",
    )
    assert_sample_distribution(
        b,
        {
            u.squares_to_bitboard(["c3", "e2", "e1"]): 1 / 8,
            u.squares_to_bitboard(["c3", "e2", "g1"]): 1 / 8,
            u.squares_to_bitboard(["c3", "f1", "e1"]): 1 / 4,
            u.squares_to_bitboard(["d1", "e2", "a1"]): 1 / 4,
            u.squares_to_bitboard(["d1", "f1", "a1"]): 1 / 4,
        },
    )


def test_split_slide_multiple_one():
    b = simulator(u.squares_to_bitboard(["a1", "d3", "f2"]))
    assert b.perform_moves(
        "d3^c3d1:SPLIT_SLIDE:BASIC",
        "f2^e2f1:SPLIT_SLIDE:BASIC",
        "a1^g1e1:SPLIT_SLIDE:BASIC",
    )
    assert_sample_distribution(
        b,
        {
            u.squares_to_bitboard(["c3", "e2", "e1"]): 1 / 8,
            u.squares_to_bitboard(["c3", "e2", "g1"]): 1 / 8,
            u.squares_to_bitboard(["c3", "f1", "e1"]): 1 / 4,
            u.squares_to_bitboard(["d1", "e2", "a1"]): 1 / 4,
            u.squares_to_bitboard(["d1", "f1", "a1"]): 1 / 4,
        },
    )


def test_split_slide_multiple_multiple():
    b = simulator(u.squares_to_bitboard(["a1", "d3", "f2"]))
    assert b.perform_moves(
        "d3^a3d1:SPLIT_SLIDE:BASIC",
        "f2^a2f1:SPLIT_SLIDE:BASIC",
        "a1^g1a5:SPLIT_SLIDE:BASIC",
    )
    assert_sample_distribution(
        b,
        {
            u.squares_to_bitboard(["a3", "a2", "g1"]): 1 / 4,
            u.squares_to_bitboard(["a3", "f1", "a1"]): 1 / 4,
            u.squares_to_bitboard(["d1", "a2", "a1"]): 1 / 4,
            u.squares_to_bitboard(["d1", "f1", "a5"]): 1 / 4,
        },
    )


@pytest.mark.parametrize("board", BIG_CIRQ_BOARDS)
def test_excluded_slide(board):
    """Test excluded slide.

    Slides from a1 to c1.  b1 will block path in superposition
    and c1 will be blocked/excluded in superposition.
    """
    b = board(u.squares_to_bitboard(["a1", "b2", "c2"]))
    did_it_move = b.perform_moves(
        "b2^b1a2:SPLIT_JUMP:BASIC",
        "c2^c1d2:SPLIT_JUMP:BASIC",
        "a1c1:SLIDE:EXCLUDED",
    )
    samples = b.sample(100)

    if did_it_move:
        possibilities = [
            u.squares_to_bitboard(["c1", "a2", "d2"]),
            u.squares_to_bitboard(["a1", "b1", "d2"]),
        ]
        assert all(sample in possibilities for sample in samples)
    else:
        possibilities = [
            u.squares_to_bitboard(["a1", "b1", "c1"]),
            u.squares_to_bitboard(["a1", "a2", "c1"]),
        ]
        assert all(sample in possibilities for sample in samples)


@pytest.mark.parametrize("board", BIG_CIRQ_BOARDS)
def test_capture_slide(board):
    """Tests a capture slide move.

    Splits from a1 to b1/c1 then tries to capture a piece on c3.
    Will test most cases, since c1, c2, and c3 will all be in
    superposition.
    """
    b = board(u.squares_to_bitboard(["a1", "a2", "a3"]))
    did_it_move = b.perform_moves(
        "a1^b1c1:SPLIT_JUMP:BASIC",
        "a2^b2c2:SPLIT_JUMP:BASIC",
        "a3^b3c3:SPLIT_JUMP:BASIC",
        "c1c3:SLIDE:CAPTURE",
    )
    samples = b.sample(1000)
    if did_it_move:
        possibilities = [
            u.squares_to_bitboard(["c3", "b2"]),
            u.squares_to_bitboard(["c3", "b2", "b3"]),
        ]
        assert all(sample in possibilities for sample in samples)
    else:
        possibilities = [
            u.squares_to_bitboard(["c1", "c2", "b3"]),
            u.squares_to_bitboard(["c1", "c2", "c3"]),
            u.squares_to_bitboard(["b1", "b2", "b3"]),
            u.squares_to_bitboard(["b1", "b2", "c3"]),
            u.squares_to_bitboard(["b1", "c2", "b3"]),
            u.squares_to_bitboard(["b1", "c2", "c3"]),
        ]
        assert all(sample in possibilities for sample in samples)


def test_split_one_slide():
    """Tests a split slide with one blocked path.

    a1 will split to a3 and c1 with square a2 blocked in superposition.
    """
    b = simulator(u.squares_to_bitboard(["a1", "b2"]))
    assert b.perform_moves(
        "b2^a2c2:SPLIT_JUMP:BASIC",
        "a1^a3c1:SPLIT_SLIDE:BASIC",
    )
    samples = b.sample(100)
    possibilities = [
        u.squares_to_bitboard(["c1", "a2"]),
        u.squares_to_bitboard(["a3", "c2"]),
        u.squares_to_bitboard(["c1", "c2"]),
    ]
    assert all(sample in possibilities for sample in samples)
    probs = b.get_probability_distribution(10000)
    assert_fifty_fifty(probs, qb.square_to_bit("a2"))
    assert_fifty_fifty(probs, qb.square_to_bit("c2"))
    assert_prob_about(probs, qb.square_to_bit("a3"), 0.25)
    assert_prob_about(probs, qb.square_to_bit("c1"), 0.75)
    board_probs = b.get_board_probability_distribution(10000)
    assert len(board_probs) == len(possibilities)
    assert_prob_about(board_probs, possibilities[0], 0.5)
    assert_prob_about(board_probs, possibilities[1], 0.25)
    assert_prob_about(board_probs, possibilities[2], 0.25)


def test_split_both_sides():
    """Tests a split slide move where both paths of the slide
    are blocked in superposition.

    The piece will split from a1 to a5/e1.
    The squares c1 and d1 will block one side of the path in superposition.
    The square a3 will be blocked on the other path.

    This will create a lop-sided distribution to test multi-square paths.
    """
    b = simulator(u.squares_to_bitboard(["a1", "b3", "c3", "d3"]))
    assert b.perform_moves(
        "b3^a3b4:SPLIT_JUMP:BASIC",
        "c3^c2c1:SPLIT_JUMP:BASIC",
        "d3^d2d1:SPLIT_JUMP:BASIC",
        "a1^a5e1:SPLIT_SLIDE:BASIC",
    )
    samples = b.sample(1000)
    assert len(samples) == 1000
    possibilities = [
        #                 a1/a5/e1 a3/b4 c1/c2 d1/d2
        u.squares_to_bitboard(["a1", "a3", "c1", "d1"]),
        u.squares_to_bitboard(["a1", "a3", "c1", "d2"]),
        u.squares_to_bitboard(["a1", "a3", "c2", "d1"]),
        u.squares_to_bitboard(["e1", "a3", "c2", "d2"]),
        u.squares_to_bitboard(["a5", "b4", "c1", "d1"]),
        u.squares_to_bitboard(["a5", "b4", "c1", "d2"]),
        u.squares_to_bitboard(["a5", "b4", "c2", "d1"]),
        u.squares_to_bitboard(["a5", "b4", "c2", "d2"]),
        u.squares_to_bitboard(["e1", "b4", "c2", "d2"]),
    ]
    assert all(sample in possibilities for sample in samples)
    probs = b.get_probability_distribution(25000)
    assert_fifty_fifty(probs, qb.square_to_bit("a3"))
    assert_fifty_fifty(probs, qb.square_to_bit("b4"))
    assert_fifty_fifty(probs, qb.square_to_bit("c1"))
    assert_fifty_fifty(probs, qb.square_to_bit("c2"))
    assert_fifty_fifty(probs, qb.square_to_bit("d1"))
    assert_fifty_fifty(probs, qb.square_to_bit("d2"))
    assert_prob_about(probs, qb.square_to_bit("a1"), 0.375)
    assert_prob_about(probs, qb.square_to_bit("e1"), 0.1875)
    assert_prob_about(probs, qb.square_to_bit("a5"), 0.4375)
    board_probs = b.get_board_probability_distribution(25000)
    assert len(board_probs) == len(possibilities)
    for possibility in possibilities[:7]:
        assert_prob_about(board_probs, possibility, 0.125)
    for possibility in possibilities[7:]:
        assert_prob_about(board_probs, possibility, 0.0625)


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_split_merge_slide_self_intersecting(board):
    """Tests merge slide with a source square in the path."""
    b = board(u.squares_to_bitboard(["c1"]))
    assert b.perform_moves(
        "c1^e3g5:SPLIT_SLIDE:BASIC",
        "e3g5^d2:MERGE_SLIDE:BASIC",
    )
    assert_samples_in(b, [u.squares_to_bitboard(["d2"])])


def test_merge_slide_one_side():
    """Tests merge slide.

    Splits a1 to a4 and d1 and then merges to d4.
    The square c4 will block one path of the merge in superposition.
    """
    b = simulator(u.squares_to_bitboard(["a1", "c3"]))
    assert b.perform_moves(
        "a1^a4d1:SPLIT_SLIDE:BASIC",
        "c3^c4c5:SPLIT_JUMP:BASIC",
        "a4d1^d4:MERGE_SLIDE:BASIC",
    )
    possibilities = [
        u.squares_to_bitboard(["a4", "c4"]),
        u.squares_to_bitboard(["d4", "c4"]),
        u.squares_to_bitboard(["d4", "c5"]),
    ]
    assert_samples_in(b, possibilities)
    probs = b.get_probability_distribution(20000)
    assert_fifty_fifty(probs, qb.square_to_bit("c4"))
    assert_fifty_fifty(probs, qb.square_to_bit("c5"))
    assert_prob_about(probs, qb.square_to_bit("a4"), 0.25)
    assert_prob_about(probs, qb.square_to_bit("d4"), 0.75)
    board_probs = b.get_board_probability_distribution(20000)
    assert len(board_probs) == len(possibilities)
    assert_prob_about(board_probs, possibilities[0], 0.25)
    assert_prob_about(board_probs, possibilities[1], 0.25)
    assert_prob_about(board_probs, possibilities[2], 0.5)


def test_merge_slide_both_side():
    """Tests a merge slide where both paths are blocked.

    Splits a1 to a4/d1 and merges back to d4. c4 and d3 will
    block one square on each path.
    """
    b = simulator(u.squares_to_bitboard(["a1", "c2", "c3"]))
    assert b.perform_moves(
        "a1^a4d1:SPLIT_SLIDE:BASIC",
        "c3^c4c5:SPLIT_JUMP:BASIC",
        "c2^d2e2:SPLIT_JUMP:BASIC",
        "a4d1^d4:MERGE_SLIDE:BASIC",
    )
    possibilities = [
        u.squares_to_bitboard(["a4", "d2", "c4"]),
        u.squares_to_bitboard(["d1", "d2", "c4"]),
        u.squares_to_bitboard(["a4", "e2", "c4"]),
        u.squares_to_bitboard(["d4", "e2", "c4"]),
        u.squares_to_bitboard(["d1", "d2", "c5"]),
        u.squares_to_bitboard(["d4", "d2", "c5"]),
        u.squares_to_bitboard(["d4", "e2", "c5"]),
    ]
    assert_samples_in(b, possibilities)
    probs = b.get_probability_distribution(20000)
    assert_fifty_fifty(probs, qb.square_to_bit("c4"))
    assert_fifty_fifty(probs, qb.square_to_bit("c5"))
    assert_fifty_fifty(probs, qb.square_to_bit("d2"))
    assert_fifty_fifty(probs, qb.square_to_bit("e2"))
    assert_fifty_fifty(probs, qb.square_to_bit("d4"))
    assert_prob_about(probs, qb.square_to_bit("a4"), 0.25)
    assert_prob_about(probs, qb.square_to_bit("d1"), 0.25)
    board_probs = b.get_board_probability_distribution(20000)
    assert len(board_probs) == len(possibilities)
    for possibility in possibilities[:6]:
        assert_prob_about(board_probs, possibility, 0.125)
    assert_prob_about(board_probs, possibilities[6], 0.25)


def test_merge_slide_zero_one():
    b = simulator(u.squares_to_bitboard(["a1", "d3"]))
    assert b.perform_moves(
        "a1^a4f1:SPLIT_SLIDE:BASIC",
        "d3^c3d1:SPLIT_SLIDE:BASIC",
        "a4f1^a1:MERGE_SLIDE:BASIC",
    )
    assert_sample_distribution(
        b,
        {
            u.squares_to_bitboard(["c3", "a1"]): 1 / 2,
            u.squares_to_bitboard(["d1", "f1"]): 1 / 4,
            u.squares_to_bitboard(["d1", "a1"]): 1 / 4,
        },
    )


def test_merge_slide_zero_multiple():
    b = simulator(u.squares_to_bitboard(["a1", "d3", "f2"]))
    assert b.perform_moves(
        "a1^a5g1:SPLIT_SLIDE:BASIC",
        "d3^c3d1:SPLIT_SLIDE:BASIC",
        "f2^e2f1:SPLIT_SLIDE:BASIC",
        "a5g1^a1:MERGE_SLIDE:BASIC",
    )
    assert_sample_distribution(
        b,
        {
            u.squares_to_bitboard(["c3", "e2", "a1"]): 1 / 4,
            u.squares_to_bitboard(["c3", "f1", "g1"]): 1 / 8,
            u.squares_to_bitboard(["c3", "f1", "a1"]): 1 / 8,
            u.squares_to_bitboard(["d1", "e2", "g1"]): 1 / 8,
            u.squares_to_bitboard(["d1", "e2", "a1"]): 1 / 8,
            u.squares_to_bitboard(["d1", "f1", "g1"]): 1 / 8,
            u.squares_to_bitboard(["d1", "f1", "a1"]): 1 / 8,
        },
    )


def test_merge_slide_multiple_zero():
    b = simulator(u.squares_to_bitboard(["a1", "d3", "f2"]))
    assert b.perform_moves(
        "a1^a5g1:SPLIT_SLIDE:BASIC",
        "d3^c3d1:SPLIT_SLIDE:BASIC",
        "f2^e2f1:SPLIT_SLIDE:BASIC",
        "g1a5^a1:MERGE_SLIDE:BASIC",
    )
    assert_sample_distribution(
        b,
        {
            u.squares_to_bitboard(["c3", "e2", "a1"]): 1 / 4,
            u.squares_to_bitboard(["c3", "f1", "g1"]): 1 / 8,
            u.squares_to_bitboard(["c3", "f1", "a1"]): 1 / 8,
            u.squares_to_bitboard(["d1", "e2", "g1"]): 1 / 8,
            u.squares_to_bitboard(["d1", "e2", "a1"]): 1 / 8,
            u.squares_to_bitboard(["d1", "f1", "g1"]): 1 / 8,
            u.squares_to_bitboard(["d1", "f1", "a1"]): 1 / 8,
        },
    )


def test_merge_slide_one_one_same_qubit():
    b = simulator(u.squares_to_bitboard(["a1", "d3"]))
    assert b.perform_moves(
        "a1^e1f1:SPLIT_SLIDE:BASIC",
        "d3^c3d1:SPLIT_SLIDE:BASIC",
        "e1f1^a1:MERGE_SLIDE:BASIC",
    )
    assert_sample_distribution(
        b,
        {
            u.squares_to_bitboard(["d1", "e1"]): 1 / 4,
            u.squares_to_bitboard(["d1", "f1"]): 1 / 4,
            u.squares_to_bitboard(["c3", "a1"]): 1 / 2,
        },
    )


def test_merge_slide_one_one_diff_qubits():
    b = simulator(u.squares_to_bitboard(["c1", "d3"]))
    assert b.perform_moves(
        "c1^c5f1:SPLIT_SLIDE:BASIC",
        "d3^c3d1:SPLIT_SLIDE:BASIC",
        "c5f1^c1:MERGE_SLIDE:BASIC",
    )
    assert_sample_distribution(
        b,
        {
            u.squares_to_bitboard(["c3", "c5"]): 1 / 4,
            u.squares_to_bitboard(["c3", "c1"]): 1 / 4,
            u.squares_to_bitboard(["d1", "f1"]): 1 / 4,
            u.squares_to_bitboard(["d1", "c1"]): 1 / 4,
        },
    )


def test_merge_slide_one_multiple():
    b = simulator(u.squares_to_bitboard(["a1", "d3", "f2"]))
    assert b.perform_moves(
        "a1^e1g1:SPLIT_SLIDE:BASIC",
        "d3^c3d1:SPLIT_SLIDE:BASIC",
        "f2^e2f1:SPLIT_SLIDE:BASIC",
        "e1g1^a1:MERGE_SLIDE:BASIC",
    )
    assert_sample_distribution(
        b,
        {
            u.squares_to_bitboard(["c3", "e2", "a1"]): 1 / 4,
            u.squares_to_bitboard(["c3", "f1", "g1"]): 1 / 8,
            u.squares_to_bitboard(["c3", "f1", "a1"]): 1 / 8,
            u.squares_to_bitboard(["d1", "e2", "e1"]): 1 / 8,
            u.squares_to_bitboard(["d1", "e2", "g1"]): 1 / 8,
            u.squares_to_bitboard(["d1", "f1", "e1"]): 1 / 8,
            u.squares_to_bitboard(["d1", "f1", "g1"]): 1 / 8,
        },
    )


def test_merge_slide_multiple_one():
    b = simulator(u.squares_to_bitboard(["a1", "d3", "f2"]))
    assert b.perform_moves(
        "a1^e1g1:SPLIT_SLIDE:BASIC",
        "d3^c3d1:SPLIT_SLIDE:BASIC",
        "f2^e2f1:SPLIT_SLIDE:BASIC",
        "g1e1^a1:MERGE_SLIDE:BASIC",
    )
    assert_sample_distribution(
        b,
        {
            u.squares_to_bitboard(["c3", "e2", "a1"]): 1 / 4,
            u.squares_to_bitboard(["c3", "f1", "g1"]): 1 / 8,
            u.squares_to_bitboard(["c3", "f1", "a1"]): 1 / 8,
            u.squares_to_bitboard(["d1", "e2", "e1"]): 1 / 8,
            u.squares_to_bitboard(["d1", "e2", "g1"]): 1 / 8,
            u.squares_to_bitboard(["d1", "f1", "e1"]): 1 / 8,
            u.squares_to_bitboard(["d1", "f1", "g1"]): 1 / 8,
        },
    )


def test_merge_slide_multiple_multiple():
    b = simulator(u.squares_to_bitboard(["a1", "d3", "f2"]))
    assert b.perform_moves(
        "a1^a5g1:SPLIT_SLIDE:BASIC",
        "d3^a3d1:SPLIT_SLIDE:BASIC",
        "f2^a2f1:SPLIT_SLIDE:BASIC",
        "a5g1^a1:MERGE_SLIDE:BASIC",
    )
    assert_sample_distribution(
        b,
        {
            u.squares_to_bitboard(["a3", "a2", "a1"]): 1 / 8,
            u.squares_to_bitboard(["a3", "a2", "a5"]): 1 / 8,
            u.squares_to_bitboard(["a3", "f1", "g1"]): 1 / 8,
            u.squares_to_bitboard(["a3", "f1", "a5"]): 1 / 8,
            u.squares_to_bitboard(["d1", "a2", "a5"]): 1 / 8,
            u.squares_to_bitboard(["d1", "a2", "g1"]): 1 / 8,
            u.squares_to_bitboard(["d1", "f1", "a1"]): 1 / 8,
            u.squares_to_bitboard(["d1", "f1", "g1"]): 1 / 8,
        },
    )


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_unentangled_pawn_capture(board):
    """Classical pawn capture."""
    b = board(u.squares_to_bitboard(["a4", "b3", "c3"]))
    m = move.Move(
        "b3",
        "a4",
        move_type=enums.MoveType.PAWN_CAPTURE,
        move_variant=enums.MoveVariant.BASIC,
    )
    assert b.do_move(m)
    assert_samples_in(b, [u.squares_to_bitboard(["a4", "c3"])])


def test_pawn_capture():
    """Tests pawn capture with entanglement.

    Rook on a3 => a4/a5 and rook on c3 => c4/c5.
    Pawn on b3 attempts to capture both rooks.
    The first capture should put the pawn in super-position,
    and the second should force a measurement.
    """
    b = simulator(u.squares_to_bitboard(["a3", "b3", "c3"]))
    # Capture and put the pawn in superposition
    assert b.perform_moves("a3^a4a5:SPLIT_JUMP:BASIC", "b3a4:PAWN_CAPTURE:BASIC")
    possibilities = [
        u.squares_to_bitboard(["a5", "b3", "c3"]),
        u.squares_to_bitboard(["a4", "c3"]),
    ]
    assert_samples_in(b, possibilities)
    probs = b.get_probability_distribution(5000)
    assert_fifty_fifty(probs, qb.square_to_bit("a5"))
    assert_fifty_fifty(probs, qb.square_to_bit("a4"))
    assert_fifty_fifty(probs, qb.square_to_bit("b3"))
    board_probs = b.get_board_probability_distribution(5000)
    assert len(board_probs) == len(possibilities)
    assert_fifty_fifty(board_probs, possibilities[0])
    assert_fifty_fifty(board_probs, possibilities[1])

    did_it_move = b.perform_moves("c3^c4c5:SPLIT_JUMP:BASIC", "b3c4:PAWN_CAPTURE:BASIC")
    if did_it_move:
        possibilities = [
            u.squares_to_bitboard(["a5", "b3", "c5"]),
            u.squares_to_bitboard(["a5", "c4"]),
        ]
    else:
        possibilities = [
            u.squares_to_bitboard(["a4", "c4"]),
            u.squares_to_bitboard(["a4", "c5"]),
        ]
    assert_samples_in(b, possibilities)
    board_probs = b.get_board_probability_distribution(5000)
    assert len(board_probs) == 2
    assert_fifty_fifty(board_probs, possibilities[0])
    assert_fifty_fifty(board_probs, possibilities[1])


@pytest.mark.parametrize("board", BIG_CIRQ_BOARDS)
def test_pawn_capture_bits(board):
    """Tests correctly setting the classical bits with pawn capture.

    White: Nb4
    Black: Pa7, Pb7
    """
    b = board(u.squares_to_bitboard(["c2", "a6", "b7"]))
    assert b.perform_moves(
        "c2^b4a1:SPLIT_JUMP:BASIC", "b4a6.m1:JUMP:CAPTURE", "b7a6:PAWN_CAPTURE:CAPTURE"
    )
    assert_samples_in(b, [u.squares_to_bitboard(["a6"])])


@pytest.mark.parametrize(
    "initial_board,source,target",
    (
        (u.squares_to_bitboard(["e1", "f1", "h1"]), "e1", "g1"),
        (u.squares_to_bitboard(["e1", "g1", "h1"]), "e1", "g1"),
        (u.squares_to_bitboard(["e1", "f1", "g1", "h1"]), "e1", "g1"),
        (u.squares_to_bitboard(["e8", "f8", "h8"]), "e8", "g8"),
        (u.squares_to_bitboard(["e8", "g8", "h8"]), "e8", "g8"),
        (u.squares_to_bitboard(["e8", "f8", "g8", "h8"]), "e8", "g8"),
        (u.squares_to_bitboard(["e1", "b1", "a1"]), "e1", "c1"),
        (u.squares_to_bitboard(["e1", "c1", "a1"]), "e1", "c1"),
        (u.squares_to_bitboard(["e1", "d1", "a1"]), "e1", "c1"),
        (u.squares_to_bitboard(["e1", "b1", "c1", "d1", "a1"]), "e1", "c1"),
    ),
)
def test_illegal_castle(initial_board, source, target):
    """Tests various combinations of illegal capture.

    Args:
        initial_board: bitboard to set up
        source: king to move (should be e1 or e8)
        target: square to move king to.
    """
    b = simulator(initial_board)
    if target in ["g1", "g8"]:
        move_type = move_type = enums.MoveType.KS_CASTLE
    else:
        move_type = move_type = enums.MoveType.QS_CASTLE
    m = move.Move(
        source, target, move_type=move_type, move_variant=enums.MoveVariant.BASIC
    )
    assert not b.do_move(m)
    assert_samples_in(b, [initial_board])


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_unblocked_black_castle(board):
    """Tests classical kingside black castling move."""
    b = board(u.squares_to_bitboard(["e8", "h8"]))
    m = move.Move(
        "e8",
        "g8",
        move_type=enums.MoveType.KS_CASTLE,
        move_variant=enums.MoveVariant.BASIC,
    )
    assert b.do_move(m)
    assert_samples_in(b, [u.squares_to_bitboard(["f8", "g8"])])


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_unblocked_white_castle(board):
    """Tests classical kingside white castling move."""
    b = board(u.squares_to_bitboard(["e1", "h1"]))
    m = move.Move(
        "e1",
        "g1",
        move_type=enums.MoveType.KS_CASTLE,
        move_variant=enums.MoveVariant.BASIC,
    )
    assert b.do_move(m)
    assert_samples_in(b, [u.squares_to_bitboard(["f1", "g1"])])


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_unblocked_whitequeenside_castle(board):
    """Tests classical queenside white castling move."""
    b = board(u.squares_to_bitboard(["e1", "a1"]))
    m = move.Move(
        "e1",
        "c1",
        move_type=enums.MoveType.QS_CASTLE,
        move_variant=enums.MoveVariant.BASIC,
    )
    assert b.do_move(m)
    assert_samples_in(b, [u.squares_to_bitboard(["d1", "c1"])])


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_unblocked_blackqueenside_castle(board):
    """Tests classical queenside black castling move."""
    b = board(u.squares_to_bitboard(["e8", "a8"]))
    m = move.Move(
        "e8",
        "c8",
        move_type=enums.MoveType.QS_CASTLE,
        move_variant=enums.MoveVariant.BASIC,
    )
    assert b.do_move(m)
    assert_samples_in(b, [u.squares_to_bitboard(["d8", "c8"])])


@pytest.mark.parametrize("board", BIG_CIRQ_BOARDS)
def test_2block_ks_castle(board):
    """Kingside castling move blocked by 2 pieces in superposition."""
    b = board(u.squares_to_bitboard(["e8", "f6", "g6", "h8"]))
    did_it_move = b.perform_moves(
        "f6^f7f8:SPLIT_JUMP:BASIC",
        "g6^g7g8:SPLIT_JUMP:BASIC",
        "e8g8:KS_CASTLE:BASIC",
    )
    if did_it_move:
        possibilities = [u.squares_to_bitboard(["g8", "f7", "g7", "f8"])]
    else:
        possibilities = [
            u.squares_to_bitboard(["e8", "f8", "g7", "h8"]),
            u.squares_to_bitboard(["e8", "f7", "g8", "h8"]),
            u.squares_to_bitboard(["e8", "f8", "g8", "h8"]),
        ]
    assert_samples_in(b, possibilities)


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_1block_ks_castle(board):
    """Kingside castling move blocked by 1 piece in superposition."""
    b = board(u.squares_to_bitboard(["e1", "f3", "h1"]))
    b.do_move(
        move.Move(
            "f3",
            "f2",
            target2="f1",
            move_type=enums.MoveType.SPLIT_JUMP,
            move_variant=enums.MoveVariant.BASIC,
        )
    )
    did_it_move = b.do_move(
        move.Move(
            "e1",
            "g1",
            move_type=enums.MoveType.KS_CASTLE,
            move_variant=enums.MoveVariant.BASIC,
        )
    )
    if did_it_move:
        expected = u.squares_to_bitboard(["f1", "f2", "g1"])
    else:
        expected = u.squares_to_bitboard(["e1", "f1", "h1"])
    assert_samples_in(b, [expected])


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_entangled_qs_castle(board):
    """Queenside castling move with b-square blocked by superposition.

    This should entangle the castling rook/king with the piece.
    """
    b = board(u.squares_to_bitboard(["e1", "b3", "a1"]))
    b.do_move(
        move.Move(
            "b3",
            "b2",
            target2="b1",
            move_type=enums.MoveType.SPLIT_SLIDE,
            move_variant=enums.MoveVariant.BASIC,
        )
    )
    assert b.do_move(
        move.Move(
            "e1",
            "c1",
            move_type=enums.MoveType.QS_CASTLE,
            move_variant=enums.MoveVariant.BASIC,
        )
    )
    possibilities = [
        u.squares_to_bitboard(["a1", "b1", "e1"]),
        u.squares_to_bitboard(["c1", "b2", "d1"]),
    ]
    assert_samples_in(b, possibilities)


@pytest.mark.parametrize("board", BIG_CIRQ_BOARDS)
def test_entangled_qs_castle2(board):
    """Queenside castling move with all intervening squares blocked."""
    b = board(u.squares_to_bitboard(["e1", "b3", "c3", "d3", "a1"]))
    did_it_move = b.perform_moves(
        "b3^b2b1:SPLIT_JUMP:BASIC",
        "c3^c2c1:SPLIT_JUMP:BASIC",
        "d3^d2d1:SPLIT_JUMP:BASIC",
        "e1c1:QS_CASTLE:BASIC",
    )
    if did_it_move:
        possibilities = [
            u.squares_to_bitboard(["a1", "b1", "e1", "c2", "d2"]),
            u.squares_to_bitboard(["c1", "b2", "d1", "c2", "d2"]),
        ]
    else:
        possibilities = [
            u.squares_to_bitboard(["a1", "b1", "e1", "c1", "d2"]),
            u.squares_to_bitboard(["a1", "b2", "e1", "c1", "d2"]),
            u.squares_to_bitboard(["a1", "b1", "e1", "c2", "d1"]),
            u.squares_to_bitboard(["a1", "b2", "e1", "c2", "d1"]),
            u.squares_to_bitboard(["a1", "b1", "e1", "c1", "d1"]),
            u.squares_to_bitboard(["a1", "b2", "e1", "c1", "d1"]),
        ]
    assert_samples_in(b, possibilities)


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_classical_ep(board):
    """Fully classical en passant."""
    b = board(u.squares_to_bitboard(["e5", "d5"]))
    m = move.Move(
        "e5",
        "d6",
        move_type=enums.MoveType.PAWN_EP,
        move_variant=enums.MoveVariant.BASIC,
    )
    b.do_move(m)
    assert_samples_in(b, [u.squares_to_bitboard(["d6"])])


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_classical_ep2(board):
    """Fully classical en passant."""
    b = board(u.squares_to_bitboard(["e4", "d4"]))
    m = move.Move(
        "e4",
        "d3",
        move_type=enums.MoveType.PAWN_EP,
        move_variant=enums.MoveVariant.BASIC,
    )
    b.do_move(m)
    assert_samples_in(b, [u.squares_to_bitboard(["d3"])])


# Seems to be problematic on real device
def test_capture_ep():
    """Tests capture en passant.

    Splits b8 to a6/c6.  Capture a6 with b5 pawn to put the source pawn
    into superposition.

    Finally, move c7 to c5 and perform en passant on c6.
    """
    b = simulator(u.squares_to_bitboard(["b8", "b5", "c7"]))
    did_it_move = b.perform_moves(
        "b8^a6c6:SPLIT_JUMP:BASIC",
        "b5a6:PAWN_CAPTURE:BASIC",
        "c7c5:PAWN_TWO_STEP:BASIC",
        "b5c6:PAWN_EP:CAPTURE",
    )
    if did_it_move:
        expected = u.squares_to_bitboard(["c6", "c7"])
    else:
        expected = u.squares_to_bitboard(["a6", "c5"])
    assert_samples_in(b, [expected])


def test_capture_ep2():
    """Tests capture en passant.

    Splits b8 to a6/c6. Move c7 to c5 and perform en passant on c6.  This should
    either capture the knight or the pawn.
    """
    b = simulator(u.squares_to_bitboard(["b8", "b5", "c7"]))
    did_it_move = b.perform_moves(
        "b8^a6c6:SPLIT_JUMP:BASIC",
        "c7c5:PAWN_TWO_STEP:BASIC",
        "b5c6:PAWN_EP:CAPTURE",
    )
    assert did_it_move
    possibilities = [
        u.squares_to_bitboard(["c6", "c7"]),
        u.squares_to_bitboard(["c6", "a6"]),
    ]
    assert_samples_in(b, possibilities)


def test_blocked_ep():
    """Tests blocked en passant.

    Splits c4 to b6 and d6.  Moves a pawn through the piece on d6.
    Attempts to en passant the d-pawn using blocked e.p.
    """
    b = simulator(u.squares_to_bitboard(["c4", "c5", "d7"]))
    did_it_move = b.perform_moves(
        "c4^d6b6:SPLIT_JUMP:BASIC",
        "d7d5:PAWN_TWO_STEP:BASIC",
        "c5d6:PAWN_EP:EXCLUDED",
    )
    if did_it_move:
        possibilities = [
            u.squares_to_bitboard(["b6", "d6"]),
        ]
    else:
        possibilities = [
            u.squares_to_bitboard(["c5", "d6", "d7"]),
        ]
    assert_samples_in(b, possibilities)


def test_basic_ep():
    b = simulator(u.squares_to_bitboard(["b8", "b5", "c7"]))
    assert b.perform_moves(
        "b8^a6c6:SPLIT_JUMP:BASIC",
        "b5a6:PAWN_CAPTURE:BASIC",
        "c6b8:JUMP:BASIC",
        "c7c5:PAWN_TWO_STEP:BASIC",
        "b5c6:PAWN_EP:BASIC",
    )
    possibilities = [
        u.squares_to_bitboard(["b8", "c6"]),
        u.squares_to_bitboard(["a6", "c5"]),
    ]
    assert_samples_in(b, possibilities)


def test_undo_last_move():
    # TODO  (cantwellc) more comprehensive tests
    b = simulator(u.squares_to_bitboard(["a2"]))
    assert b.perform_moves("a2a4:PAWN_TWO_STEP:BASIC")
    probs = b.get_probability_distribution(1000)
    assert_prob_about(probs, qb.square_to_bit("a4"), 1.0)
    board_probs = b.get_board_probability_distribution(1000)
    assert len(board_probs) == 1
    assert_prob_about(board_probs, u.squares_to_bitboard(["a4"]), 1.0)
    assert b.undo_last_move()
    probs = b.get_probability_distribution(1000)
    assert_prob_about(probs, qb.square_to_bit("a2"), 1.0)
    board_probs = b.get_board_probability_distribution(1000)
    assert len(board_probs) == 1
    assert_prob_about(board_probs, u.squares_to_bitboard(["a2"]), 1.0)


def test_undo_entangled_measurement():
    b = simulator(u.squares_to_bitboard(["a2", "b1", "c2", "d1"]))
    assert b.perform_moves("b1^a3c3:SPLIT_JUMP:BASIC", "c2c4:PAWN_TWO_STEP:BASIC")
    probs = b.get_probability_distribution(10000)
    assert_prob_about(probs, qb.square_to_bit("a3"), 0.5)
    assert_prob_about(probs, qb.square_to_bit("c2"), 0.5)
    assert_prob_about(probs, qb.square_to_bit("c3"), 0.5)
    assert_prob_about(probs, qb.square_to_bit("c4"), 0.5)
    board_probs = b.get_board_probability_distribution(10000)
    assert len(board_probs) == 2
    assert_prob_about(board_probs, u.squares_to_bitboard(["a3", "c4", "a2", "d1"]), 0.5)
    assert_prob_about(board_probs, u.squares_to_bitboard(["c3", "c2", "a2", "d1"]), 0.5)
    b.perform_moves("d1c2:JUMP:EXCLUDED")
    assert b.undo_last_move()
    probs = b.get_probability_distribution(10000)
    assert_prob_about(probs, qb.square_to_bit("a3"), 0.5)
    assert_prob_about(probs, qb.square_to_bit("c2"), 0.5)
    assert_prob_about(probs, qb.square_to_bit("c3"), 0.5)
    assert_prob_about(probs, qb.square_to_bit("c4"), 0.5)
    board_probs = b.get_board_probability_distribution(10000)
    assert len(board_probs) == 2
    assert_prob_about(board_probs, u.squares_to_bitboard(["a3", "c4", "a2", "d1"]), 0.5)
    assert_prob_about(board_probs, u.squares_to_bitboard(["c3", "c2", "a2", "d1"]), 0.5)


def test_record_time():
    b = qb.CirqBoard(u.squares_to_bitboard(["b8", "b5", "c7"]))
    assert b.perform_moves(
        "b8^a6c6:SPLIT_JUMP:BASIC",
        "b5a6:PAWN_CAPTURE:BASIC",
        "c6b8:JUMP:BASIC",
        "c7c5:PAWN_TWO_STEP:BASIC",
        "b5c6:PAWN_EP:BASIC",
    )
    assert "sample_with_ancilla takes" in b.debug_log
    assert "seconds." in b.debug_log
    assert (
        float(b.debug_log.split("sample_with_ancilla takes ")[-1].split(" seconds.")[0])
        > 0
    )

    b.record_time("test_action", 0.12, 0.345)
    assert "test_action takes" in b.debug_log
    expected_time_1 = pytest.approx(0.345 - 0.12, 1e-7)
    assert (
        float(b.debug_log.split("test_action takes ")[-1].split(" seconds.")[0])
        == expected_time_1
    )
    assert len(b.timing_stats) == 2
    assert b.timing_stats["test_action"][-1] == expected_time_1

    b.record_time("test_action", 0.5, 0.987)
    expected_time_2 = pytest.approx(0.987 - 0.5, 1e-7)
    assert (
        float(b.debug_log.split("test_action takes ")[-1].split(" seconds.")[0])
        == expected_time_2
    )
    assert len(b.timing_stats) == 2
    assert len(b.timing_stats["test_action"]) == 2
    assert b.timing_stats["test_action"][-1] == expected_time_2


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_caching_accumulations_different_repetition_not_cached(board):
    b = board(u.squares_to_bitboard(["a1", "b1"]))
    b.do_move(
        move.Move(
            "b1",
            "c1",
            target2="d1",
            move_type=enums.MoveType.SPLIT_JUMP,
            move_variant=enums.MoveVariant.BASIC,
        )
    )
    probs1 = b.get_probability_distribution(1)
    probs2 = b.get_probability_distribution(CACHE_INVALIDATION_REPS)
    assert probs1 != probs2
    board_probs1 = b.get_board_probability_distribution(1)
    board_probs2 = b.get_board_probability_distribution(100)
    assert board_probs1 != board_probs2


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_caching_accumulations_same_repetition_cached(board):
    b = board(u.squares_to_bitboard(["a1", "b1"]))
    b.do_move(
        move.Move(
            "b1",
            "c1",
            target2="d1",
            move_type=enums.MoveType.SPLIT_JUMP,
            move_variant=enums.MoveVariant.BASIC,
        )
    )
    probs1 = b.get_probability_distribution(100)
    probs2 = b.get_probability_distribution(100)
    assert probs1 == probs2
    board_probs1 = b.get_board_probability_distribution(100)
    board_probs2 = b.get_board_probability_distribution(100)
    assert board_probs1 == board_probs2


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_get_probability_distribution_split_jump_pre_cached(board):
    b = board(u.squares_to_bitboard(["a1", "b1"]))
    # Cache a split jump in advance.
    cache_key = CacheKey(enums.MoveType.SPLIT_JUMP, CACHE_INVALIDATION_REPS)
    b.cache_results(cache_key)

    m1 = move.Move(
        "a1", "a2", move_type=enums.MoveType.JUMP, move_variant=enums.MoveVariant.BASIC
    )
    m2 = move.Move(
        "b1",
        "c1",
        target2="d1",
        move_type=enums.MoveType.SPLIT_JUMP,
        move_variant=enums.MoveVariant.BASIC,
    )
    b.do_move(m1)

    probs = list(b.get_probability_distribution(CACHE_INVALIDATION_REPS))
    b.do_move(m2)
    b.clear_debug_log()
    # Expected probability with the cache applied
    probs[square_to_bit("b1")] = 0
    probs[square_to_bit("c1")] = b.cache[cache_key]["target"]
    probs[square_to_bit("d1")] = b.cache[cache_key]["target2"]

    # Get probability distribution should apply the cache without rerunning _generate_accumulations.

    probs2 = b.get_probability_distribution(CACHE_INVALIDATION_REPS)
    full_squares = b.get_full_squares_bitboard(100)
    empty_squares = b.get_empty_squares_bitboard(100)

    assert tuple(probs) == probs2
    # Check that the second run and getting full and empty bitboards did not trigger any new logs.
    assert len(b.debug_log) == 0
    # Check bitboard updated correctly
    assert not nth_bit_of(square_to_bit("b1"), full_squares)
    assert not nth_bit_of(square_to_bit("c1"), full_squares)
    assert not nth_bit_of(square_to_bit("d1"), full_squares)
    assert nth_bit_of(square_to_bit("b1"), empty_squares)


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_get_probability_distribution_split_jump_first_move_pre_cached(board):
    b = board(u.squares_to_bitboard(["a1", "b1"]))
    # Cache a split jump in advance.
    cache_key = CacheKey(enums.MoveType.SPLIT_JUMP, CACHE_INVALIDATION_REPS)
    b.cache_results(cache_key)
    m1 = move.Move(
        "b1",
        "c1",
        target2="d1",
        move_type=enums.MoveType.SPLIT_JUMP,
        move_variant=enums.MoveVariant.BASIC,
    )
    b.do_move(m1)
    b.clear_debug_log()
    # Expected probability with the cache applied
    expected_probs = [0] * 64
    expected_probs[square_to_bit("a1")] = 1
    expected_probs[square_to_bit("b1")] = 0
    expected_probs[square_to_bit("c1")] = b.cache[cache_key]["target"]
    expected_probs[square_to_bit("d1")] = b.cache[cache_key]["target2"]

    # Get probability distribution should apply the cache without rerunning _generate_accumulations.
    probs = b.get_probability_distribution(CACHE_INVALIDATION_REPS)
    full_squares = b.get_full_squares_bitboard(100)
    empty_squares = b.get_empty_squares_bitboard(100)

    assert probs == tuple(expected_probs)
    # Check that the second run and getting full and empty bitboards did not trigger any new logs.
    assert len(b.debug_log) == 0
    # Check bitboard updated correctly
    assert not nth_bit_of(square_to_bit("b1"), full_squares)
    assert not nth_bit_of(square_to_bit("c1"), full_squares)
    assert not nth_bit_of(square_to_bit("d1"), full_squares)
    assert nth_bit_of(square_to_bit("b1"), empty_squares)


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_jump_with_successful_measurement_outcome(board):
    b = board(u.squares_to_bitboard(["b1", "c2"]))
    b.do_move(
        move.Move(
            "b1",
            "a3",
            target2="c3",
            move_type=enums.MoveType.SPLIT_JUMP,
            move_variant=enums.MoveVariant.BASIC,
        )
    )
    b.do_move(
        move.Move(
            "c2",
            "c3",
            move_type=enums.MoveType.JUMP,
            move_variant=enums.MoveVariant.EXCLUDED,
            measurement=1,
        )
    )
    assert_samples_in(b, [u.squares_to_bitboard(["c3", "a3"])])


def test_split_capture_with_successful_measurement_outcome():
    b = simulator(0)
    # Repeat the moves several times because the do_move calls will trigger a
    # measurement.
    for _ in range(10):
        b.with_state(u.squares_to_bitboard(["a1", "c3"]))
        # a1 splits into a2 + a3
        b.do_move(
            move.Move(
                "a1",
                "a2",
                target2="a3",
                move_type=enums.MoveType.SPLIT_JUMP,
                move_variant=enums.MoveVariant.BASIC,
            )
        )
        # Then a3 tries to capture c3, which requires measuring a3.
        # The provided measurement outcome says that there is a piece on a3
        # after all, so the capture is successful.
        b.do_move(
            move.Move(
                "a3",
                "c3",
                move_type=enums.MoveType.JUMP,
                move_variant=enums.MoveVariant.CAPTURE,
                measurement=1,
            )
        )
        samples = b.sample(1000)
        # The only possible outcome is successful capture.
        expected = {"c3"}
        for sample in samples:
            assert set(u.bitboard_to_squares(sample)) == expected


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_split_capture_with_failed_measurement_outcome(board):
    b = board(0)
    # Repeat the moves several times because the do_move calls will trigger a
    # measurement.
    for _ in range(20):
        b.with_state(u.squares_to_bitboard(["a1", "c3"]))
        # a1 splits into a2 + a3
        b.do_move(
            move.Move(
                "a1",
                "a2",
                target2="a3",
                move_type=enums.MoveType.SPLIT_JUMP,
                move_variant=enums.MoveVariant.BASIC,
            )
        )
        # Then a3 tries to capture c3, which requires measuring a3.
        # The provided measurement outcome says that there is not a piece on a3
        # after all, so the capture is unsuccessful.
        b.do_move(
            move.Move(
                "a3",
                "c3",
                move_type=enums.MoveType.JUMP,
                move_variant=enums.MoveVariant.CAPTURE,
                measurement=0,
            )
        )
        # The only possible outcome is unsuccessful capture -- a1 jumped to a2,
        # not a3, and the to-be-captured piece on c3 still remains.
        # However, in the presence of readout error, we may not measure the
        # piece on either a2 or a3 so post-filtered samples may not contain a2
        # in rare cases.
        probs = b.get_probability_distribution(100)
        assert_prob_about(probs, u.square_to_bit("a2"), 1, atol=0.1)
        assert_prob_about(probs, u.square_to_bit("a3"), 0, atol=0.1)


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_merge_to_fully_classical_position(board):
    """Splits piece on d4 to b3 and c2, then merge back to fully classical position on a1."""
    b = simulator(u.squares_to_bitboard(["d4"]))
    b.reset_starting_states = True
    assert b.perform_moves(
        "d4^b3c2:SPLIT_JUMP:BASIC",
        "b3c2^a1:MERGE_JUMP:BASIC",
    )

    expected_probs = [0] * 64
    expected_probs[square_to_bit("a1")] = 1
    probs = list(b.get_probability_distribution(100))

    assert b.is_classical()
    assert probs == expected_probs


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_undo_to_start_after_classical_reset(board):
    """Splits piece on f8 to d6 and h6. Piece on f4 then captures piece on d6. Then piece on h6 captures d6
    Then does three undo moves to return to initial position."""
    b = simulator(u.squares_to_bitboard(["f4", "f8", "f2"]))
    b.reset_starting_states = True
    initial_board = b.get_full_squares_bitboard()
    assert b.perform_moves(
        "f8^d6h6:SPLIT_JUMP:BASIC",
        "f4^d6:JUMP:CAPTURE",
        "f2^h6:JUMP:CAPTURE",
    )

    assert b.is_classical()
    assert b.circuit == cirq.Circuit()

    assert b.undo_last_move()
    assert b.undo_last_move()
    assert b.undo_last_move()

    assert b.get_full_squares_bitboard() == initial_board


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_quantum_capture_with_forced_measurement(board):
    """Splits piece on b1 to a1 and c1. Piece on a1 then captures piece on a5."""
    b = simulator(u.squares_to_bitboard(["b1", "a5"]))
    b.reset_starting_states = True

    b.perform_moves(
        "b1^a1c1:SPLIT_JUMP:BASIC",
        "a1^a5:JUMP:CAPTURE",
    )

    assert b.is_classical()
    assert b.move_history != []
    assert b.circuit == cirq.Circuit()
    assert b.ancilla_count == 0


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_measurement_without_fully_classical_position(board):
    """Splits piece on d3 to d7 and h3, then splits piece on b5 to b7 and b3.
    Piece on f5 then tries to capture b3, which will force a measurement but will
    not return the entire board to a fully classical position."""
    b = simulator(u.squares_to_bitboard(["b5", "f5", "d3"]))
    b.reset_starting_states = True
    b.perform_moves(
        "d3^d7h3:SPLIT_JUMP:BASIC",
        "b5^b7b3:SPLIT_JUMP:BASIC",
        "f5^b3:JUMP:CAPTURE",
    )
    assert not b.is_classical()
