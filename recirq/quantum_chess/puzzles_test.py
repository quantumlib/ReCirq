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

import recirq.quantum_chess.bit_utils as u
import recirq.quantum_chess.quantum_board_test as quantum_board_test
import recirq.quantum_chess.test_utils as test_utils

ALL_CIRQ_BOARDS = quantum_board_test.ALL_CIRQ_BOARDS
BIG_CIRQ_BOARDS = quantum_board_test.BIG_CIRQ_BOARDS


def assert_samples_in(b, possibilities):
    return quantum_board_test.assert_samples_in(b, possibilities)


@pytest.mark.parametrize(
    "solution,sq",
    (
        ("b3b7^d5", "d5"),
        ("b7b3^d5", "d5"),
        ("b3b7^f3", "f3"),
        ("b7b3^f3", "f3"),
        ("b3b7^f7", "f7"),
        ("b7b3^f7", "f7"),
    ),
)
@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_t1(solution, sq, board):
    b = board(u.squares_to_bitboard(["b5", "c5", "d5"]))
    assert b.perform_moves("d5^b3b7:SPLIT_SLIDE:BASIC", f"{solution}:MERGE_SLIDE:BASIC")
    possibilities = [u.squares_to_bitboard(["b5", "c5", sq])]
    assert_samples_in(b, possibilities)


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_t2(board):
    b = board(u.squares_to_bitboard(["a7", "c3", "g1"]))
    did_it_move = b.perform_moves(
        "c3^b5e2:SPLIT_JUMP:BASIC",
        "a7a8:SLIDE:BASIC",
        "e2g1:JUMP:CAPTURE",
    )
    if did_it_move:
        possibilities = [u.squares_to_bitboard(["a8", "g1"])]
    else:
        possibilities = [u.squares_to_bitboard(["a8", "b5", "g1"])]
    assert_samples_in(b, possibilities)


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_t3(board):
    b = board(u.squares_to_bitboard(["f1", "d7", "e8"]))
    b.perform_moves(
        "f1^b5c4:SPLIT_SLIDE:BASIC",
        "d7b5:SLIDE:CAPTURE",
        "c4b5:SLIDE:CAPTURE",
    )
    possibilities = [u.squares_to_bitboard(["b5", "e8"])]
    assert_samples_in(b, possibilities)


# Because of the long decomposition, is flaky with noisy simulator
@pytest.mark.parametrize("board", quantum_board_test.BIG_CIRQ_BOARDS)
def test_t4(board):
    b = board(u.squares_to_bitboard(["e3", "e6", "b6", "h6"]))
    did_it_move = b.perform_moves(
        "e3^d5f5:SPLIT_JUMP:BASIC",
        "e6d5:PAWN_CAPTURE:CAPTURE",
        "f5h6:JUMP:CAPTURE",
    )
    if did_it_move:
        possibilities = [u.squares_to_bitboard(["b6", "e6", "h6"])]
    else:
        possibilities = [u.squares_to_bitboard(["d5", "b6", "h6"])]
    assert_samples_in(b, possibilities)


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_t5(board):
    b = board(u.squares_to_bitboard(["d4", "a6", "g6"]))
    did_it_move = b.perform_moves(
        "d4^a7g7:SPLIT_SLIDE:BASIC",
        "a6a7:PAWN_STEP:EXCLUDED",
    )
    if did_it_move:
        expected = u.squares_to_bitboard(["a7", "g6", "g7"])
    else:
        expected = u.squares_to_bitboard(["a7", "g6", "a6"])
    assert_samples_in(b, [expected])


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_t6(board):
    b = board(u.squares_to_bitboard(["h4", "d5", "g5"]))
    did_it_move = b.perform_moves(
        "h4^f5g6:SPLIT_JUMP:BASIC",
        "g5g6:PAWN_STEP:EXCLUDED",
    )
    if did_it_move:
        did_it_move = b.perform_moves(
            "f5g7:JUMP:BASIC",
            "d5d6:PAWN_STEP:BASIC",
            "g7e6:JUMP:BASIC",
            "d6d7:PAWN_STEP:BASIC",
            "e6^d8g7:SPLIT_JUMP:BASIC",
            "d7d8:PAWN_STEP:EXCLUDED",
        )
        if did_it_move:
            expected = u.squares_to_bitboard(["d8", "g7", "g6"])
        else:
            expected = u.squares_to_bitboard(["d8", "d7", "g6"])
        assert_samples_in(b, [expected])
    else:
        did_it_move = b.perform_moves(
            "g6e5:JUMP:BASIC",
            "d5d6:PAWN_STEP:BASIC",
            "e5^d7g6:SPLIT_JUMP:BASIC",
            "d6d7:PAWN_STEP:EXCLUDED",
        )
        if did_it_move:
            expected = u.squares_to_bitboard(["d7", "g6", "g5"])
        else:
            expected = u.squares_to_bitboard(["d6", "d7", "g5"])
        assert_samples_in(b, [expected])


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_t6(board):
    b = board(u.squares_to_bitboard(["d4", "a6", "g6"]))
    did_it_move = b.perform_moves(
        "d4^a7g7:SPLIT_SLIDE:BASIC",
        "a6a7:PAWN_STEP:EXCLUDED",
    )
    if did_it_move:
        expected = u.squares_to_bitboard(["a7", "g6", "g7"])
    else:
        expected = u.squares_to_bitboard(["a7", "g6", "a6"])
    assert_samples_in(b, [expected])


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_a1(board):
    b = board(
        u.squares_to_bitboard(["a8", "b8", "g8", "f7", "g7", "g3", "f2", "f3", "h1"])
    )
    did_it_move = b.perform_moves(
        "h1^h4h8:SPLIT_SLIDE:BASIC",
        "g8^f8h7:SPLIT_JUMP:BASIC",
        "h8f8:SLIDE:CAPTURE",
    )
    if did_it_move:
        possibilities = [
            u.squares_to_bitboard(["a8", "b8", "f8", "f7", "g7", "g3", "f2", "f3"]),
            u.squares_to_bitboard(
                ["a8", "b8", "f8", "f7", "g7", "g3", "f2", "f3", "h7"]
            ),
        ]
    else:
        possibilities = [
            u.squares_to_bitboard(
                ["a8", "b8", "f8", "f7", "g7", "g3", "f2", "f3", "h4"]
            ),
            u.squares_to_bitboard(
                ["a8", "b8", "h7", "f7", "g7", "g3", "f2", "f3", "h4"]
            ),
        ]
    assert_samples_in(b, possibilities)


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_a2(board):
    b = board(
        u.squares_to_bitboard(["a8", "b8", "g8", "f7", "g7", "g3", "f2", "f3", "h1"])
    )
    did_it_move = b.perform_moves(
        "h1^h7h8:SPLIT_SLIDE:BASIC",
        "g8f8:JUMP:BASIC",
        "h8f8:SLIDE:CAPTURE",
    )
    if did_it_move:
        possibilities = [
            u.squares_to_bitboard(["a8", "b8", "f8", "f7", "g7", "g3", "f2", "f3"])
        ]
    else:
        possibilities = [
            u.squares_to_bitboard(
                ["a8", "b8", "f8", "f7", "g7", "g3", "f2", "f3", "h7"]
            )
        ]
    assert_samples_in(b, possibilities)


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_a3(board):
    b = board(u.squares_to_bitboard(["f1", "f2", "f3"]))
    assert b.perform_moves(
        "f1^e1e2:SPLIT_JUMP:BASIC",
        "f3e2:JUMP:CAPTURE",
    )
    possibilities = [
        u.squares_to_bitboard(["e2", "f2"]),
        u.squares_to_bitboard(["e1", "e2", "f2"]),
    ]
    assert_samples_in(b, possibilities)


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_a4(board):
    b = board(u.squares_to_bitboard(["g8", "f7", "g7", "h7", "g6", "g3", "g2", "e1"]))
    assert b.perform_moves(
        "e1e8:SLIDE:BASIC",
        "g8^f8h8:SPLIT_JUMP:BASIC",
        "e8f8:SLIDE:CAPTURE",
        "h8g8:JUMP:BASIC",
        "f8g8:SLIDE:CAPTURE",
    )
    possibilities = [
        u.squares_to_bitboard(["g8", "f7", "g7", "h7", "g6", "g3", "g2"]),
    ]
    assert_samples_in(b, possibilities)


@pytest.mark.parametrize("board", BIG_CIRQ_BOARDS)
def test_a5(board):
    b = board(
        u.squares_to_bitboard(
            ["f2", "g2", "h2", "g3", "c4", "c5", "c6", "f7", "g7", "h7", "a8", "h8"]
        )
    )
    did_it_move = b.perform_moves("c6^b8d8:SPLIT_JUMP:BASIC", "a8d8:SLIDE:CAPTURE")
    if did_it_move:
        expected = u.squares_to_bitboard(
            ["f2", "g2", "h2", "g3", "c4", "c5", "f7", "g7", "h7", "d8", "h8"]
        )
    else:
        expected = u.squares_to_bitboard(
            ["f2", "g2", "h2", "g3", "c4", "c5", "f7", "g7", "h7", "a8", "b8", "h8"]
        )
    assert_samples_in(b, [expected])


@pytest.mark.parametrize("board", BIG_CIRQ_BOARDS)
def test_a6(board):
    b = board(
        u.squares_to_bitboard(
            ["a2", "f2", "g2", "h2", "g3", "c6", "h6", "f7", "g7", "h7", "h8"]
        )
    )
    did_it_move = b.perform_moves(
        "a2a8:SLIDE:BASIC",
        "c6^b8d8:SPLIT_JUMP:BASIC",
        "a8d8:SLIDE:CAPTURE",
    )
    if did_it_move:
        expected = u.squares_to_bitboard(
            ["d8", "f2", "g2", "h2", "g3", "h6", "f7", "g7", "h7", "h8"]
        )
    else:
        expected = u.squares_to_bitboard(
            ["a8", "f2", "g2", "h2", "g3", "b8", "h6", "f7", "g7", "h7", "h8"]
        )
    assert_samples_in(b, [expected])


# Works on all cirq boards but is pretty slow.
@pytest.mark.parametrize("board", quantum_board_test.BIG_CIRQ_BOARDS)
def test_a6_alternate(board):
    b = board(
        u.squares_to_bitboard(
            ["a2", "f2", "g2", "h2", "g3", "c6", "h6", "f7", "g7", "h7", "h8"]
        )
    )
    did_it_move = b.perform_moves(
        "a2a8:SLIDE:BASIC",
        "c6^b8d8:SPLIT_JUMP:BASIC",
        "a8b8:SLIDE:CAPTURE",
        "h6d6:SLIDE:BASIC",
        "b8h8:SLIDE:CAPTURE",
    )
    if did_it_move:
        expected = u.squares_to_bitboard(
            ["f2", "g2", "h2", "g3", "d6", "f7", "g7", "h7", "h8"]
        )
    else:
        expected = u.squares_to_bitboard(
            ["b8", "f2", "g2", "h2", "g3", "d8", "d6", "f7", "g7", "h7", "h8"]
        )
    assert_samples_in(b, [expected])


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_a7(board):
    b = board(
        u.squares_to_bitboard(
            ["a2", "c2", "e2", "g2", "h2", "a3", "h3", "f6", "g7", "h7", "h8"]
        )
    )
    did_it_move = b.perform_moves("c2^a1e1:SPLIT_JUMP:BASIC", "e2e1:PAWN_STEP:EXCLUDED")
    if did_it_move:
        possibilities = [
            u.squares_to_bitboard(
                ["a1", "a2", "e1", "g2", "h2", "a3", "h3", "f6", "g7", "h7", "h8"]
            )
        ]
        assert_samples_in(b, possibilities)
        return
    b.perform_moves(
        "a3c2:JUMP:BASIC",
        "a2a1:PAWN_STEP:BASIC",
        "c2a1:JUMP:CAPTURE",
    )
    possibilities = [
        u.squares_to_bitboard(
            ["a1", "e1", "e2", "g2", "h2", "h3", "f6", "g7", "h7", "h8"]
        )
    ]
    assert_samples_in(b, possibilities)


@pytest.mark.parametrize("board", BIG_CIRQ_BOARDS)
def test_a8(board):
    b = board(u.squares_to_bitboard(["a1", "d1", "c5", "a7", "h8"]))
    did_it_move = b.perform_moves("c5^a4a6:SPLIT_JUMP:BASIC", "a1a6:SLIDE:CAPTURE")
    if did_it_move:
        possibilities = [u.squares_to_bitboard(["a6", "d1", "a7", "h8"])]
    else:
        possibilities = [u.squares_to_bitboard(["a1", "d1", "a4", "a7", "h8"])]
    assert_samples_in(b, possibilities)


@pytest.mark.parametrize("board", BIG_CIRQ_BOARDS)
def test_a9(board):
    b = board(
        u.squares_to_bitboard(["h1", "e2", "f2", "g2", "e3", "h3", "g6", "h6", "c7"])
    )
    did_it_move = b.perform_moves(
        "e2e3:SLIDE:CAPTURE",
        "c7^f4h2:SPLIT_SLIDE:BASIC",
        "e3h6:SLIDE:CAPTURE",
    )
    if did_it_move:
        possibilities = [
            u.squares_to_bitboard(["h1", "f2", "g2", "h3", "g6", "h6", "h2"])
        ]
    else:
        possibilities = [
            u.squares_to_bitboard(["h1", "f2", "g2", "e3", "h3", "g6", "h6", "f4"])
        ]
    assert_samples_in(b, possibilities)


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_a10(board):
    b = board(
        u.squares_to_bitboard(
            [
                "d1",
                "f1",
                "g1",
                "f2",
                "g2",
                "h2",
                "d3",
                "d4",
                "d5",
                "d6",
                "e6",
                "g6",
                "h6",
                "g7",
                "b8",
                "f8",
            ]
        )
    )
    did_it_move = b.perform_moves(
        "d3^c5e5:SPLIT_JUMP:BASIC",
        "g6h7:JUMP:BASIC",
        "c5e5^d7:MERGE_JUMP:BASIC",
        "b8d8:SLIDE:BASIC",
        "d7f8:JUMP:CAPTURE",
    )
    assert did_it_move
    possibilities = [
        u.squares_to_bitboard(
            [
                "d1",
                "f1",
                "g1",
                "f2",
                "g2",
                "h2",
                "d4",
                "d5",
                "d6",
                "e6",
                "h7",
                "h6",
                "g7",
                "d8",
                "f8",
            ]
        )
    ]
    assert_samples_in(b, possibilities)


@pytest.mark.parametrize("board", ALL_CIRQ_BOARDS)
def test_a11(board):
    b = board(
        u.squares_to_bitboard(
            ["d1", "e1", "h1", "d2", "e2", "f2", "g2", "e3", "e6", "d7", "e7", "f7"]
        )
    )
    did_it_move = b.perform_moves(
        "e3^c2f1:SPLIT_JUMP:BASIC",
        "e1g1:KS_CASTLE:BASIC",
    )
    if did_it_move:
        possibilities = [
            u.squares_to_bitboard(
                ["d1", "f1", "g1", "d2", "e2", "f2", "g2", "c2", "e6", "d7", "e7", "f7"]
            ),
        ]
    else:
        possibilities = [
            u.squares_to_bitboard(
                ["d1", "e1", "h1", "d2", "e2", "f2", "g2", "f1", "e6", "d7", "e7", "f7"]
            ),
        ]

    assert_samples_in(b, possibilities)


@pytest.mark.parametrize("board", BIG_CIRQ_BOARDS)
def test_blocked_split_slide(board):
    b = board(u.squares_to_bitboard(["d1", "g1"]))
    assert b.perform_moves(
        "g1^e2h3:SPLIT_JUMP:BASIC",
        "d1^b3g4:SPLIT_SLIDE:BASIC",
    )
    test_utils.assert_sample_distribution(
        b,
        {
            u.squares_to_bitboard(["e2", "b3"]): 1 / 2,
            u.squares_to_bitboard(["h3", "b3"]): 1 / 4,
            u.squares_to_bitboard(["h3", "g4"]): 1 / 4,
        },
    )


def test_blocked_split_slide_merge1():
    b = quantum_board_test.simulator(u.squares_to_bitboard(["d1", "g1"]))
    assert b.perform_moves(
        "g1^e2h3:SPLIT_JUMP:BASIC",
        "d1^b3g4:SPLIT_SLIDE:BASIC",
        "b3g4^e6:MERGE_SLIDE:BASIC",
    )
    test_utils.assert_sample_distribution(
        b,
        {
            u.squares_to_bitboard(["e2", "e6"]): 1 / 4,
            u.squares_to_bitboard(["e2", "g4"]): 1 / 4,
            u.squares_to_bitboard(["h3", "e6"]): 1 / 2,
        },
    )


def test_blocked_split_slide_merge2():
    b = quantum_board_test.simulator(u.squares_to_bitboard(["d1", "g1"]))
    assert b.perform_moves(
        "g1^e2h3:SPLIT_JUMP:BASIC",
        "d1^b3g4:SPLIT_SLIDE:BASIC",
        "g4b3^e6:MERGE_SLIDE:BASIC",
    )
    test_utils.assert_sample_distribution(
        b,
        {
            u.squares_to_bitboard(["e2", "e6"]): 1 / 4,
            u.squares_to_bitboard(["e2", "b3"]): 1 / 4,
            u.squares_to_bitboard(["h3", "e6"]): 1 / 2,
        },
    )
