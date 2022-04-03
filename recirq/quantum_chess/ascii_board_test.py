import recirq.quantum_chess.ascii_board as ab
import recirq.quantum_chess.constants as c
import recirq.quantum_chess.move as m
from recirq.quantum_chess.enums import MoveType, MoveVariant


def test_squares():
    assert ab.to_square(0, 0) == "a1"
    assert ab.to_square(1, 0) == "b1"
    assert ab.to_square(1, 1) == "b2"
    assert ab.to_square(7, 7) == "h8"
    assert ab.x_of("a1") == 0
    assert ab.y_of("a1") == 0
    assert ab.x_of("h7") == 7
    assert ab.y_of("h7") == 6


def test_fen():
    b = ab.AsciiBoard()
    b.load_fen("8/8/8/8/2pB2p1/1kP3P1/P3P3/2bb4")
    expected_pieces = {
        "c4": -c.PAWN,
        "d4": c.BISHOP,
        "g4": -c.PAWN,
        "b3": -c.KING,
        "c3": c.PAWN,
        "g3": c.PAWN,
        "a2": c.PAWN,
        "e2": c.PAWN,
        "c1": -c.BISHOP,
        "d1": -c.BISHOP,
    }
    for x in range(8):
        for y in range(8):
            square = m.to_square(x, y)
            assert b._pieces[square] == expected_pieces.get(square, c.EMPTY)


def test_fen_existing_state():
    b = ab.AsciiBoard()
    b.reset()
    b.apply(
        m.Move(
            source="b1",
            target="a3",
            target2="c3",
            move_type=MoveType.SPLIT_JUMP,
            move_variant=MoveVariant.BASIC,
        )
    )
    b.reset()

    expected_board = ab.AsciiBoard()
    expected_board.reset()
    assert str(b) == str(expected_board)


def test_ep_flag():
    b = ab.AsciiBoard()
    b.reset()
    assert b._ep_flag(m.Move("b2", "b3"), c.PAWN) is None
    assert b._ep_flag(m.Move("b2", "b4"), c.KNIGHT) is None
    assert b._ep_flag(m.Move("b2", "b4"), c.PAWN) == 1


def test_kingside_castle():
    b = ab.AsciiBoard()
    b.load_fen("4k2r/8/8/8/8/8/8/8")
    b.apply(
        m.Move(
            source="e8",
            target="g8",
            move_type=MoveType.KS_CASTLE,
            move_variant=MoveVariant.BASIC,
        )
    )
    assert b.piece("e8") == c.EMPTY
    assert b.piece("f8") == -c.ROOK
    assert b.piece("g8") == -c.KING


def test_queenside_castle():
    b = ab.AsciiBoard()
    b.load_fen("8/8/8/8/8/8/8/R3K3")
    b.apply(
        m.Move(
            source="e1",
            target="c1",
            move_type=MoveType.QS_CASTLE,
            move_variant=MoveVariant.BASIC,
        )
    )
    assert b.piece("e1") == c.EMPTY
    assert b.piece("d1") == c.ROOK
    assert b.piece("c1") == c.KING


def test_pawn_promotion():
    b = ab.AsciiBoard()
    b.load_fen("7P/8/8/8/8/8/8/8")
    b.apply(
        m.Move(
            source="h7",
            target="h8",
            move_type=MoveType.JUMP,
            move_variant=MoveVariant.BASIC,
            promotion_piece=c.ROOK,
        )
    )
    assert b.piece("h7") == c.EMPTY
    assert b.piece("h8") == c.ROOK
