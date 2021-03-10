import recirq.quantum_chess.ascii_board as ab
import recirq.quantum_chess.constants as c
import recirq.quantum_chess.move as m


def test_squares():
    assert ab.to_square(0, 0) == 'a1'
    assert ab.to_square(1, 0) == 'b1'
    assert ab.to_square(1, 1) == 'b2'
    assert ab.to_square(7, 7) == 'h8'
    assert ab.x_of('a1') == 0
    assert ab.y_of('a1') == 0
    assert ab.x_of('h7') == 7
    assert ab.y_of('h7') == 6


def test_fen():
    b = ab.AsciiBoard()
    b.load_fen('8/8/8/8/2pB2p1/1kP3P1/P3P3/2bb4')
    expected_pieces = {
        'c4': -c.PAWN,
        'd4': c.BISHOP,
        'g4': -c.PAWN,
        'b3': -c.KING,
        'c3': c.PAWN,
        'g3': c.PAWN,
        'a2': c.PAWN,
        'e2': c.PAWN,
        'c1': -c.BISHOP,
        'd1': -c.BISHOP,
    }
    for x in range(8):
        for y in range(8):
            square = m.to_square(x, y)
            assert b._pieces[square] == expected_pieces.get(square, c.EMPTY)


def test_ep_flag():
    b = ab.AsciiBoard()
    b.reset()
    assert b._ep_flag(m.Move('b2', 'b3'), c.PAWN) is None
    assert b._ep_flag(m.Move('b2', 'b4'), c.KNIGHT) is None
    assert b._ep_flag(m.Move('b2', 'b4'), c.PAWN) == 1
