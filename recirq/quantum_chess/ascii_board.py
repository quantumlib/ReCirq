import math
from typing import Optional, Sequence

import recirq.quantum_chess.bit_utils as bu
import recirq.quantum_chess.constants as c
import recirq.quantum_chess.enums as enums
import recirq.quantum_chess.quantum_board as qb
from recirq.quantum_chess.move import to_rank, to_square, x_of, y_of, Move


class AsciiBoard:
    """A ASCII front-end for the Quantum Board API.

    This board can be used to test and demo sequences of
    moves.  However, this board does not enforce legality,
    so use with caution.
    """

    def __init__(self, size: Optional[int] = 8, reps: Optional[int] = 1000, board=None):

        self.size = size
        self.reps = reps
        self.board = board or qb.CirqBoard(0)

        self._reset_state()

    _probs: Sequence[float]

    def _reset_state(self):
        """Resets all the values that represent the state to default values."""
        self._pieces = {}
        self._sampled = {}

        for x in range(self.size):
            for y in range(self.size):
                s = to_square(x, y)
                self._pieces[s] = 0

        self.ep_flag = None
        self.white_moves = True
        self.castling_flags = {
            "o-o": True,
            "o-o-o": True,
            "O-O": True,
            "O-O-O": True,
        }

    def _bit_board(self) -> int:
        rtn = 0
        for x in range(self.size):
            for y in range(self.size):
                s = to_square(x, y)
                if self._pieces[s] != c.EMPTY:
                    rtn = bu.set_nth_bit(bu.xy_to_bit(x, y), rtn, True)
        return rtn

    def load_fen(self, fen):
        """Sets up a position from the first component of a FEN string."""
        y = self.size
        for row in fen.split("/"):
            y -= 1
            x = 0
            for char in row:
                if "1" <= char <= "9":
                    x += int(char)
                else:
                    piece = c.REV_PIECES[char]
                    square = to_square(x, y)
                    self._pieces[square] = piece
                    x += 1
        self.board.with_state(self._bit_board())
        self._probs = self.board.get_probability_distribution(self.reps)

    def reset(self):
        """Resets to initial classical chess starting position."""
        self._reset_state()
        if self.size != 8:
            raise ValueError("board size must be 8 for standard chess position")
        self.load_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")

    def __str__(self):
        """Renders a ASCII diagram showing the board position."""
        s = ""
        s += " +-------------------------+\n"
        for y in reversed(range(self.size)):
            s += str(y + 1) + "| "
            for x in range(self.size):
                piece = c.PIECES[self._pieces[to_square(x, y)]]
                bit = bu.xy_to_bit(x, y)
                p = self._probs[bit]
                if p < 0.01:
                    piece = "."
                s += piece + "  "
            s += "|\n |"
            for x in range(self.size):
                bit = bu.xy_to_bit(x, y)
                p = self._probs[bit]
                if 0.01 < p < 0.99:
                    prob = str(int(100 * p))
                    if len(prob) <= 2:
                        s += " "
                    s += prob
                    if len(prob) < 2:
                        s += " "
                else:
                    s += "   "
            s += " |\n"
        s += " +-------------------------+\n   "
        for x in range(self.size):
            s += to_rank(x) + "  "
        return s

    def piece(self, square: str):
        """Returns the type of the piece on the given square."""
        return self._pieces[square]

    @staticmethod
    def _ep_flag(move: Move, piece_type: int):
        if abs(piece_type) != c.PAWN:
            return None
        if abs(y_of(move.target) - y_of(move.source)) == 2:
            return x_of(move.target)
        return None

    def blocked_squares(self, source, target):
        """Returns occupied squares between source and target."""
        rtn = []
        xs = x_of(source)
        ys = y_of(source)
        xt = x_of(target)
        yt = y_of(target)
        if xt > xs:
            dx = 1
        elif xt < xs:
            dx = -1
        else:
            dx = 0
        if yt > ys:
            dy = 1
        elif yt < ys:
            dy = -1
        else:
            dy = 0
        max_slide = max(abs(xs - xt), abs(ys - yt))
        if max_slide > 1:
            for t in range(1, max_slide):
                path = to_square(xs + dx * t, ys + dy * t)
                if self._pieces[path] != c.EMPTY:
                    rtn.append(path)
        return rtn

    def apply(self, move: Move):
        """Applies a move to the board."""

        s = self._pieces[move.source]

        # Defaults to a BASIC JUMP move if not specified
        if not move.move_type:
            move.move_type = enums.MoveType.JUMP
        if not move.move_variant:
            move.move_variant = enums.MoveVariant.BASIC

        # Call the quantum board to apply the move
        meas = self.board.do_move(move)

        # Cache the probability distribution
        self._probs = self.board.get_probability_distribution(self.reps)

        # Set the turn to be the next player
        self.white_moves = not self.white_moves

        # If the move was not successful, return
        # Otherwise, update classical bits
        if not meas:
            return meas

        # Assume that squares with low probability are empty
        if self._probs[bu.square_to_bit(move.source)] < 0.01:
            self._pieces[move.source] = c.EMPTY

        # Update classical bits
        self._pieces[move.target] = move.promotion_piece or s
        if move.target2:
            self._pieces[move.target2] = s
        if move.source2 and self._probs[bu.square_to_bit(move.source2)] < 0.01:
            self._pieces[move.source2] = c.EMPTY
        if move.move_type == enums.MoveType.KS_CASTLE:
            self._pieces[move.source.replace("e", "f")] = int(math.copysign(c.ROOK, s))
        if move.move_type == enums.MoveType.QS_CASTLE:
            self._pieces[move.source.replace("e", "d")] = int(math.copysign(c.ROOK, s))

        # Update en passant and castling flags
        self.ep_flag = self._ep_flag(move, s)
        if s == c.KING:
            self.castling_flags["O-O"] = False
            self.castling_flags["O-O-O"] = False
        if s == -c.KING:
            self.castling_flags["o-o"] = False
            self.castling_flags["o-o-o"] = False
        if move.source == "a1" or move.target == "a1":
            self.castling_flags["O-O-O"] = False
        if move.source == "a8" or move.target == "a8":
            self.castling_flags["o-o-o"] = False
        if move.source == "h1" or move.target == "h1":
            self.castling_flags["O-O"] = False
        if move.source == "h8" or move.target == "h8":
            self.castling_flags["o-o"] = False

        return meas
