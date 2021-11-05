from dataclasses import dataclass
from typing import Tuple
import recirq.quantum_chess.enums as enums
import recirq.quantum_chess.move as move


@dataclass(frozen=True, eq=True)
class CacheKey:
    move_type: enums.MoveType
    repetitions: int


def cache_key_from_move(m: move.Move, repetitions: int) -> CacheKey:
    return CacheKey(m.move_type, repetitions)


@dataclass(frozen=True)
class ProbabilityHistory:
    """Stores square occupancy histogram for a point in the move history.

    Args:
        repetitions: number of samples used to generate probabilities
        probabilities: maps square -> probability of being occupied
        full_squares: (derived from self.probabilities) full squares bitboard
        empty_squares: (derived from self.probabilities) empty squares bitboard
    """

    repetitions: int
    probabilities: Tuple[float, ...]  # for each square
    full_squares: int  # bitboard derived from square_probabilities
    empty_squares: int  # bitboard derived from square_probabilities
