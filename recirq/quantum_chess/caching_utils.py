from dataclasses import dataclass
import recirq.quantum_chess.enums as enums
import recirq.quantum_chess.move as move


@dataclass(frozen=True, eq=True)
class CacheKey:
    move_type: enums.MoveType
    repetitions: int


def cache_key_from_move(m: move.Move, repetitions: int) -> CacheKey:
    return CacheKey(m.move_type, repetitions)
