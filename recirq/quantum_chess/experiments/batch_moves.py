"""
Runs a series of moves on a quantum chess board then displays the
result.

Run with:
  python -m recirq.quantum_chess.experiments.batch_moves \
      FILENAME PROCESSOR_NAME FEN

FILENAME is a filename with moves in the move format described
by recirq.quantum_chess.move (see also interactive_board for
examples).

PROCESSOR_NAME is a processor name from engine_utils.py.
Defaults to a 54 qubit sycamore noiseless simulator.

FEN is a initial position in chess FEN notation. Optional.
Default is the normal classical chess starting position.
"""
from typing import List
import sys

import cirq.google as cg
import recirq.engine_utils as utils
import recirq.quantum_chess.ascii_board as ab
import recirq.quantum_chess.enums as enums
import recirq.quantum_chess.move as m
import recirq.quantum_chess.quantum_board as qb


def create_board(processor_name: str, *, noise_mitigation: float):
    return qb.CirqBoard(init_basis_state=0,
                        sampler=utils.get_sampler_by_name(
                            processor_name, gateset=cg.SQRT_ISWAP_GATESET),
                        device=utils.get_device_obj_by_name(processor_name),
                        error_mitigation=enums.ErrorMitigation.Correct,
                        noise_mitigation=noise_mitigation)


def apply_moves(b: ab.AsciiBoard, moves: List[str]) -> bool:
    meas = False
    for move_str in moves:
        if ':' not in move_str:
            next_move = m.Move.from_string(move_str + ':JUMP:BASIC')
        else:
            next_move = m.Move.from_string(move_str)
        meas = b.apply(next_move)
    return meas


def main_loop(filename: str,
              processor_name: str,
              fen_position=None,
              noise_mitigation=0.1):
    f = open(filename, 'r')
    moves = [line.strip() for line in f]
    board = create_board(processor_name=processor_name,
                         noise_mitigation=noise_mitigation)
    b = ab.AsciiBoard(board=board)
    if fen_position:
        b.load_fen(fen_position)
    else:
        b.reset()
    print(f'Applying moves to board...')
    apply_moves(b, moves)
    print(b)


if __name__ == '__main__':
    filename = sys.argv[1] if len(sys.argv) > 1 else None
    processor_name = sys.argv[2] if len(sys.argv) > 2 else 'Syc54-noiseless'
    position = sys.argv[3] if len(sys.argv) > 3 else None
    main_loop(filename=filename,
              processor_name=processor_name,
              fen_position=position)
