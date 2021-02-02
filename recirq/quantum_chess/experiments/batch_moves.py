"""
Runs a series of moves on a quantum chess board then displays the
result.

Run with:
  python -m recirq.quantum_chess.experiments.batch_moves \
      FILENAME --processor_name=PROCESSOR_NAME \
      --position=FEN

FILENAME is a filename with moves in the move format described
by recirq.quantum_chess.move (see also interactive_board for
examples).

PROCESSOR_NAME is a processor name from engine_utils.py.
Defaults to a 54 qubit sycamore noiseless simulator.

FEN is a initial position in chess FEN notation. Optional.
Default is the normal classical chess starting position.
"""
import argparse
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
                            processor_name, gateset='sqrt-iswap'),
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


def main_loop(args):
    f = open(args.filename, 'r')
    moves = [line.strip() for line in f]
    board = create_board(processor_name=args.processor_name,
                         noise_mitigation=0.1)
    b = ab.AsciiBoard(board=board)
    if args.position:
        b.load_fen(args.position)
    else:
        b.reset()
    print(f'Applying moves to board...')
    apply_moves(b, moves)
    print(b)


def parse():
    parser = argparse.ArgumentParser(
        description='Interactive quantum chess board.')
    parser.add_argument('filename',
                        type=str,
                        help='path to file that contains one move per line')
    parser.add_argument('--processor_name',
                        type=str,
                        default='Syc54-noiseless',
                        help='name of the QuantumProcessor object to use')
    parser.add_argument('--position',
                        type=str,
                        help='FEN representation of the initial position')
    return parser.parse_args()


if __name__ == '__main__':
    main_loop(parse())
