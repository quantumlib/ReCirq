"""
Interactive ascii-based quantum chess game.

Run with:

python python -m recirq.quantum_chess.experiments.interactive_board \
    --position <FEN>

The FEN argument denotes the starting position using FEN
notation (see https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation).

If no position is given, the board is initialized in the
classical chess starting position.

Moves should be entered using the following format:
's1t1t2s2:TYPE:VARIANT'

Where s1 and t1 are the source and target squares using algebraic
notation.  t2 is the second target square for split moves.  s2
is the second source square for merge moves.  Merge moves should
specify '--' for t2.  TYPE is the type of quantum chess move.
VARIANT is the variant (BASIC, CAPTURE, or EXCLUDED).

If no type and variant are included it is assumed to be a
JUMP move with BASIC variant (i.e. a normal chess move).

The various types and variants can be found in the enums.py file.

Examples:
  'e2e4' - Move the (pawn) on e2 to e4 using a basic jump.
  'd7d5' - Move the (pawn) on d7 to d5 using a basic jump.
  'b1a3c3:SPLIT_JUMP:BASIC' - Split the (knight) on b1 to both a3 and c3
  'a3b1--c3:MERGE_JUMP:BASIC' - Merge the (knight) on a3 and c3 back to b1
  'd5e4:PAWN_CAPTURE:CAPTURE' - Capture the pawn on e4 with pawn on d5
  'exit' - exits the quantum chess program

The interactive board uses a simulator with an unconstrained device.
"""
import argparse
import sys

import recirq.quantum_chess.ascii_board as ab
import recirq.quantum_chess.move as m


def main_loop(args):
    b = ab.AsciiBoard()
    if args.position:
        b.load_fen(args.position)
    else:
        b.reset()
    print(b)
    b.board.clear_debug_log()
    for in_str in sys.stdin:
        in_str = in_str.strip()
        if in_str == "exit":
            return
        if not in_str:
            continue
        if ":" not in in_str:
            in_str = in_str + ":JUMP:BASIC"
        in_move = m.Move.from_string(in_str)
        meas = b.apply(in_move)
        print(b.board.circuit)
        b.board.print_debug_log()
        print(f"Measurement outcome = {meas}")
        print("")
        print(b)
        print("")


def parse():
    parser = argparse.ArgumentParser(description="Interactive quantum chess board.")
    parser.add_argument(
        "--position", type=str, help="FEN representation of the initial position"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main_loop(parse())
