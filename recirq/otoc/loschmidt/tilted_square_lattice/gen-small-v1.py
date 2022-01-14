import numpy as np

import cirq
from recirq.otoc.loschmidt.tilted_square_lattice import \
    get_all_tilted_square_lattice_executables

FN = 'loschmidt.tilted_square_lattice.small-v1.json.gz'


def main():
    exes = get_all_tilted_square_lattice_executables(
        min_side_length=2, max_side_length=3, side_length_step=1,
        n_instances=3,
        macrocycle_depths=np.arange(0, 4 + 1, 1))
    print(len(exes), 'executables')

    cirq.to_json_gzip(exes, FN)
    print(f'Wrote {FN}')


if __name__ == '__main__':
    main()
