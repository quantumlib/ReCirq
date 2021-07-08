import cirq
import numpy as np

from recirq.algo_benchmarks.loschmidt.loschmidt import get_all_tilted_square_lattice_executables_flat
from recirq.cirqflow.quantum_runtime import TimeAndPrint


def main():
    with TimeAndPrint('Create executables'):
        pg = get_all_tilted_square_lattice_executables_flat(
            min_side_length=2, max_side_length=3, n_instances=3,
            macrocycle_depths=np.arange(0, 8 + 1, 2))
    with TimeAndPrint('Save gzip'):
        cirq.to_json_gzip(pg, 'loschmidt-small-v1.json.gz')


if __name__ == '__main__':
    main()
