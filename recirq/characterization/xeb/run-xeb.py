from recirq.characterization.xeb.grid_parallel_two_qubit_xeb import CircuitGenerationTask, \
    generate_grid_parallel_two_qubit_circuits


def main():
    cgen_task = CircuitGenerationTask(
        dataset_id='2020-09',
        topology_name='2x2-grid',
        two_qubit_gate_name='fsim_pi4',
        n_circuits=20,
        max_cycles=203,
        seed=52,
    )
    generate_grid_parallel_two_qubit_circuits(cgen_task)


if __name__ == '__main__':
    main()
