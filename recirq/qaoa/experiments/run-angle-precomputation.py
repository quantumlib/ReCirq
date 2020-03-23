import recirq
from recirq.qaoa.experiments.angle_precomputation_tasks import AnglePrecomputationTask, \
    precompute_angles
from recirq.qaoa.experiments.problem_generation_tasks import HardwareGridProblemGenerationTask, \
    SKProblemGenerationTask, ThreeRegularProblemGenerationTask


def main():
    input_dataset_id = '2020-03-19'
    hardware_grid_tasks = [
        HardwareGridProblemGenerationTask(
            dataset_id=input_dataset_id,
            device_name='Sycamore23',
            instance_i=i,
            n_qubits=n
        )
        for n in list((range(2, 23 + 1)))
        for i in range(10)
    ]
    sk_problem_tasks = [
        SKProblemGenerationTask(
            dataset_id=input_dataset_id,
            instance_i=i,
            n_qubits=n
        )
        for n in list(range(3, 18 + 1))
        for i in range(10)
    ]
    three_regular_tasks = [
        ThreeRegularProblemGenerationTask(
            dataset_id=input_dataset_id,
            instance_i=i,
            n_qubits=n
        )
        for n in list(range(3, 23 + 1)) if 3 * n % 2 == 0
        for i in range(10)
    ]

    precompute_tasks = [
        AnglePrecomputationTask(
            dataset_id='2020-03-23',
            generation_task=gen_task,
            p=p)
        for gen_task in recirq.roundrobin(
            hardware_grid_tasks,
            sk_problem_tasks,
            three_regular_tasks)
        for p in range(1, 5 + 1)
    ]

    for task in precompute_tasks:
        precompute_angles(task)


if __name__ == '__main__':
    main()
