import datetime

from recirq.qaoa.experiments.optimization_tasks import (
    OptimizationAlgorithm, OptimizationTask, collect_optimization_data)
from recirq.qaoa.experiments.problem_generation_tasks import (
    HardwareGridProblemGenerationTask, SKProblemGenerationTask,
    ThreeRegularProblemGenerationTask)


def main():
    pgen_dataset_id = '2020-03-19'
    dcol_dataset_id = datetime.datetime.now().isoformat(timespec='minutes')

    hardware_grid_problem_task = HardwareGridProblemGenerationTask(
        dataset_id=pgen_dataset_id,
        device_name='Sycamore23',
        instance_i=0,
        n_qubits=23)
    sk_problem_task = SKProblemGenerationTask(
        dataset_id=pgen_dataset_id,
        instance_i=0,
        n_qubits=12,
    )
    three_regular_problem_task = ThreeRegularProblemGenerationTask(
        dataset_id=pgen_dataset_id, instance_i=0, n_qubits=12)

    hardware_grid_algorithm = OptimizationAlgorithm(
        method='MGD',
        n_shots=25000,
        options={
            'max_iterations': 10,
            'rate': 0.1,
            'sample_radius': 0.1,
            'n_sample_points_ratio': 1.0,
            'rate_decay_exponent': 0.5,
            'stability_constant': 400,
            'sample_radius_decay_exponent': 0.08,
        })
    hardware_grid_optimization_task = OptimizationTask(
        dataset_id=dcol_dataset_id,
        generation_task=hardware_grid_problem_task,
        device_name='Syc23-noiseless',
        p=1,
        algorithm=hardware_grid_algorithm,
        x0=[0.3, 0.2])

    sk_algorithm = OptimizationAlgorithm(method='MGD',
                                         n_shots=25000,
                                         options={
                                             'max_iterations': 10,
                                             'rate': 0.2,
                                             'sample_radius': 0.1,
                                             'n_sample_points_ratio': 1.0,
                                             'rate_decay_exponent': 0.6,
                                             'stability_constant': 400,
                                             'sample_radius_decay_exponent':
                                             0.08,
                                         })
    sk_optimization_task = OptimizationTask(dataset_id=dcol_dataset_id,
                                            generation_task=sk_problem_task,
                                            device_name='Syc23-noiseless',
                                            p=1,
                                            algorithm=sk_algorithm,
                                            x0=[0.15, 0.15])

    three_regular_algorithm = OptimizationAlgorithm(
        method='MGD',
        n_shots=25000,
        options={
            'max_iterations': 10,
            'rate': 0.1,
            'sample_radius': 0.1,
            'n_sample_points_ratio': 1.0,
            'rate_decay_exponent': 0.5,
            'stability_constant': 400,
            'sample_radius_decay_exponent': 0.08,
        })
    three_regular_optimization_task = OptimizationTask(
        dataset_id=dcol_dataset_id,
        generation_task=three_regular_problem_task,
        device_name='Syc23-noiseless',
        p=1,
        algorithm=three_regular_algorithm,
        x0=[0.3, 0.2])

    collect_optimization_data(hardware_grid_optimization_task)
    collect_optimization_data(sk_optimization_task)
    collect_optimization_data(three_regular_optimization_task)


if __name__ == '__main__':
    main()
