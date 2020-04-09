import asyncio
import datetime
import random

from recirq.qaoa.experiments.angle_precomputation_tasks import AnglePrecomputationTask
from recirq.qaoa.experiments.precomputed_execution_tasks import collect_data, \
    PrecomputedDataCollectionTask
from recirq.qaoa.experiments.problem_generation_tasks import HardwareGridProblemGenerationTask, \
    SKProblemGenerationTask, ThreeRegularProblemGenerationTask
import recirq


async def main():
    pgen_dataset_id = '2020-03-19'
    hardware_grid_tasks = [
        HardwareGridProblemGenerationTask(
            dataset_id=pgen_dataset_id,
            device_name='Sycamore23',
            instance_i=i,
            n_qubits=n
        )
        for n in list((range(2, 23 + 1)))
        for i in range(10)
    ]
    sk_problem_tasks = [
        SKProblemGenerationTask(
            dataset_id=pgen_dataset_id,
            instance_i=i,
            n_qubits=n
        )
        for n in list(range(3, 17 + 1))
        for i in range(10)
    ]
    three_regular_tasks = [
        ThreeRegularProblemGenerationTask(
            dataset_id=pgen_dataset_id,
            instance_i=i,
            n_qubits=n
        )
        for n in list(range(3, 23 + 1)) if 3 * n % 2 == 0
        for i in range(10)
    ]

    apre_dataset_id = '2020-03-23'
    precompute_tasks = []
    for gen_task in recirq.roundrobin(
            hardware_grid_tasks,
            sk_problem_tasks,
            three_regular_tasks,
    ):
        if isinstance(gen_task, HardwareGridProblemGenerationTask):
            p_vals = [1, 2, 3, 4, 5]
        elif isinstance(gen_task, SKProblemGenerationTask):
            p_vals = [1, 2, 3]
        elif isinstance(gen_task, ThreeRegularProblemGenerationTask):
            p_vals = [1, 2, 3]
        else:
            raise ValueError()
        for p in p_vals:
            precompute_tasks.append(
                AnglePrecomputationTask(
                    dataset_id=apre_dataset_id,
                    generation_task=gen_task,
                    p=p)
            )

    dcol_dataset_id = datetime.datetime.now().isoformat(timespec='minutes')
    data_collection_tasks = [
        PrecomputedDataCollectionTask(
            dataset_id=dcol_dataset_id,
            precomputation_task=pre_task,
            device_name='Syc23-zeros',  # NOTE: change to a real device
            n_shots=50_000,
            structured=True,
        )
        for pre_task in precompute_tasks
    ]
    random.shuffle(data_collection_tasks)

    await recirq.execute_in_queue(collect_data, data_collection_tasks, num_workers=5)


if __name__ == '__main__':
    asyncio.run(main())
