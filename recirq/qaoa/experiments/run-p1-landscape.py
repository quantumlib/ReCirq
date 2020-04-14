import asyncio
import datetime

import recirq
from recirq.qaoa.experiments.p1_landscape_tasks import get_data_collection_tasks_on_a_grid, \
    collect_either_landscape_or_cal
from recirq.qaoa.experiments.problem_generation_tasks import HardwareGridProblemGenerationTask, \
    SKProblemGenerationTask, ThreeRegularProblemGenerationTask


async def main():
    pgen_dataset_id = '2020-03-19'
    dcol_dataset_id = datetime.datetime.now().isoformat(timespec='minutes')

    hardware_grid_problem_task = HardwareGridProblemGenerationTask(
        dataset_id=pgen_dataset_id,
        device_name='Sycamore23',
        instance_i=0,
        n_qubits=23
    )
    data_collection_tasks = get_data_collection_tasks_on_a_grid(
        pgen_task=hardware_grid_problem_task,
        dataset_id=dcol_dataset_id,
        device_name='Sycamore23',
        epoch="grid")
    await recirq.execute_in_queue(collect_either_landscape_or_cal,
                                  data_collection_tasks,
                                  num_workers=2)

    sk_problem_task = SKProblemGenerationTask(
        dataset_id=pgen_dataset_id,
        instance_i=0,
        n_qubits=12,
    )
    data_collection_tasks = get_data_collection_tasks_on_a_grid(
        pgen_task=sk_problem_task,
        dataset_id=dcol_dataset_id,
        device_name='Sycamore23',
        epoch="sk")
    await recirq.execute_in_queue(collect_either_landscape_or_cal,
                                  data_collection_tasks,
                                  num_workers=2)

    three_regular_problem_task = ThreeRegularProblemGenerationTask(
        dataset_id=pgen_dataset_id,
        instance_i=0,
        n_qubits=12
    )
    data_collection_tasks = get_data_collection_tasks_on_a_grid(
        pgen_task=three_regular_problem_task,
        dataset_id=dcol_dataset_id,
        device_name='Sycamore23',
        epoch="tr")
    await recirq.execute_in_queue(collect_either_landscape_or_cal,
                                  data_collection_tasks,
                                  num_workers=2)


if __name__ == '__main__':
    asyncio.run(main())
