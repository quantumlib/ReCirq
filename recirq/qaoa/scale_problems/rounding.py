import os
import random
from collections import defaultdict
from functools import lru_cache

import numpy as np
from multiprocessing import Pool

import recirq
from recirq.qaoa.classical_angle_optimization import optimize_instance_interp_heuristic
from recirq.qaoa.problems import asymmetric_coupling_3reg, asymmetric_coupling_ferromagnet_chain, \
    beta_distributed_sk, partially_rounded_sk, get_all_sk_problems, gaussian_sk
from recirq.qaoa.rounding import round_graph

EXPERIMENT_NAME = 'qaoa-scale-rounding'
DEFAULT_BASE_DIR = os.path.expanduser(f'~/cirq-results/{EXPERIMENT_NAME}')


@recirq.json_serializable_dataclass(namespace='recirq.qaoa',
                                    registry=recirq.Registry,
                                    frozen=True)
class RoundingTask:
    dataset_id: str
    problem_family: str
    n_qubits: int
    rounding: float
    p_max: int
    instance_i: int

    @property
    def fn(self):
        return (f'{self.dataset_id}/'
                f'{self.problem_family}/'
                f'n-{self.n_qubits}/'
                f'round-{self.rounding:.3f}/'
                f'pmax-{self.p_max}/'
                f'{self.instance_i}')


@lru_cache()
def _all_3reg(
        other_coupling: float,
        seed=52,
        max_n_qubits=50,
        max_n_instances=100,
):
    rs = np.random.RandomState(seed)
    problems = defaultdict(list)
    for n in range(4, max_n_qubits + 1, 2):
        for _ in range(max_n_instances):
            problems[n].append(asymmetric_coupling_3reg(
                n=n, rs=rs, other_coupling=other_coupling))
    return problems



@lru_cache()
def _all_sk(
        seed=52,
        max_n_qubits=50,
        max_n_instances=100,
):
    rs = np.random.RandomState(seed)
    problems = defaultdict(list)
    for n in range(2, max_n_qubits + 1):
        for _ in range(max_n_instances):
            problems[n].append(gaussian_sk(n=n, rs=rs))
    return problems

def get_graph(n: int, problem_family: str, instance_i: int):
    if problem_family == '3-regular':
        return _all_3reg(other_coupling=0.5)[n][instance_i]
    if problem_family == 'skg':
        return _all_sk()[n][instance_i]
    if problem_family == 'ferro-chain':
        if instance_i != 0:
            raise ValueError("For non-random problems, it doesn't "
                             "make sense to have instance_i != 0")
        return asymmetric_coupling_ferromagnet_chain(
            n=n, rs=None, other_coupling=0.5, shuffle=False)

    raise ValueError(f"Unknown problem family {problem_family}.")


def simulate_rounding_problem(task: RoundingTask, base_dir=None):
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR

    if recirq.exists(task, base_dir=base_dir):
        print(f"{task.fn} already exists. Skipping.")
        return

    print(f"{task.fn}")
    graph = get_graph(n=task.n_qubits, problem_family=task.problem_family,
                      instance_i=task.instance_i)
    graph2 = round_graph(graph)
    results = optimize_instance_interp_heuristic(graph=graph, p_max=task.p_max,
                                                 flag_z2_sym=False,
                                                 ansatz_modifier_graph=graph2,
                                                 ansatz_modification_factor=task.rounding,
                                                 verbose=True)

    recirq.save(task=task, data={
        'results': results,
    }, base_dir=base_dir)


def main():
    tasks = []
    for problem_family in ['3-regular', 'ferro-chain', 'skg']:
        # for n in [8, 10, 14]:
        for n in [14]:
            for rounding in np.linspace(0, 1, 20):
                if problem_family == 'ferro-chain':
                    instance_is = [0]
                else:
                    instance_is = list(range(10))

                for instance_i in instance_is:
                    tasks += [RoundingTask(
                        dataset_id='v2',
                        problem_family=problem_family,
                        n_qubits=n,
                        rounding=rounding,
                        p_max=10,
                        instance_i=instance_i,
                    )]

    random.shuffle(tasks)
    with Pool() as pool:
        pool.map(simulate_rounding_problem, tasks)


if __name__ == '__main__':
    main()
