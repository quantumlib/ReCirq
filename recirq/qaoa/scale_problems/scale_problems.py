import os
import random
from collections import defaultdict
from functools import lru_cache

import numpy as np
from multiprocessing import Pool

import recirq
from recirq.qaoa.classical_angle_optimization import optimize_instance_interp_heuristic
from recirq.qaoa.problems import asymmetric_coupling_3reg, asymmetric_coupling_ferromagnet_chain, \
    beta_distributed_sk, partially_rounded_sk, asymmetric_coupling_ferromagnet_grid

EXPERIMENT_NAME = 'qaoa-scale-scale'
DEFAULT_BASE_DIR = os.path.expanduser(f'~/cirq-results/{EXPERIMENT_NAME}')


@recirq.json_serializable_dataclass(namespace='recirq.qaoa',
                                    registry=recirq.Registry,
                                    frozen=True)
class ScaleProblemTask:
    dataset_id: str
    problem_family: str
    n_qubits: int
    other_coupling: float
    p_max: int
    instance_i: int

    @property
    def fn(self):
        return (f'{self.dataset_id}/'
                f'{self.problem_family}/'
                f'n-{self.n_qubits}/'
                f'oc-{self.other_coupling:.3f}/'
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
def _all_random_ferro_chain(
        other_coupling: float,
        seed=52,
        max_n_qubits=50,
        max_n_instances=100,
):
    rs = np.random.RandomState(seed)
    problems = defaultdict(list)
    for n in range(2, max_n_qubits + 1):
        for _ in range(max_n_instances):
            problems[n].append(asymmetric_coupling_ferromagnet_chain(
                n=n, rs=rs, other_coupling=other_coupling, shuffle=True))
    return problems


@lru_cache()
def _all_partial_round_sk(
        round_factor: float,
        seed=52,
        max_n_qubits=50,
        max_n_instances=100,
):
    rs = np.random.RandomState(seed)
    problems = defaultdict(list)
    for n in range(2, max_n_qubits + 1):
        for _ in range(max_n_instances):
            problems[n].append(partially_rounded_sk(
                n=n, rs=rs, round_factor=round_factor))
    return problems


@lru_cache()
def _all_beta_sk(
        log_beta_ab: float,
        seed=52,
        max_n_qubits=50,
        max_n_instances=100,
):
    rs = np.random.RandomState(seed)
    problems = defaultdict(list)
    for n in range(2, max_n_qubits + 1):
        for _ in range(max_n_instances):
            problems[n].append(beta_distributed_sk(
                n=n, rs=rs, shape_param=2 ** log_beta_ab))
    return problems


def get_graph(n: int, problem_family: str, other_coupling: float, instance_i: int):
    if problem_family == '3-regular':
        return _all_3reg(other_coupling=other_coupling)[n][instance_i]
    if problem_family == 'ferro-chain':
        if instance_i != 0:
            raise ValueError("For non-random problems, it doesn't "
                             "make sense to have instance_i != 0")
        return asymmetric_coupling_ferromagnet_chain(
            n=n, rs=None, other_coupling=other_coupling, shuffle=False)
    if problem_family == 'ferro-grid':
        if instance_i != 0:
            raise ValueError("For non-random problems, it doesn't "
                             "make sense to have instance_i != 0")
        return asymmetric_coupling_ferromagnet_grid(
            n=n, rs=None, other_coupling=other_coupling, shuffle=False)
    if problem_family == 'random-ferro-chain':
        return _all_random_ferro_chain(other_coupling=other_coupling)[n][instance_i]

    if problem_family == 'beta-sk':
        return _all_beta_sk(log_beta_ab=other_coupling)[n][instance_i]

    if problem_family == 'partial-round-sk':
        return _all_partial_round_sk(round_factor=other_coupling)[n][instance_i]

    raise ValueError(f"Unknown problem family {problem_family}.")


def simulate_scale_problem(task: ScaleProblemTask, base_dir=None):
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR

    if recirq.exists(task, base_dir=base_dir):
        print(f"{task.fn} already exists. Skipping.")
        return

    print(f"{task.fn}")

    graph = get_graph(n=task.n_qubits, problem_family=task.problem_family,
                      other_coupling=task.other_coupling, instance_i=task.instance_i)
    results = optimize_instance_interp_heuristic(graph=graph, p_max=task.p_max,
                                                 flag_z2_sym=False,
                                                 verbose=True)

    recirq.save(task=task, data={
        'results': results,
    }, base_dir=base_dir)


def main():
    tasks = []
    for problem_family in ['3-regular', 'ferro-chain', 'random-ferro-chain',
                           'partial-round-sk']:
        # for n in [8, 10, 14]:
        for n in [14]:
            for oc in np.linspace(1, 0, 20)[:-1]:
                if problem_family in ['ferro-chain', 'ferro-grid']:
                    instance_is = [0]
                else:
                    instance_is = list(range(10))

                for instance_i in instance_is:
                    tasks += [ScaleProblemTask(
                        dataset_id='v1',
                        problem_family=problem_family,
                        n_qubits=n,
                        other_coupling=oc,
                        p_max=10,
                        instance_i=instance_i,
                    )]

    for problem_family in ['ferro-grid']:
        for n in [12]:
            for oc in np.linspace(1, 0, 20)[:-1]:
                if problem_family in ['ferro-grid']:
                    instance_is = [0]
                else:
                    instance_is = list(range(10))

                for instance_i in instance_is:
                    tasks += [ScaleProblemTask(
                        dataset_id='v1',
                        problem_family=problem_family,
                        n_qubits=n,
                        other_coupling=oc,
                        p_max=10,
                        instance_i=instance_i,
                    )]

    for problem_family in ['beta-sk']:
        for n in [14]:
            for log_beta_ab in np.linspace(-3, 2, 20 - 1):
                for instance_i in range(10):
                    tasks += [ScaleProblemTask(
                        dataset_id='v1',
                        problem_family=problem_family,
                        n_qubits=n,
                        other_coupling=log_beta_ab,
                        p_max=10,
                        instance_i=instance_i,
                    )]

    random.shuffle(tasks)
    with Pool() as pool:
        pool.map(simulate_scale_problem, tasks)


if __name__ == '__main__':
    main()
