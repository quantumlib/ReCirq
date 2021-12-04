import os
from importlib import import_module
from types import ModuleType

import pytest

from recirq.algorithmic_benchmark_library import BENCHMARKS, get_all_algo_configs

RECIRQ_DIR = os.path.abspath(os.path.dirname(__file__) + '/../')


@pytest.mark.parametrize('algo', BENCHMARKS)
def test_domain(algo):
    # By convention, the domain should be a recirq module.
    assert algo.domain.startswith('recirq.')
    mod = import_module(algo.domain)
    assert isinstance(mod, ModuleType)


def test_benchmark_name_unique_in_domain():
    # In a given domain, all benchmark names should be unique
    pairs = [(algo.domain, algo.name) for algo in BENCHMARKS]
    assert len(set(pairs)) == len(pairs)


@pytest.mark.parametrize('algo', BENCHMARKS)
def test_executable_family_is_formulaic(algo):
    # Check consistency in the AlgorithmicBenchmark dataclass:
    assert algo.executable_family == algo.spec_class.executable_family

    # By convention, we set this to be the module name. By further convention,
    # {algo.domain}.{algo.name} should be the module name.
    assert algo.executable_family == f'{algo.domain}.{algo.name}'

    # Check the convention that it should give a module
    mod = import_module(algo.executable_family)
    assert isinstance(mod, ModuleType)


@pytest.mark.parametrize('algo', BENCHMARKS)
def test_classes_and_funcs(algo):
    # The various class objects should exist in the module
    mod = import_module(algo.executable_family)
    assert algo.spec_class == getattr(mod, algo.spec_class.__name__)
    assert algo.data_class == getattr(mod, algo.data_class.__name__)
    assert algo.gen_func == getattr(mod, algo.gen_func.__name__)


def test_globally_unique_executable_family():
    # Each entry should have a unique executable family
    fams = [algo.executable_family for algo in BENCHMARKS]
    assert len(set(fams)) == len(fams)


def test_globally_unique_config_full_name():
    full_names = [config.full_name for algo, config in get_all_algo_configs()]
    assert len(set(full_names)) == len(full_names)


@pytest.mark.parametrize('algo_config', get_all_algo_configs())
def test_gen_script(algo_config):
    algo, config = algo_config

    # Make sure it's formulaic
    assert config.gen_script == f'gen-{config.short_name}.py'

    # Make sure it exists
    gen_script_path = (f"{RECIRQ_DIR}/{algo.domain.replace('.', '/')}/"
                       f"{algo.name.replace('.', '/')}/{config.gen_script}")
    assert os.path.exists(gen_script_path)
