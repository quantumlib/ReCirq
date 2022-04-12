import os
from importlib import import_module
from types import ModuleType

import pytest

from recirq.algorithmic_benchmark_library import BENCHMARKS, get_all_algo_configs, workflow, \
    get_algo_benchmark_by_executable_family

RECIRQ_DIR = os.path.abspath(os.path.dirname(__file__) + '/../')

if not workflow:
    pytestmark = pytest.mark.skip('algorithmic_benchmark_library requires pre-release of Cirq.')


@pytest.mark.parametrize('algo', BENCHMARKS)
def test_domain(algo):
    # By convention, the domain should be a recirq module.
    assert algo.domain.startswith('recirq.'), 'domain should be a recirq module.'
    mod = import_module(algo.domain)
    assert isinstance(mod, ModuleType), 'domain should be a recirq module.'


def test_benchmark_name_unique_in_domain():
    # In a given domain, all benchmark names should be unique
    pairs = [(algo.domain, algo.name) for algo in BENCHMARKS]
    assert len(set(pairs)) == len(pairs)


@pytest.mark.parametrize('algo', BENCHMARKS)
def test_executable_family_is_formulaic(algo):
    # Check consistency in the AlgorithmicBenchmark dataclass:
    assert algo.executable_family == algo.spec_class.executable_family, \
        "benchmark's executable_family should match that of the spec_class"

    # By convention, we set this to be the module name. By further convention,
    # {algo.domain}.{algo.name} should be the module name.
    assert algo.executable_family == f'{algo.domain}.{algo.name}', \
        "The executable family should be set to the benchmarks's domain.name"

    # Check the convention that it should give a module
    mod = import_module(algo.executable_family)
    assert isinstance(mod, ModuleType), \
        "The executable family should specify an importable module."


@pytest.mark.parametrize('algo', BENCHMARKS)
def test_classes_and_funcs(algo):
    # The various class objects should exist in the module
    mod = import_module(algo.executable_family)
    assert algo.spec_class == getattr(mod, algo.spec_class.__name__), \
        "The spec_class must exist in the benchmark's module"
    assert algo.executable_generator_func == getattr(mod, algo.executable_generator_func.__name__), \
        "the executable_generator_func must exist in the benchmark's module"


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
    assert config.gen_script == f'gen-{config.short_name}.py', \
        "The gen_script should be of the form 'gen-{short_name}'"

    # Make sure it exists
    gen_script_path = (f"{RECIRQ_DIR}/{algo.domain.replace('.', '/')}/"
                       f"{algo.name.replace('.', '/')}/{config.gen_script}")
    assert os.path.exists(gen_script_path)


def test_get_things_by_name():
    algo = get_algo_benchmark_by_executable_family('recirq.otoc.loschmidt.tilted_square_lattice')
    assert algo == BENCHMARKS[0]

    config = algo.get_config_by_full_name('loschmidt.tilted_square_lattice.small-v1')
    assert config == algo.configs[0]
