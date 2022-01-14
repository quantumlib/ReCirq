import dataclasses
from dataclasses import dataclass
from functools import lru_cache
from typing import Type, Callable, List, Tuple, Any

import pandas as pd

from recirq.otoc.loschmidt.tilted_square_lattice import (
    TiltedSquareLatticeLoschmidtSpec,
    get_all_tilted_square_lattice_executables,
)

try:
    from cirq_google.workflow import ExecutableSpec, QuantumExecutableGroup

    workflow = True
except ImportError as e:
    import os

    if 'RECIRQ_IMPORT_FAILSAFE' in os.environ:
        workflow = False
    else:
        raise ImportError(f"This functionality requires a pre-release version of Cirq: {e}")


@dataclass
class AlgorithmicBenchmark:
    """A record of relevant elements that comprise a benchmark based on a quantum computing
    algorithm of interest.

    Args:
        domain: The problem domain represented as the corresponding high-level ReCirq module.
            This must be a valid module name beginning with "recirq.".
        name: The benchmark name. Must be unique within the domain.
        executable_family: A globally unique identifier for this AlgorithmicBenchmark.
            This should match up with this Benchmark's `spec_class.executable_family`.
            The executable family is the fully-qualified leaf-module where the code
            for this AlgorithmicBenchmark lives.
        spec_class: The ExecutableSpec subclass for this AlgorithmicBenchmark.
        executable_generator_func: The function that returns a QuantumExecutableGroup for a
            given Config.
        configs: A list of available `BenchmarkConfig` for this benchmark.
    """

    domain: str
    name: str
    executable_family: str
    spec_class: Type['ExecutableSpec']
    executable_generator_func: Callable[[Any], 'QuantumExecutableGroup']
    configs: List['BenchmarkConfig']

    def as_strings(self):
        """Get values of this class as strings suitable for printing."""
        ret = {k: str(v) for k, v in dataclasses.asdict(self).items()}
        ret['spec_class'] = self.spec_class.__name__
        ret['executable_generator_func'] = self.executable_generator_func.__name__
        return ret


@dataclass
class BenchmarkConfig:
    """A particular configuration of an AlgorithmicBenchmark

    Args:
        short_name: The short name for this config. Unique within an AlgorithmicBenchmark.
        full_name: A globally unique name for this config.
        gen_script: The script filename that generates the QuantumExecutableGroup for this Config.
            Should begin with prefix "gen-".
        run_scripts: A list of script filenames that execute (or can execute) this config.
    """

    short_name: str
    full_name: str
    gen_script: str
    run_scripts: List[str]


BENCHMARKS = [
    AlgorithmicBenchmark(
        domain='recirq.otoc',
        name='loschmidt.tilted_square_lattice',
        executable_family='recirq.otoc.loschmidt.tilted_square_lattice',
        spec_class=TiltedSquareLatticeLoschmidtSpec,
        executable_generator_func=get_all_tilted_square_lattice_executables,
        configs=[
            BenchmarkConfig(
                short_name='small-v1',
                full_name='loschmidt.tilted_square_lattice.small-v1',
                gen_script='gen-small-v1.py',
                run_scripts=['run-simulator.py']
            )
        ]
    ),
]


@lru_cache()
def get_all_algo_configs() -> List[Tuple[AlgorithmicBenchmark, BenchmarkConfig]]:
    """Return a tuple of (AlgorithmsBenchmark, BenchmarkConfig) for each BenchmarkConfig in
    `BENCHMARKS`.

    This "flattens" the list of BenchmarkConfig from their nested structure in `BENCHMARKS`.
    """
    ret = []
    for algo in BENCHMARKS:
        for config in algo.configs:
            ret.append((algo, config))
    return ret


def print_table_of_benchmarks():
    """Print the AlgorithmicBenchmarks in a tabular form."""
    df = pd.DataFrame([algo.as_strings() for algo in BENCHMARKS]).set_index('executable_family')
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(df.transpose())
