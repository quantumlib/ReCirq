import dataclasses
from dataclasses import dataclass
from functools import lru_cache
from typing import Type, Callable, List, Tuple

import pandas as pd

from cirq_google.workflow import ExecutableSpec, QuantumExecutableGroup


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
            By convention, the executable family is the fully-qualified leaf-module where the code
            for this AlgorithmicBenchmark lives.
        spec_class: The ExecutableSpec subclass for this AlgorithmicBenchmark.
        data_class: The class which can contain ETL-ed data for this AlgorithmicBenchmark.
        executable_generator_func: The function that returns a QuantumExecutableGroup for a
            given Config.
        configs: A list of available `BenchmarkConfig` for this benchmark.
    """

    domain: str
    name: str
    executable_family: str
    spec_class: Type[ExecutableSpec]
    data_class: Type
    executable_generator_func: Callable[[...], QuantumExecutableGroup]
    configs: List['BenchmarkConfig']

    def as_strings(self):
        """Get values of this class as strings suitable for printing."""
        ret = {k: str(v) for k, v in dataclasses.asdict(self).items()}
        ret['spec_class'] = self.spec_class.__name__
        ret['data_class'] = self.data_class.__name__
        ret['executable_generator_func'] = self.executable_generator_func.__name__
        return ret


@dataclass
class BenchmarkConfig:
    short_name: str
    """The short name for this config. Unique within an AlgorithmicBenchmark."""

    full_name: str
    """A globally unique name for this config."""

    gen_script: str
    """The script filename that generates the QuantumExecutableGroup for this Config."""

    run_scripts: List[str]
    """A list of script filenames that execute (or can execute) this config."""


BENCHMARKS = [
]


@lru_cache()
def get_all_algo_configs() -> List[Tuple[AlgorithmicBenchmark, BenchmarkConfig]]:
    ret = []
    for algo in BENCHMARKS:
        for config in algo.configs:
            ret.append((algo, config))
    return ret


def main():
    df = pd.DataFrame([algo.as_strings() for algo in BENCHMARKS]).set_index('executable_family')
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(df.transpose())


if __name__ == '__main__':
    main()
