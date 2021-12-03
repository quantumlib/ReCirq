import dataclasses
from dataclasses import dataclass
from functools import lru_cache
from typing import Type, Callable, List, Tuple

import pandas as pd

from cirq_google.workflow import ExecutableSpec, QuantumExecutableGroup


@dataclass
class AlgoBenchmark:
    domain: str
    """The problem domain. Usually a high-level ReCirq module."""

    name: str
    """The benchmark name. Must be unique within the domain."""

    executable_family: str
    """A globally unique identifier for this AlgoBenchmark.
    
    This should match up with this Benchmark's `spec_class.executable_family`.
    
    By convention, the executable family is the fully-qualified leaf-module where the code
    for this AlgoBenchmark lives.
    """

    spec_class: Type[ExecutableSpec]
    """The ExecutableSpec subclass for this AlgoBenchmark."""

    data_class: Type
    """The class which can contain ETL-ed data for this AlgoBenchmark."""

    gen_func: Callable[[...], QuantumExecutableGroup]
    """The function that returns a QuantumExecutableGroup for a given Config."""

    configs: List['BenchmarkConfig']

    def as_strings(self):
        ret = {k: str(v) for k, v in dataclasses.asdict(self).items()}
        ret['spec_class'] = self.spec_class.__name__
        ret['data_class'] = self.data_class.__name__
        ret['gen_func'] = self.gen_func.__name__
        return ret


@dataclass
class BenchmarkConfig:
    short_name: str
    """The short name for this config. Unique within an AlgoBenchmark."""

    full_name: str
    """A globally unique name for this config."""

    gen_script: str
    """The script filename that generates the QuantumExecutableGroup for this Config."""

    run_scripts: List[str]
    """A list of script filenames that execute (or can execute) this config."""


BENCHMARKS = [
]


@lru_cache()
def get_all_algo_configs() -> List[Tuple[AlgoBenchmark, BenchmarkConfig]]:
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
