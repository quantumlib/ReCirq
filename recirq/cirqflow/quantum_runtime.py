import abc
import glob
import os
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Sequence, AbstractSet, Dict, Any, Tuple

import cirq.work
import networkx as nx
import numpy as np
from cirq.protocols import dataclass_json_dict

from recirq.cirqflow.quantum_executable import QuantumExecutableGroup, ExecutableSpec, Bitstrings
from recirq.cirqflow.qubit_placement import QubitPlacer, NaiveQubitPlacer, CouldNotPlaceError

SYC23_GRAPH = nx.from_edgelist([
    ((3, 2), (4, 2)), ((4, 1), (5, 1)), ((4, 2), (4, 1)),
    ((4, 2), (4, 3)), ((4, 2), (5, 2)), ((4, 3), (5, 3)),
    ((5, 1), (5, 0)), ((5, 1), (5, 2)), ((5, 1), (6, 1)),
    ((5, 2), (5, 3)), ((5, 2), (6, 2)), ((5, 3), (5, 4)),
    ((5, 3), (6, 3)), ((5, 4), (6, 4)), ((6, 1), (6, 2)),
    ((6, 2), (6, 3)), ((6, 2), (7, 2)), ((6, 3), (6, 4)),
    ((6, 3), (7, 3)), ((6, 4), (6, 5)), ((6, 4), (7, 4)),
    ((6, 5), (7, 5)), ((7, 2), (7, 3)), ((7, 3), (7, 4)),
    ((7, 3), (8, 3)), ((7, 4), (7, 5)), ((7, 4), (8, 4)),
    ((7, 5), (7, 6)), ((7, 5), (8, 5)), ((8, 3), (8, 4)),
    ((8, 4), (8, 5)), ((8, 4), (9, 4)),
])


class QubitsDevice(cirq.Device):
    def __init__(self, qubits: Sequence['cirq.Qid']):
        self.qubits = qubits

    def qubit_set(self) -> Optional[AbstractSet['cirq.Qid']]:
        return frozenset(self.qubits)

    def validate_circuit(self, circuit: 'cirq.Circuit') -> None:
        circuit_qubits = circuit.all_qubits()
        dev_qubits = self.qubit_set()

        bad_qubits = circuit_qubits - dev_qubits
        if bad_qubits:
            # raise ValueError(f"Circuit qubits {bad_qubits} don't exist on "
            #                  f"device with qubits {dev_qubits}.")
            pass  # TODO: care


_QCS_DEVICES = {
    'rainbow-23': QubitsDevice(sorted(SYC23_GRAPH.nodes())),
}

_QCS_MAPPING = {
    'rainbow-23': ('rainbow', 'sqrt_iswap'),
}


class GateSynthesizer:
    pass


class AutoCalibrator:
    pass


import cirq_google as cg

_QCS_NAMES = [
    'rainbow-23',
    'weber',
]

_SIM_NAMES = [
    'zeros-sampler',
    'simulator',
    'noisy-simulator',
]
_REQUIRES_MAX_QUBITS = _SIM_NAMES

_ALL_NAMES = set(_QCS_NAMES) | set(_SIM_NAMES)

_SUPPORTS_NOISE_STRENGTH = [
    'noisy-simulator',
]

_SUPPORTS_MAX_QUBITS = set(_QCS_NAMES) | set(_SIM_NAMES)


class QuantumBackend(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_sampler_and_device(self):
        pass

    @property
    @abc.abstractmethod
    def desc(self):
        pass


@dataclass(frozen=True)
class QCSBackend(QuantumBackend):
    name: str

    @lru_cache()
    def get_sampler_and_device(self):
        processor_id, gate_set_name = _QCS_MAPPING[self.name]
        sampler = cg.get_engine_sampler(processor_id, gate_set_name)
        device = _QCS_DEVICES[self.name]
        return sampler, device

    @property
    @lru_cache()
    def desc(self):
        return self.name

    def _json_dict_(self):
        return dataclass_json_dict(self, namespace='cirq.google')


@dataclass(frozen=True)
class SimulatorBackend(QuantumBackend):
    name: str
    noise_strength: float = 0

    @lru_cache()
    def get_sampler_and_device(self):
        device = _QCS_DEVICES[self.name]
        if self.noise_strength == 0:
            return cirq.Simulator(), device
        if self.noise_strength == float('inf'):
            return cirq.ZerosSampler(), device

        return cirq.DensityMatrixSimulator(noise=cirq.depolarize(p=self.noise_strength)), device

    @property
    @lru_cache()
    def desc(self):
        if self.noise_strength == 0:
            suffix = '-simulator'
        elif self.noise_strength == float('inf'):
            suffix = '-zeros-sampler'
        else:
            suffix = f'-depol-{self.noise_strength:.3e}'
        return self.name + suffix

    def _json_dict_(self):
        return dataclass_json_dict(self, namespace='cirq.google')


@dataclass()
class SharedRuntimeInfo:
    run_id: str

    def _json_dict_(self):
        return dataclass_json_dict(self, namespace='cirq.google')


def _change_dict_to_list_of_items(d: Dict[str, Any], key: str):
    d[key] = sorted(d[key].items())


@dataclass()
class QuantumRuntimeInfo:
    run_id: str
    placement: Dict[Any, cirq.Qid] = None
    timings: Dict[str, float] = None

    def _json_dict_(self):
        json_dict = dataclass_json_dict(self, namespace='cirq.google')
        _change_dict_to_list_of_items(json_dict, 'placement')
        _change_dict_to_list_of_items(json_dict, 'timings')
        return json_dict

    @classmethod
    def _from_json_dict_(cls, run_id: str, placement: List[Tuple[Any, cirq.Qid]],
                         timings: List[Tuple[str, float]], **kwargs):
        return cls(run_id=run_id, placement=dict(placement), timings=dict(timings))


@dataclass()
class RawExecutableResult:
    spec: ExecutableSpec
    runtime_info: QuantumRuntimeInfo
    raw_data: cirq.Result

    def _json_dict_(self):
        return dataclass_json_dict(self, namespace='cirq.google')


@dataclass()
class ExecutionResult:
    runtime_configuration: 'QuantumRuntime'
    shared_runtime_info: SharedRuntimeInfo
    executable_results: List[RawExecutableResult]

    def _json_dict_(self):
        return dataclass_json_dict(self, namespace='cirq.google')


class TimeAndPrint:
    def __init__(self, name=''):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print(f'{self.name} took {self.interval}s')


class TimeIntoRuntimeInfo:
    def __init__(self, runtime_info: QuantumRuntimeInfo, name: str):
        self.runtime_info = runtime_info
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        end = time.time()
        interval = end - self.start
        print(f'{self.name} took {interval}s')
        self.runtime_info.timings[self.name] = interval


@dataclass
class QuantumRuntimeConfiguration:
    """Configuration of the runtime.

    Right now, this is just a backend (sampler, device combo) and a random seed
    but we might add more options later.
    """
    backend: QuantumBackend
    gate_synthesizer: GateSynthesizer = None
    calibrator: AutoCalibrator = None
    readout_corrector = None
    qubit_placer: QubitPlacer = NaiveQubitPlacer()
    batcher = None
    seed: int = None
    run_id: str = None

    def _json_dict_(self):
        return dataclass_json_dict(self, namespace='cirq.google')


def execute(
        rt_config: QuantumRuntimeConfiguration,
        executable_group: QuantumExecutableGroup,
        base_data_dir: str = ".",
) -> ExecutionResult:
    """Execute a bunch of Executables given the runtime configuration."""
    # Set up
    sampler, device = rt_config.backend.get_sampler_and_device()
    if rt_config.run_id is None:
        run_id = str(uuid.uuid4())
    else:
        run_id = rt_config.run_id
    if base_data_dir == "":
        raise ValueError()
    os.makedirs(f'{base_data_dir}/{run_id}', exist_ok=False)
    rs = np.random.RandomState(rt_config.seed)

    # Results object that we will fill in in the main loop.
    full_results = ExecutionResult(runtime_configuration=rt_config,
                                   shared_runtime_info=SharedRuntimeInfo(run_id=run_id),
                                   executable_results=[])
    cirq.to_json_gzip(full_results, f'{base_data_dir}/{run_id}/ExecutionResult.json.gz')

    # The main loop
    print('# Executables:', len(executable_group), flush=True)
    i = 0
    for exe in executable_group:
        runtime_info = QuantumRuntimeInfo(
            run_id=run_id,
            placement={},  # to be filled in
            timings={}  # will be filled in via `TimeIntoRuntimeInfo` context managers.
        )

        if not isinstance(exe.measurement, Bitstrings):
            raise NotImplementedError()

        if exe.params != tuple():
            raise NotImplementedError()

        try:
            # 1. place on device
            with TimeIntoRuntimeInfo(runtime_info, 'placement'):
                circuit, placement = rt_config.qubit_placer.place_circuit(exe.circuit,
                                                                          exe.spec.topology)
                runtime_info.placement = placement
                device.validate_circuit(circuit)

            # 2. run the circuit
            with TimeIntoRuntimeInfo(runtime_info, 'sampler_run'):
                sampler_run_result = sampler.run(circuit, repetitions=exe.spec.n_repetitions)

            # 3. package and save result
            raw_result = RawExecutableResult(
                spec=exe.spec,
                runtime_info=runtime_info,
                raw_data=sampler_run_result,
            )
            cirq.to_json_gzip(
                raw_result, f'{base_data_dir}/{run_id}/RawExecutableResult.{i}.json.gz')
            full_results.executable_results.append(raw_result)

        except CouldNotPlaceError:
            # Ignore topologies that are too big.
            print('x', end=' ')

        print(i, end=' ', flush=True)
        i += 1
    return full_results


def load_raw_results(run_id: str, base_data_dir: str = '.') -> ExecutionResult:
    assert base_data_dir

    full_results = cirq.read_json_gzip(f'{base_data_dir}/{run_id}/ExecutionResult.json.gz')
    full_results.executable_results = [
        cirq.read_json_gzip(fn) for fn in
        glob.glob(f'{base_data_dir}/{run_id}/RawExecutableResult.*.json.gz')]
    return full_results
