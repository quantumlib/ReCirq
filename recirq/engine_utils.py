# Copyright 2020 Google
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import math
import os
import uuid
import datetime
from dataclasses import dataclass, field
from typing import List, Any, Optional, Callable, Dict, Union

import numpy as np

import cirq
from cirq import work, study, circuits, ops
from cirq.google import devices as cg_devices, gate_sets, engine as cg_engine
from cirq.google.engine.engine_job import TERMINAL_STATES
from cirq.google.engine.client.quantum_v1alpha1.gapic import enums
from cirq.google import EngineTimeSlot


def _get_program_id(program: Any):
    """Get a program id from program.program_id.

    This is not an actual attribute of cirq.Circuit, but thanks to the magic
    of python, it can be! If your circuit does not have a program_id,
    this function will return a uuid4().

    Program ids can only contain alphanumeric and -_ so we replace
    "/" and ":" which are common in our data collection idioms.
    Program ids must be unique and sometimes you need to re-try a particular
    experiment, so we append a random component.

    Program ids can only be 64 characters. The following
    compression is performed: the program id is split according to `/`
    and the middle part of each resulting string is omitted to get the
    length below 64. The parts are joined back with _ since `/` is not
    allowed. If your program id is *really* long, we give up and return
    a uuid4().
    """
    if not hasattr(program, 'program_id'):
        return str(uuid.uuid4())

    program_id: str = program.program_id
    program_id = program_id.replace(':', '')
    parts = program_id.split('/')
    parts.append(str(uuid.uuid4()))
    chars_per_part = math.floor(64 / len(parts)) - 1
    if chars_per_part < 3:
        print("Program id too long!")
        return str(uuid.uuid4())

    parts = [p if len(p) <= chars_per_part
             else p[:chars_per_part // 2] + p[-chars_per_part // 2:]
             for p in parts]
    return '_'.join(parts)


class EngineSampler(work.Sampler):
    """Temporary shim; to be replaced with QuantumEngineSampler.

    Missing features from QuantumEngineSampler:
     - Gateset by string name and project_id by environment variable.
       See https://github.com/quantumlib/Cirq/pull/2767.
     - Extracts program_id from an optional attribute on Circuit.
       Potentially to be fixed by using the "tags" feature and
       adding this as an optional attribute to Circuit.
       See https://github.com/quantumlib/Cirq/issues/2816
     - Asynchronous execution
     - No maximum number of requests for results.
       See https://github.com/quantumlib/Cirq/issues/2817

    """

    def __init__(self, processor_id: str, gateset: str):
        project_id = os.environ['GOOGLE_CLOUD_PROJECT']
        engine = cg_engine.Engine(project_id=project_id,
                                  proto_version=cg_engine.ProtoVersion.V2)
        self.engine = engine
        self.processor_id = processor_id
        if gateset == 'sycamore':
            self.gate_set = gate_sets.SYC_GATESET
        elif gateset == 'sqrt-iswap':
            self.gate_set = gate_sets.SQRT_ISWAP_GATESET
        else:
            raise ValueError("Unknown gateset {}".format(gateset))

    def run(
            self,
            program: 'cirq.Circuit',
            param_resolver: 'cirq.ParamResolverOrSimilarType' = None,
            repetitions: int = 1,
    ) -> 'cirq.TrialResult':
        if param_resolver is None:
            param_resolver = study.ParamResolver({})
        return self.engine.run(
            program=program,
            program_id=_get_program_id(program),
            param_resolver=param_resolver,
            repetitions=repetitions,
            processor_ids=[self.processor_id],
            gate_set=self.gate_set,
        )

    def run_sweep(
            self,
            program: 'cirq.Circuit',
            params: 'cirq.Sweepable',
            repetitions: int = 1,
    ) -> List['cirq.TrialResult']:
        return self.engine.run_sweep(
            program=program,
            params=params,
            program_id=_get_program_id(program),
            repetitions=repetitions,
            processor_ids=[self.processor_id],
            gate_set=self.gate_set,
        ).results()

    async def run_async(self, program: 'cirq.Circuit',
                        *, repetitions: int) -> 'cirq.TrialResult':

        program_id = _get_program_id(program)
        engine_job = self.engine.run_sweep(
            program=program,
            program_id=program_id,
            repetitions=repetitions,
            processor_ids=[self.processor_id],
            gate_set=self.gate_set,
        )
        job = engine_job._refresh_job()
        while True:
            if job.execution_status.state in TERMINAL_STATES:
                break
            await asyncio.sleep(1.0)
            job = engine_job._refresh_job()
        print(f"Done: {program_id}")
        engine_job._raise_on_failure(job)
        response = engine_job.context.client.get_job_results(
            engine_job.project_id, engine_job.program_id, engine_job.job_id)
        result = response.result
        v2_parsed_result = cirq.google.api.v2.result_pb2.Result()
        v2_parsed_result.ParseFromString(result.value)
        return engine_job._get_job_results_v2(v2_parsed_result)[0]


class ZerosSampler(work.Sampler):
    """Shim for an object that should be in Cirq.

    See https://github.com/quantumlib/Cirq/issues/2818.
    """

    def run_sweep(
            self,
            program: 'cirq.Circuit',
            params: 'cirq.Sweepable',
            repetitions: int = 1,
    ) -> List['cirq.TrialResult']:
        assert isinstance(program, circuits.Circuit)
        meas = list(program.findall_operations_with_gate_type(
            ops.MeasurementGate))
        if len(meas) == 0:
            raise ValueError()
        elif len(meas) > 1:
            for _, m, _ in meas:
                assert len(m.qubits) == 1
            results = [
                study.TrialResult(
                    params=p,
                    measurements={gate.key: np.zeros(
                        (repetitions, 1), dtype=int)
                        for _, _, gate in meas})
                for p in study.to_resolvers(params)
            ]
        else:
            assert len(meas) == 1
            i, op, gate = meas[0]
            n_qubits = len(op.qubits)
            k = gate.key
            results = [
                study.TrialResult(
                    params=p,
                    measurements={k: np.zeros(
                        (repetitions, n_qubits), dtype=int)})
                for p in study.to_resolvers(params)
            ]
        return results

    async def run_async(self, program: 'cirq.Circuit',
                        *, repetitions: int) -> 'cirq.TrialResult':
        program_id = _get_program_id(program)

        await asyncio.sleep(0.1)
        results = self.run_sweep(program, study.UnitSweep, repetitions)
        print(f"Done: {program_id}")
        return results[0]


@dataclass(frozen=True)
class QuantumProcessor:
    """Grouping of relevant info

    https://github.com/quantumlib/Cirq/issues/2820
    """
    name: str
    device_obj: cirq.Device
    processor_id: Optional[str]
    is_simulator: bool
    _cached_samplers: Dict[Union[None, str], cirq.Sampler] \
        = field(default_factory=dict)
    _get_sampler_func: Callable[['QuantumProcessor', str], cirq.Sampler] = None

    def get_sampler(self, gateset: str = None):
        """Why must gateset be supplied?

        https://github.com/quantumlib/Cirq/issues/2819
        """
        if gateset not in self._cached_samplers:
            sampler = self._get_sampler_func(self, gateset)
            self._cached_samplers[gateset] = sampler
        return self._cached_samplers[gateset]


class EngineQuantumProcessor:
    def __init__(self, processor_id: str):
        self.name = processor_id
        self.processor_id = processor_id
        self.is_simulator = False
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            project_id = os.environ['GOOGLE_CLOUD_PROJECT']
            engine = cg_engine.Engine(project_id=project_id,
                                      proto_version=cg_engine.ProtoVersion.V2)
            self._engine = engine
        return self._engine

    def get_sampler(self, gateset: str = None):
        if gateset == 'sycamore':
            gateset = gate_sets.SYC_GATESET
        elif gateset == 'sqrt-iswap':
            gateset = gate_sets.SQRT_ISWAP_GATESET
        else:
            raise ValueError("Unknown gateset {}".format(gateset))
        return self.engine.sampler(processor_id=self.processor_id, gate_set=gateset)

    @property
    def device_obj(self):
        dspec = self.engine.get_processor(self.processor_id).get_device_specification()
        device = cg_devices.SerializableDevice.from_proto(proto=dspec, gate_sets=[])
        return device


QUANTUM_PROCESSORS = {
    'Sycamore23': QuantumProcessor(
        name='Sycamore23',
        device_obj=cg_devices.Sycamore23,
        processor_id='rainbow',
        is_simulator=False,
        _get_sampler_func=lambda x, gs: EngineSampler(
            processor_id=x.processor_id, gateset=gs),
    ),
    'Syc23-noiseless': QuantumProcessor(
        name='Syc23-noiseless',
        device_obj=cg_devices.Sycamore23,
        processor_id=None,
        is_simulator=True,
        _get_sampler_func=lambda x, gs: cirq.Simulator(),
    ),
    'Syc23-simulator': QuantumProcessor(
        name='Syc23-simulator',
        device_obj=cg_devices.Sycamore23,
        processor_id=None,
        is_simulator=True,
        _get_sampler_func=lambda x, gs: cirq.DensityMatrixSimulator(
            noise=cirq.ConstantQubitNoiseModel(
                qubit_noise_gate=cirq.DepolarizingChannel(0.005)
            ))
    ),
    'Syc23-simulator-tester': QuantumProcessor(
        # This simulator has a constant seed for consistent testing
        name='Syc23-simulator-tester',
        device_obj=cg_devices.Sycamore23,
        processor_id=None,
        is_simulator=True,
        _get_sampler_func=lambda x, gs: cirq.DensityMatrixSimulator(
            noise=cirq.ConstantQubitNoiseModel(
                qubit_noise_gate=cirq.DepolarizingChannel(0.005)
            ), seed=1234)
    ),
    'Syc23-zeros': QuantumProcessor(
        name='Syc23-zeros',
        device_obj=cg_devices.Sycamore23,
        processor_id=None,
        is_simulator=True,
        _get_sampler_func=lambda x, gs: ZerosSampler()
    ),
    'Syc54-noiseless': QuantumProcessor(
        name='Syc54-noiseless',
        device_obj=cg_devices.Sycamore,
        processor_id=None,
        is_simulator=True,
        _get_sampler_func=lambda x, gs: cirq.Simulator(),
    ),
    'Syc54-simulator': QuantumProcessor(
        name='Syc54-simulator',
        device_obj=cg_devices.Sycamore,
        processor_id=None,
        is_simulator=True,
        _get_sampler_func=lambda x, gs: cirq.DensityMatrixSimulator(
            noise=cirq.ConstantQubitNoiseModel(
                qubit_noise_gate=cirq.DepolarizingChannel(0.005)
            ))
    ),
    'Syc54-zeros': QuantumProcessor(
        name='Syc54-zeros',
        device_obj=cg_devices.Sycamore,
        processor_id=None,
        is_simulator=True,
        _get_sampler_func=lambda x, gs: ZerosSampler()
    )
}


def get_device_obj_by_name(device_name: str):
    return QUANTUM_PROCESSORS[device_name].device_obj


def get_processor_id_by_device_name(device_name: str):
    return QUANTUM_PROCESSORS[device_name].processor_id


def get_sampler_by_name(device_name: str, *,
                        gateset='sycamore'):
    return QUANTUM_PROCESSORS[device_name].get_sampler(gateset)


async def execute_in_queue(func, tasks, num_workers: int):
    """Maintain a respectful queue of work

    Args:
        func: This function will be called on each param
        tasks: Call func on each of these
        num_workers: The number of async workers. This corresponds roughly
            to the maintained queue depth.
    """
    queue = asyncio.Queue()

    async def worker():
        while True:
            task = await queue.get()
            print(f"Processing {task.fn}. Current queue size: {queue.qsize()}")
            await func(task)
            print(f"{task.fn} completed")
            queue.task_done()

    worker_jobs = [asyncio.create_task(worker()) for _ in range(num_workers)]
    for task in tasks:
        await queue.put(task)
    print("Added everything to the queue. Current queue size: {}"
          .format(queue.qsize()))
    await queue.join()
    for wjob in worker_jobs:
        wjob.cancel()


def _get_current_time():
    return datetime.datetime.now()


def get_available_processors(processor_names: List[str]):
    """Returns a list of available processors.

    Checks the reservation status of the processors and returns a list of
    processors that are available to run on at the present time.

    Args:
        processor_names: A list of processor names which are keys from QUANTUM_PROCESSORS.
    """
    project_id = os.environ['GOOGLE_CLOUD_PROJECT']
    engine = cirq.google.get_engine()
    available_processors = []
    current_time = _get_current_time()
    for processor_name in processor_names:
        processor_id = get_processor_id_by_device_name(processor_name)
        if processor_id is None:
            # Skip the check if this is a simulator.
            continue
        processor = engine.get_processor(processor_id)
        for time_slot in processor.get_schedule():
            try:
                time_slot = EngineTimeSlot.from_proto(time_slot)
            # Parsing the end_time of the last time range in the schedule might give throw an error.
            # TODO: remove this check once it is fixed in cirq.
            except ValueError as e:
                continue
            # Ignore time slots that do not contain the current time.
            if time_slot.start_time < current_time < time_slot.end_time:
                # Time slots need to be either in OPEN_SWIM or reserved by the
                # current project to be considered available.
                if (time_slot.slot_type
                        == enums.QuantumTimeSlot.TimeSlotType.OPEN_SWIM
                   ) or (time_slot.slot_type
                         == enums.QuantumTimeSlot.TimeSlotType.RESERVATION and
                         time_slot.project_id == project_id):
                    available_processors.append(processor_name)
    return available_processors
