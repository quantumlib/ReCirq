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

import os
import uuid
from datetime import datetime

import pytest
from unittest.mock import patch

from google.protobuf.timestamp_pb2 import Timestamp
import cirq
from cirq.google.engine import EngineTimeSlot
from cirq.google.engine.client.quantum_v1alpha1.gapic import enums
from cirq.google.engine.client.quantum_v1alpha1 import types as qtypes
import recirq
from recirq.engine_utils import _get_program_id



def test_get_program_id():
    circuit = cirq.Circuit(cirq.H(cirq.LineQubit(0)))
    prog_id = _get_program_id(circuit)
    assert isinstance(prog_id, str)
    assert uuid.UUID(prog_id, version=4)


def test_get_program_id_2():
    circuit = cirq.Circuit(cirq.H(cirq.LineQubit(0)))
    circuit.program_id = "my_fancy_var_1/my_fancy_var_2/x-3"
    prog_id = _get_program_id(circuit)
    assert isinstance(prog_id, str)
    with pytest.raises(ValueError):
        val = uuid.UUID(prog_id, version=4)
    assert prog_id.startswith("my_fancy_var_1_my_fancy_var_2_x-3")


def test_get_program_id_3():
    circuit = cirq.Circuit(cirq.H(cirq.LineQubit(0)))
    circuit.program_id = ("my_fancy_var_1/my_fancy_var_2/my_fancy_var_3/"
                          "my_fancy_var_4/my_fancy_var_5/my_fancy_var_6")
    assert len(circuit.program_id) > 64
    prog_id = _get_program_id(circuit)
    assert isinstance(prog_id, str)
    with pytest.raises(ValueError):
        val = uuid.UUID(prog_id, version=4)
    assert prog_id.startswith(
        "my_far_1_my_far_2_my_far_3_my_far_4_my_far_5_my_far_6_")


def test_get_program_id_4():
    circuit = cirq.Circuit(cirq.H(cirq.LineQubit(0)))
    circuit.program_id = "abcd1234/" * 20
    assert len(circuit.program_id) > 64
    prog_id = _get_program_id(circuit)
    assert isinstance(prog_id, str)
    # too many parts, it defaults back to uuid
    val = uuid.UUID(prog_id, version=4)


def test_zeros_sampler_one_big_measure():
    qubits = cirq.LineQubit.range(6)
    circuit = cirq.Circuit(cirq.H.on_each(*qubits),
                           cirq.measure(*qubits, key='asdf'))

    sampler = recirq.ZerosSampler()
    result = sampler.run(circuit, repetitions=155)
    assert len(result.measurements) == 1
    bitstrings = result.measurements['asdf']
    assert bitstrings.shape == (155, 6)


def test_zeros_sampler_many_measure():
    qubits = cirq.LineQubit.range(6)
    circuit = cirq.Circuit(cirq.H.on_each(*qubits),
                           cirq.measure_each(*qubits, key_func=str))
    sampler = recirq.ZerosSampler()
    result = sampler.run(circuit, repetitions=155)
    assert len(result.measurements) == 6
    for k, v in result.measurements.items():
        assert v.shape == (155, 1)


def test_device_obj():
    assert recirq.get_device_obj_by_name('Sycamore23') \
           == cirq.google.Sycamore23


def test_processor_id():
    assert recirq.get_processor_id_by_device_name('Sycamore23') == 'rainbow'
    assert recirq.get_processor_id_by_device_name('Syc23-simulator') is None


def test_sampler_by_name():
    if 'GOOGLE_CLOUD_PROJECT' in os.environ:
        assert isinstance(recirq.get_sampler_by_name('Sycamore23'),
                          recirq.EngineSampler)
    assert isinstance(recirq.get_sampler_by_name('Syc23-simulator'),
                      cirq.DensityMatrixSimulator)
    assert isinstance(recirq.get_sampler_by_name('Syc23-noiseless'),
                      cirq.Simulator)
    assert isinstance(recirq.get_sampler_by_name('Syc23-zeros'),
                      recirq.ZerosSampler)
    assert isinstance(recirq.get_sampler_by_name('Syc23-zeros',
                                                 gateset='sqrt-iswap'),
                      recirq.ZerosSampler)


@patch('cirq.google.engine.engine_client.quantum.QuantumEngineServiceClient')
@patch('recirq.engine_utils._get_current_time')
@patch('cirq.google.engine.EngineProcessor.get_schedule')
def test_get_available_processors_open_swim_in_time_window(schedule_mock, time_mock, engine_mock):
    os.environ['GOOGLE_CLOUD_PROJECT'] = 'some_project'
    schedule_mock.return_value = [
        qtypes.QuantumTimeSlot(
            processor_name='Sycamore23',
            start_time=Timestamp(seconds=100),
            end_time=Timestamp(seconds=500),
            slot_type=enums.QuantumTimeSlot.TimeSlotType.OPEN_SWIM)
    ]
    time_mock.return_value = datetime.fromtimestamp(300)
    assert 'Sycamore23' in recirq.get_available_processors(['Sycamore23'])


@patch('cirq.google.engine.engine_client.quantum.QuantumEngineServiceClient')
@patch('recirq.engine_utils._get_current_time')
@patch('cirq.google.engine.EngineProcessor.get_schedule')
def test_get_available_processors_open_swim_outside_window(schedule_mock, time_mock, engine_mock):
    os.environ['GOOGLE_CLOUD_PROJECT'] = 'some_project'
    schedule_mock.return_value = [
        qtypes.QuantumTimeSlot(
            processor_name='Sycamore23',
            start_time=Timestamp(seconds=300),
            end_time=Timestamp(seconds=500),
            slot_type=enums.QuantumTimeSlot.TimeSlotType.OPEN_SWIM)
    ]
    time_mock.return_value = datetime.fromtimestamp(700)
    assert recirq.get_available_processors(['Sycamore23']) == []


@patch('cirq.google.engine.engine_client.quantum.QuantumEngineServiceClient')
@patch('recirq.engine_utils._get_current_time')
@patch('cirq.google.engine.EngineProcessor.get_schedule')
def test_get_available_processors_current_project_reservation(
        schedule_mock, time_mock, engine_mock):
    os.environ['GOOGLE_CLOUD_PROJECT'] = 'some_project'
    schedule_mock.return_value = [
        qtypes.QuantumTimeSlot(
            processor_name='Sycamore23',
            start_time=Timestamp(seconds=100),
            end_time=Timestamp(seconds=500),
            slot_type=enums.QuantumTimeSlot.TimeSlotType.RESERVATION,
            reservation_config=qtypes.QuantumTimeSlot.ReservationConfig(
                project_id='some_project'),
        )
    ]
    time_mock.return_value = datetime.fromtimestamp(300)
    assert 'Sycamore23' in recirq.get_available_processors(['Sycamore23'])


@patch('cirq.google.engine.engine_client.quantum.QuantumEngineServiceClient')
@patch('recirq.engine_utils._get_current_time')
@patch('cirq.google.engine.EngineProcessor.get_schedule')
def test_get_available_processors_other_project_reservation(
        schedule_mock, time_mock, engine_mock):
    os.environ['GOOGLE_CLOUD_PROJECT'] = 'some_project'
    schedule_mock.return_value = [
        qtypes.QuantumTimeSlot(
            processor_name='Sycamore23',
            start_time=Timestamp(seconds=100),
            end_time=Timestamp(seconds=500),
            slot_type=enums.QuantumTimeSlot.TimeSlotType.RESERVATION,
            reservation_config=qtypes.QuantumTimeSlot.ReservationConfig(
                project_id='other_project'),
        )
    ]
    time_mock.return_value = datetime.fromtimestamp(300)
    assert recirq.get_available_processors(['Sycamore23']) == []
