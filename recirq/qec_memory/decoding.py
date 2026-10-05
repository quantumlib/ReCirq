# Copyright 2026 Google
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

import cirq
import pymatching
import stim
import stimcirq
import stimflow
import re
import numpy as np
import pandas as pd
from typing import Sequence


def _get_pre_loop_key(keys: Sequence[str]) -> int:
    """Find the last measure key before the ones from within the CircuitOperation.

    Args:
        keys: A list of the measure keys.

    Returns:
        The key as an integer.

    Raises:
        ValueError: If keys does not contain some non-consecutive integers strings.
    """
    digit_keys = sorted([int(key) for key in keys if key.isdigit()])
    for i in range(len(digit_keys) - 1):
        if digit_keys[i + 1] - digit_keys[i] > 1:
            return digit_keys[i]
    raise ValueError(
        "`keys` must contain some integer strings and they must not all be consecutive"
    )


def cirq_df_to_stim_array(data: pd.DataFrame) -> np.ndarray:
    """
    Convert from cirq.Result.data to the format expected by stim.

    Args:
        data: The measurement, as returned by cirq.Result.data.

    Returns:
        The measurement array expected by stim.
    """
    pre_loop_key = _get_pre_loop_key(data.columns)

    def _parse_column_name(col_name: str) -> tuple[int, int]:
        """Used for sorting measurement key names to match the order expcted
        by stim.

        Args:
            col_name: The measurement key.

        Returns:
            A tuple that can be used to sort the keys.
        """
        col_str = str(col_name)
        re_match = re.match(r"(\d+)\[(\d+)\]", col_str)
        if re_match:
            return int(re_match.group(2)) + 1, int(re_match.group(1))

        if col_str.isdigit():
            if int(col_str) <= pre_loop_key:
                return 0, int(col_str)
            else:
                return int(col_str) + len(data.columns), 0  # sends these to the end

    df_sorted = data.sort_index(axis=1, key=lambda cols: cols.map(_parse_column_name))
    return df_sorted.values.astype(dtype=np.bool_)


def cirq_to_stim_hardware_circuit(hardware_circuit: cirq.Circuit) -> stim.Circuit:
    """Convert a cirq circuit meant for running on hardware to a stim circuit.

    Args:
        hardware_circuit: The cirq circuit meant to run on hardware.

    Returns:
        A corresponding stim circuit.
    """
    zero_adept_phases = {key: 0 for key in hardware_circuit._parameter_names_()}
    hardware_circuit_no_adept = cirq.resolve_parameters(
        hardware_circuit, zero_adept_phases
    )
    hardware_circuit_no_adept = cirq.transformers.apply_lazy_args_on_circuit_operation(
        hardware_circuit_no_adept
    )
    return stimcirq.cirq_circuit_to_stim_circuit(hardware_circuit_no_adept)


def create_dem(
    hardware_stim_circuit: stim.Circuit, si_1000_error_rate: float
) -> stim.DetectorErrorModel:
    """Create a detector error model for a given circuit, assuming the si1000 noise model.

    Args:
        hardware_stim_circuit: The circuit that was run on hardware, converted to a stim circuit.
        si_1000_error_rate: The parameter p in stimflow.NoiseModel.si1000.

    Returns:
        The detector error model
    """
    noisy_hardware_stim_circuit = stimflow.NoiseModel.si1000(
        p=si_1000_error_rate
    ).noisy_circuit(hardware_stim_circuit)
    return noisy_hardware_stim_circuit.detector_error_model()


def extract_detection_events(
    data: pd.DataFrame, hardware_stim_circuit: stim.Circuit
) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw measurement outcomes into detection events.

    Args:
        data: The measurement results, formatted as in cirq.Result.data.
        hardware_stim_circuit: The circuit that was run on hardware, converted to a stim circuit.

    Returns:
        The detection event data and the observable flip data. See
            stim.CompiledMeasurementsToDetectionEventsConverter.convert
    """
    meas_arr = cirq_df_to_stim_array(data)
    m2d = hardware_stim_circuit.compile_m2d_converter()
    return m2d.convert(
        measurements=meas_arr,
        sweep_bits=None,
        append_observables=False,
        separate_observables=True,
        bit_packed=False,
    )


def decode(detection_events: np.ndarray, dem: stim.DetectorErrorModel) -> np.ndarray:
    """Decode the measured detection events using pymatching.

    Args:
        detection_events: The measured detection events. The first output
            argument of extract_detection_events.
        dem: The detector error model, can be constructed using create_dem.

    Returns:
        The decoded observables.
    """
    matcher = pymatching.Matching.from_detector_error_model(dem)
    return matcher.decode_batch(
        detection_events, bit_packed_shots=False, bit_packed_predictions=False
    )


def get_logical_error_probability(
    data: pd.DataFrame, circuit: cirq.Circuit, si_1000_error_rate: float
) -> list[float]:
    """Decode using pymatching and get the logical error probability.

    Args:
        data: The measurement results, formatted as in cirq.Result.data.
        circuit: The circuit that was run on hardware.
        si_1000_error_rate: The parameter p in stimflow.NoiseModel.si1000.

    Returns:
        The logical error probability.
    """
    hardware_stim_circuit = cirq_to_stim_hardware_circuit(circuit)
    dem = create_dem(hardware_stim_circuit, si_1000_error_rate)
    dets_arr, obs_arr = extract_detection_events(data, hardware_stim_circuit)
    predictions = decode(dets_arr, dem)
    decoded_obs = predictions ^ obs_arr
    return decoded_obs.mean()
