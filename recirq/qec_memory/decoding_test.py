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
import numpy as np
import pandas as pd
import pytest
import stim
import stimcirq

import recirq.qec_memory.decoding as qec_decoding


def _generate_fake_qec_data(
    keys: list[str], error_rate: float, shots: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Generate fake QEC data for testing.

    Args:
        keys: The measurement keys.
        error_rate: The probability with which to add noise.
        shots: The number of shots to use.
        rng: The psueodrandom number generator

    Returns:
        A dataframe of the format returned by Cirq.
    """
    return pd.DataFrame.from_dict(
        {
            key: dict(
                zip(
                    np.arange(shots),
                    rng.choice([0, 1], size=shots, p=[1 - error_rate, error_rate]),
                )
            )
            for key in keys
        }
    )


def test_get_pre_loop_key():
    keys = ["0", "1", "2", "3[0]", "3[1]", "3[2]", "4[0]", "4[1]", "4[2]", "5", "6"]
    assert qec_decoding._get_pre_loop_key(keys) == 2


@pytest.mark.skipif(
    not hasattr(cirq.transformers, "apply_lazy_args_on_circuit_operation"),
    reason="Requires pre-release version of Cirq",
)
def test_get_logical_error_probability():
    rng = np.random.default_rng(0)
    keys = [
        "0",
        "1",
        "2[0]",
        "2[1]",
        "2[2]",
        "2[3]",
        "3[0]",
        "3[1]",
        "3[2]",
        "3[3]",
        "10",
        "11",
        "12",
    ]
    shots = 10
    stim_circuit = stim.Circuit.generated(
        "repetition_code:memory", rounds=5, distance=3
    )
    circuit = stimcirq.stim_circuit_to_cirq_circuit(stim_circuit)
    for error_rate, expected_lep in zip([0.0, 0.01, 0.1, 0.5], [0.0, 0.0, 0.0, 0.6]):
        data = _generate_fake_qec_data(keys, error_rate, shots, rng)
        assert (
            qec_decoding.get_logical_error_probability(data, circuit, 0.001)
            == expected_lep
        )


@pytest.mark.skipif(
    not hasattr(cirq.transformers, "apply_lazy_args_on_circuit_operation"),
    reason="Requires pre-release version of Cirq",
)
def test_get_logical_error_probability_multi_observable():
    rng = np.random.default_rng(0)
    keys = [
        "0",
        "1",
        "2[0]",
        "2[1]",
        "2[2]",
        "2[3]",
        "3[0]",
        "3[1]",
        "3[2]",
        "3[3]",
        "10",
        "11",
        "12",
    ]
    shots = 10
    stim_circuit = stim.Circuit.generated(
        "repetition_code:memory", rounds=5, distance=3
    )
    stim_circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-2)], 1)
    circuit = stimcirq.stim_circuit_to_cirq_circuit(stim_circuit)

    data_noiseless = _generate_fake_qec_data(keys, 0.0, shots, rng)
    assert (
        qec_decoding.get_logical_error_probability(data_noiseless, circuit, 0.001)
        == 0.0
    )
    np.testing.assert_array_equal(
        qec_decoding.get_logical_error_probability(
            data_noiseless, circuit, 0.001, per_observable=True
        ),
        np.array([0.0, 0.0]),
    )

    data_noisy = _generate_fake_qec_data(keys, 0.5, shots, rng)
    joint_lep = qec_decoding.get_logical_error_probability(data_noisy, circuit, 0.001)
    per_obs_lep = qec_decoding.get_logical_error_probability(
        data_noisy, circuit, 0.001, per_observable=True
    )
    assert isinstance(joint_lep, float)
    assert per_obs_lep.shape == (2,)
    assert np.max(per_obs_lep) <= joint_lep <= np.sum(per_obs_lep)
