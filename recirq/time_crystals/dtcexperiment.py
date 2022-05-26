# Copyright 2022 Google
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
import itertools
import numpy as np
from typing import Sequence, Optional, Iterator
import os

EXPERIMENT_NAME = "time_crystals"
DEFAULT_BASE_DIR = os.path.expanduser(f"~/cirq_results/{EXPERIMENT_NAME}")


class DTCExperiment:
    """Manage inputs to a DTC experiment, over some number of disorder instances

    Attributes:
        qubits: a chain of connected qubits available for the circuit.
            Defaults to 16 `cirq.Gridqubits` connected in a chain.
        disorder_instances: number of disorder instances averaged over.
            Defaults to 36.
        g: control constant used in circuit.
            Defaults to 0.94.
        initial_states: initial state of the system used in circuit.
            Defaults to `np.ndarray` of shape (disorder_instances, num_qubits) of ints,
            randomly selected from {0,1}.
        local_fields: local fields that break integrability.
            Defaults to `np.ndarray` of shape (disorder_instances, num_qubits) of floats,
            randomly and uniformly selected from the range [-1.0, 1.0].
        thetas: theta parameters for FSim Gate used in circuit.
            Defaults to `np.ndarray` of shape (disorder_instnaces, num_qubits - 1) of ints,
            all set to zero.
        zetas: zeta parameters for FSim Gate used in circuit.
            Defaults to `np.ndarray` of shape (disorder_instnaces, num_qubits - 1) of ints,
            all set to zero.
        chis: chi parameters for FSim Gate used in circuit.
            Defaults to `np.ndarray` of shape (disorder_instnaces, num_qubits - 1) of ints,
            all set to zero.
        phis: phi parameters for FSim Gate used in circuit.
            Defaults to `np.ndarray` of shape (disorder_instances, num_qubits - 1) of floats,
            randomly and uniformly selected from the range [-0.5*`np.pi`, -1.5*`np.pi`].
        gammas: gamma parameters for FSim Gate used in circuit.
            Defaults to `np.ndarray` of shape (disorder_instances, num_qubits - 1) of floats,
            computed as -2*gammas.

    """

    def __init__(
        self,
        qubits: Optional[Sequence[cirq.Qid]] = None,
        disorder_instances: Optional[int] = 36,
        g: Optional[int] = 0.94,
        initial_states: Optional[np.ndarray] = None,
        local_fields: Optional[np.ndarray] = None,
        thetas: Optional[np.ndarray] = None,
        zetas: Optional[np.ndarray] = None,
        chis: Optional[np.ndarray] = None,
        gammas: Optional[np.ndarray] = None,
        phis: Optional[np.ndarray] = None,
    ):

        self.qubits = qubits
        self.disorder_instances = disorder_instances
        self.g = g

        if qubits is None:
            qubit_locations = [
                (3, 9),
                (3, 8),
                (3, 7),
                (4, 7),
                (4, 8),
                (5, 8),
                (5, 7),
                (5, 6),
                (6, 6),
                (6, 5),
                (7, 5),
                (8, 5),
                (8, 4),
                (8, 3),
                (7, 3),
                (6, 3),
            ]
            self.qubits = [cirq.GridQubit(*idx) for idx in qubit_locations]
        else:
            self.qubits = qubits

        num_qubits = len(self.qubits)

        if initial_states is None:
            self.initial_states = np.random.choice(
                2, (self.disorder_instances, num_qubits)
            )
        else:
            self.initial_states = initial_states

        if local_fields is None:
            self.local_fields = np.random.uniform(
                -1.0, 1.0, (self.disorder_instances, num_qubits)
            )
        else:
            self.local_fields = local_fields

        zero_params = [thetas, zetas, chis]
        for index, zero_param in enumerate(zero_params):
            if zero_param is None:
                zero_params[index] = np.zeros((self.disorder_instances, num_qubits - 1))
            else:
                zero_params[index] = zero_param
        self.thetas, self.zetas, self.chis = zero_params

        # if gamma or phi is not supplied, generate it from the other such that phis == -2*gammas
        if gammas is None and phis is None:
            self.gammas = -np.random.uniform(
                0.5 * np.pi, 1.5 * np.pi, (self.disorder_instances, num_qubits - 1)
            )
            self.phis = -2 * self.gammas
        elif phis is None:
            self.gammas = gammas
            self.phis = -2 * self.gammas
        elif gammas is None:
            self.phis = phis
            self.gammas = -1 / 2 * self.phis
        else:
            self.phis = phis
            self.gammas = gammas

    def param_resolvers(self) -> cirq.Zip:
        """return a sweep over disorder instances for the parameters of this experiment

        Returns:
            `cirq.Zip` object with self.disorder_instances many `cirq.ParamResolver`s

        """

        # initialize the dict and add the first, non-qubit-dependent parameter, g
        factor_dict = {"g": np.full(self.disorder_instances, self.g).tolist()}

        # iterate over the different parameters
        qubit_varying_factors = [
            "initial_states",
            "local_fields",
            "thetas",
            "zetas",
            "chis",
            "gammas",
            "phis",
        ]
        for factor in qubit_varying_factors:
            parameter = getattr(self, factor)
            # iterate over each index in the qubit chain and the various options for that qubit
            for index, qubit_factor_options in enumerate(parameter.transpose()):
                factor_name = factor[:-1]
                factor_dict[f"{factor_name}_{index}"] = qubit_factor_options.tolist()

        return cirq.study.dict_to_zip_sweep(factor_dict)


def comparison_experiments(
    qubits: Sequence[cirq.Qid],
    disorder_instances: int,
    g_cases: Optional[Sequence[int]] = None,
    initial_states_cases: Optional[Sequence[np.ndarray]] = None,
    local_fields_cases: Optional[Sequence[np.ndarray]] = None,
    phis_cases: Optional[Sequence[np.ndarray]] = None,
) -> Iterator[DTCExperiment]:
    """Yield DTCExperiments with parameters taken from the cartesian product of input parameters

    Args:
        Any number of (parameter, parameter_values) pairs

    Yields:
        DTCTasks with parameters taken from self.options_dict

    """

    # take product over elements of options_dict, in the order of options_order
    argument_cases = [
        ([x] if x is None else x)
        for x in [g_cases, initial_states_cases, local_fields_cases, phis_cases]
    ]
    argument_names = ["g", "initial_states", "local_fields", "phis"]
    for arguments in itertools.product(*argument_cases):
        # prepare arguments for DTCExperiment
        named_args = zip(argument_names, arguments)
        kwargs = {name: arg for (name, arg) in named_args if arg is not None}
        yield DTCExperiment(
            qubits=qubits, disorder_instances=disorder_instances, **kwargs
        )
