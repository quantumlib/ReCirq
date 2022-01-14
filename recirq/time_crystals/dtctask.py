# Copyright 2021 Google
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
import recirq
import datetime
import itertools
import numpy as np
from typing import Sequence, Optional, Dict
import os

EXPERIMENT_NAME = "time_crystals"
DEFAULT_BASE_DIR = os.path.expanduser(f'~/cirq_results/{EXPERIMENT_NAME}')


@recirq.json_serializable_dataclass(namespace='recirq.time_crystals',
                                    registry=recirq.Registry,
                                    frozen=False)
class CompareDTCTask:
    """ A task to a comparison DTC experiment, comparing different options for parameters

    Attributes
        dataset_id: unique identifier for this dataset
        qubits: chain of connected qubits available for the circuit
        cycles: number of DTC cycles to consider for circuits
        circuit_list: symbolic DTC circuit list
        disorder_instances: number of disorder instances averaged over
        options_dict: dict mapping DTCTask attribute names to options for that attribute, to take a product over
        options_order: sequence of keys in options_dict, defining order of product over options
    """

    # Task parameters
    dataset_id: str

    # experiment parameters
    qubits: Sequence[cirq.Qid]
    cycles: int
    disorder_instances: int
    circuit_list: Sequence[cirq.Circuit]

    # options to take product over
    options_dict: Dict[str, Sequence[np.ndarray]]
    options_order: Sequence[str]

    def __init__(
            self,
            qubits: Sequence[cirq.Qid],
            cycles: int,
            disorder_instances: int,
            options_dict: Dict[str, Sequence[np.ndarray]],
            options_order: Optional[Sequence[str]] = None):

        self.dataset_id = datetime.datetime.utcnow()

        self.qubits = qubits
        self.cycles = cycles
        self.disorder_instances = disorder_instances

        # create symbolic circuit list from qubits and cycles count
        self.circuit_list = recirq.time_crystals.symbolic_dtc_circuit_list(qubits, cycles)

        self.options_dict = options_dict
        self.options_order = list(self.options_dict.keys()) if options_order is None else options_order

        # check that the input parameters are consistent
        assert set(self.options_order) == set(self.options_dict.keys()), 'options_order and the keys of options_dict are not the same'
        assert not {'initial_states', 'initial_state'} <= self.options_dict.keys(), 'do not supply both initial_states and initial_state'


    @property
    def fn(self):
        fn = (f'{self.dataset_id}/'
                f'{len(self.qubits)}/'
                f'{self.cycles}/'
                f'{self.disorder_instances}/'
                f'{self.options_dict}')
        return fn


    def dtctasks(self):
        """ Yield a sequence of DTCTasks that are the product of the options in self.options_dict.
                All DTCTask attributes not in options_dict are taken to be their default values
        Yields:
            DTCTasks with parameters taken from self.options_dict
        """

        # take product over elements of options_dict, in the order of options_order
        for components in itertools.product(*(self.options_dict[attribute_name] for attribute_name in self.options_order)):
            # prepare arguments for DTCTask
            kwargs = dict(zip(self.options_order, components))
            yield DTCTask(qubits=self.qubits, disorder_instances=self.disorder_instances, **kwargs)


@recirq.json_serializable_dataclass(namespace='recirq.time_crystals',
                                    registry=recirq.Registry,
                                    frozen=False)
class DTCTask:
    """ A task for managing inputs to a Discrete Time Crystal experiment, over some number of disorder instances

    Attributes:
        dataset_id: unique identifier for this dataset
        qubits: a chain of connected qubits available for the circuit
        disorder_instances: number of disorder instances averaged over
        initial_states: initial state of the system used in circuit
        g: thermalization constant used in circuit
        local_fields: random noise used in circuit
        thetas: theta parameters for FSim Gate used in circuit
        zetas: zeta parameters for FSim Gate used in circuit
        chis: chi parameters for FSim Gate used in circuit
        phis: phi parameters for FSim Gate used in circuit
        gammas: gamma parameters for FSim Gate used in circuit

    """
    # Task parameters
    dataset_id: str

    # experiment parameters
    qubits: Sequence[cirq.Qid]
    disorder_instances: int

    # FSim Gate parameters
    # ndarrays in this section are in shape (disorder_instances, len(qubits) - 1)
    g: int
    initial_states: np.ndarray
    local_fields: np.ndarray

    # FSim Gate Parameters
    # ndarrays in this section are in shape (disorder_instances, len(qubits) - 1)
    thetas: np.ndarray
    zetas: np.ndarray
    chis: np.ndarray
    gammas: np.ndarray
    phis: np.ndarray


    def __init__(
            self,
            qubits: Optional[Sequence[cirq.Qid]] = None,
            disorder_instances: Optional[int] = None,
            g: Optional[int] = None,
            initial_state: Optional[np.ndarray] = None,
            initial_states: Optional[np.ndarray] = None,
            local_fields: Optional[np.ndarray] = None,
            thetas: Optional[np.ndarray] = None,
            zetas: Optional[np.ndarray] = None,
            chis: Optional[np.ndarray] = None,
            gammas: Optional[np.ndarray] = None,
            phis: Optional[np.ndarray] = None
            ):

        self.dataset_id = datetime.datetime.utcnow()

        self.disorder_instances = 36 if disorder_instances is None else disorder_instances

        self.g = 0.94 if g is None else g

        if qubits is None:
            qubit_locations = [(3, 9), (3, 8), (3, 7), (4, 7), (4, 8), (5, 8), (5, 7), (5, 6), (6, 6), (6, 5), (7, 5), (8, 5), (8, 4), (8, 3), (7, 3), (6, 3)]
            self.qubits = [cirq.GridQubit(*idx) for idx in qubit_locations]
        else:
            self.qubits = qubits

        num_qubits = len(self.qubits)

        # only enable use of initial_state or initial_states
        assert initial_state is None or initial_states is None, 'do not supply both initial_state and initial_states'
        if initial_state is None and initial_states is None:
            self.initial_states = np.random.choice(2, (self.disorder_instances, num_qubits))
        elif initial_states is None:
            assert len(initial_state) == num_qubits, f'initial_state is of shape {str(len(initial_state))}, not (num_qubits,)'
            self.initial_states = np.tile(initial_state, (self.disorder_instances, 1))
        elif initial_state is None:
            assert initial_states.shape == (self.disorder_instances, num_qubits), f'initial_states is of shape {initial_states.shape}, not (disorder_instances, num_qubits)'
            self.initial_states = initial_states

        if local_fields is None:
            self.local_fields = np.random.uniform(-1.0, 1.0, (self.disorder_instances, num_qubits))
        else:
            assert local_fields.shape == (self.disorder_instances, num_qubits), f'local_fields is of shape {local_fields.shape}, not (disorder_instnaces, num_qubits)'
            self.local_fields = local_fields

        zero_params = [thetas, zetas, chis]
        for index, zero_param in enumerate(zero_params):
            if zero_param is None:
                zero_params[index] = np.zeros((self.disorder_instances, num_qubits - 1))
            else:
                assert zero_param.shape == (self.disorder_instances, num_qubits - 1), f'thetas, zetas or chis is of shape {zero_param.shape}, not (disorder_instances, num_qubits - 1)'
        self.thetas, self.zetas, self.chis = zero_params

        # if gamma or phi is not supplied, generate it from the other such that phis == -2*gammas
        if gammas is None and phis is None:
            self.gammas = -np.random.uniform(0.5*np.pi, 1.5*np.pi, (self.disorder_instances, num_qubits - 1))
            self.phis = -2*self.gammas
        elif phis is None:
            assert gammas.shape == (self.disorder_instances, num_qubits - 1), f'gammas is of shape {gammas.shape}, not (disorder_instances, num_qubits - 1)'
            self.gammas = gammas
            self.phis = -2*self.gammas
        elif gammas is None:
            assert phis.shape == (self.disorder_instances, num_qubits - 1), f'phis is of shape {phis.shape}, not (disorder_instances, num_qubits - 1)'
            self.phis = phis
            self.gammas = -1/2*self.phis
        else:
            assert gammas.shape == (self.disorder_instances, num_qubits - 1), f'gammas is of shape {gammas.shape}, not (disorder_instances, num_qubits - 1)'
            assert phis.shape == (self.disorder_instances, num_qubits - 1), f'phis is of shape {phis.shape}, not (disorder_instances, num_qubits - 1)'
            self.phis = phis
            self.gammas = gammas


    @property
    def fn(self):
        fn = (f'{self.dataset_id}/'
                f'{self.qubits}/'
                f'{self.disorder_instances}/'
                f'{self.g}/'
                f'{self.initial_states}/'
                f'{self.local_fields}/'
                f'{self.thetas}/'
                f'{self.zetas}/'
                f'{self.chis}/'
                f'{self.gammas}/'
                f'{self.phis}/')
        return fn


    def param_resolvers(self):
        """ return a sweep over param resolvers for the parameters of this task
        Returns:
            `cirq.Zip` object with self.disorder_instances many `cirq.ParamResolver`s
        """

        # initialize the dict and add the first, non-qubit-dependent parameter, g
        factor_dict = {'g': np.full(self.disorder_instances, self.g).tolist()}

        # iterate over the different parameters
        qubit_varying_factors = ["initial_states", "local_fields", "thetas", "zetas", "chis", "gammas", "phis"]
        for factor in qubit_varying_factors:
            factor_options = getattr(self, factor)
            # iterate over each index in the qubit chain and the various options for that qubit
            for index, qubit_factor_options in enumerate(factor_options.transpose()):
                factor_name = factor[:-1]
                factor_dict[f'{factor_name}_{index}'] = qubit_factor_options.tolist()

        return cirq.study.dict_to_zip_sweep(factor_dict)
