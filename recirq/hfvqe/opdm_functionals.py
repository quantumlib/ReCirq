from typing import List, Optional
import cirq
import numpy as np
import scipy as sp
import openfermioncirq.experiments.hfvqe.circuits as ccc
import openfermioncirq.experiments.hfvqe.analysis as cca
import openfermioncirq.experiments.hfvqe.util as ccu
from openfermioncirq.experiments.hfvqe.objective import rhf_params_to_matrix


class RDMGenerator():  # testpragma: no cover
    def __init__(self, blackbox, purification: Optional[bool]=True):
        self.blackbox = blackbox
        self.noisy_opdms = []
        self.variance_dicts = []
        self.purification = purification

    def opdm_generator(self, parameters):
        opdm, var_dict = self.blackbox.calculate_rdm(parameters)
        self.noisy_opdms.append(opdm)
        self.variance_dicts.append(var_dict)
        if self.purification:
            opdm = cca.mcweeny_purification(opdm)
        return opdm


class OpdmFunctional():  # testpragma: no cover
    def __init__(self, qubits: List[cirq.Qid],
                 sampler: cirq.Sampler,
                 constant: float,
                 one_body_integrals: np.ndarray,
                 two_body_integrals: np.ndarray,
                 num_electrons: int,
                 num_samples: Optional[int] = 250_000,
                 post_selection: Optional[bool] = True,
                 purification: Optional[bool] = True,
                 clean_xxyy: Optional[bool] = True,
                 verbose: Optional[bool] = False):
        self.qubits = qubits
        self.constant = constant
        self.one_body_integrals = one_body_integrals
        self.two_body_integrals = two_body_integrals
        self.num_electrons = num_electrons
        self.sampler = sampler
        self.num_samples = num_samples
        self.post_selection = post_selection
        self.purification = purification
        self.clean_xxyy = clean_xxyy
        self.num_qubits = len(self.qubits)
        self.verbose = verbose  # type: bool

        self._last_noisy_opdm = None

        norbs = len(self.qubits)
        self.occ = list(range(num_electrons))
        self.virt = list(range(num_electrons, norbs))
        self.nocc = len(self.occ)
        self.nvirt = len(self.virt)
        self.clean_xxyy = clean_xxyy

    def calculate_data(self, parameters):
        if len(parameters.shape) == 2:  # testpragma: no cover
            u = parameters
        else:
            u = sp.linalg.expm(rhf_params_to_matrix(parameters,
                                                    len(self.qubits),
                                                    occ=self.occ,
                                                    virt=self.virt
                                                    )
                               )

        circuits = ccc.generate_circuits_from_params_or_u(
            self.qubits, u, self.num_electrons,
            occ=self.occ,
            virt=self.virt,
            clean_ryxxy=self.clean_xxyy
        )
        circuits_dict = ccc.circuits_with_measurements(
            self.qubits, circuits, clean_xxyy=self.clean_xxyy
        )

        # Take data
        data_dict = {'z': {},
                     'xy_even': {},
                     'xy_odd': {},
                     'qubits': [f'({q.row}, {q.col})' for q in self.qubits],
                     'qubit_permutations': ccu.generate_permutations(
                         len(self.qubits)),
                     'circuits': circuits,
                     'circuits_with_measurement': circuits_dict}
        for measure_type in circuits_dict.keys():
            circuits = circuits_dict[measure_type]
            for circuit_index in circuits.keys():
                circuit = circuits[circuit_index]
                if self.verbose:  # testpragma: no cover
                    print(circuit.to_text_diagram(transpose=True))
                data = self.sampler.run(circuit, repetitions=self.num_samples)
                if self.post_selection:
                    # PostSelect the data
                    good_indices = \
                        np.where(np.sum(np.array(data.data), axis=1) ==
                                            self.num_electrons)[0]
                    good_data = data.data[data.data.index.isin(good_indices)]
                    data_dict[measure_type][circuit_index] = good_data
                else:  # testpragma: no cover
                    data_dict[measure_type][circuit_index] = data.data
        return data_dict

    def calculate_rdm(self, parameters):
        data_dict = self.calculate_data(parameters)
        opdm, opdm_var_dict = cca.compute_opdm(data_dict, return_variance=True)
        if self.purification:
            self._last_noisy_opdm = opdm
            opdm = cca.mcweeny_purification(opdm)

        return opdm, opdm_var_dict

    def energy_from_opdm(self, opdm):
        return cca.energy_from_opdm(opdm, self.constant,
                                    self.one_body_integrals,
                                    self.two_body_integrals)
