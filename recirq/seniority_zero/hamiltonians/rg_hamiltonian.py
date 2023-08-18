# Copyright 2023 Google
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

from itertools import product
from typing import List, Tuple

import cirq
import numpy as np
import openfermion as of


class RGDOCI:
    """Object for initializing the DOCI Hamiltonian for an RG Hamiltonian

    0) user specifies num-qubits and g value
    1) get fermion RG Hamiltonian
    2) get qubit DOCI operator
    3) get cirq DOCI operators as PauliSum
    4) get the occupied virtual basis bersion
    """

    def __init__(self, g_value: float, norbs: int):
        r"""Initialize object for holding all relevant Hamiltonian data.

        H = sum_{i=1}^{n}(i)(n_{i,alpha} + n_{i,beta} + g\sum_{i =\= j}P_{i}^ P_{j}

        P_{i} = a_{ibeta}a_{ialpha}

        :param g_value: Interaction parameter for RG model
        :param norbs: Number of spatial orbitals
        """
        self.g_value = g_value
        self.norbs = norbs
        self.n_qubits = norbs

    def get_spinorb_ints(self):
        r"""H = sum_{i=1}^{n}(i)(n_{i,alpha} + n_{i,beta} + g/2\sum_{i =\= j}P_{i}^ P_{j}

        but symmeterized for ab and ba space
        :return:
        """
        spatial_orbs = self.norbs
        h1 = np.diag(np.arange(spatial_orbs) + 1)
        h1 = np.kron(h1, np.eye(2))
        h2 = np.zeros((2 * spatial_orbs,) * 4)
        for p, q in product(range(spatial_orbs), repeat=2):
            if p != q:
                h2[2 * p, 2 * p + 1, 2 * q + 1, 2 * q] = self.g_value / 2
                h2[2 * p + 1, 2 * p, 2 * q, 2 * q + 1] = self.g_value / 2
        return h1, h2

    def get_spinorb_ints_ov_basis(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the integrals in ov order. Only the 1-body part changes"""
        norbs = self.norbs
        _, h2 = self.get_spinorb_ints()
        perm_order = [(xx, (norbs // 2) + xx) for xx in range(norbs // 2)]
        perm_order = [k for pair in perm_order for k in pair]
        h1 = np.diag(np.arange(norbs) + 1)
        h1 = h1[:, perm_order]
        h1 = h1[perm_order, :]
        h1 = np.kron(h1, np.eye(2))
        return h1, h2

    def get_cc_spinorbs(self):
        """Spin-orbitals for CC solver. 2-electron part must be antisymmeterized

        In this code we account for the antisymmeterization with a factor of
        0.5 for double counting on the alpha-beta space.
        """
        h1, h2 = self.get_spinorb_ints()
        antih2 = 0.5 * (h2 - np.einsum('ijlk', h2))
        return h1, antih2

    def get_fermion_ham(self) -> of.FermionOperator:
        """Get the RG Hamiltonain as an OpenFermion Fermion Operator"""
        spatial_orbs = self.norbs
        h1, h2 = self.get_spinorb_ints()
        fham = of.FermionOperator()
        for pp in range(spatial_orbs):
            fham += of.FermionOperator(
                ((2 * pp, 1), (2 * pp, 0)), coefficient=float(h1[2 * pp, 2 * pp])
            )
            fham += of.FermionOperator(
                ((2 * pp + 1, 1), (2 * pp + 1, 0)), coefficient=float(h1[2 * pp + 1, 2 * pp + 1])
            )
        for p, q in product(range(spatial_orbs), repeat=2):
            if p != q:
                ab = ((2 * p, 1), (2 * p + 1, 1), (2 * q + 1, 0), (2 * q, 0))
                ba = ((2 * p + 1, 1), (2 * p, 1), (2 * q, 0), (2 * q + 1, 0))
                fham += of.FermionOperator(ab, coefficient=h2[2 * p, 2 * p + 1, 2 * q + 1, 2 * q])
                fham += of.FermionOperator(ba, coefficient=h2[2 * p + 1, 2 * p, 2 * q, 2 * q + 1])
        return fham

    def get_fermion_ham_ov_basis(self) -> of.FermionOperator:
        spatial_orbs = self.norbs
        h1, h2 = self.get_spinorb_ints_ov_basis()
        fham = of.FermionOperator()
        for pp in range(spatial_orbs):
            fham += of.FermionOperator(
                ((2 * pp, 1), (2 * pp, 0)), coefficient=float(h1[2 * pp, 2 * pp])
            )
            fham += of.FermionOperator(
                ((2 * pp + 1, 1), (2 * pp + 1, 0)), coefficient=float(h1[2 * pp + 1, 2 * pp + 1])
            )
        for p, q in product(range(spatial_orbs), repeat=2):
            if p != q:
                ab = ((2 * p, 1), (2 * p + 1, 1), (2 * q + 1, 0), (2 * q, 0))
                ba = ((2 * p + 1, 1), (2 * p, 1), (2 * q, 0), (2 * q + 1, 0))
                fham += of.FermionOperator(ab, coefficient=h2[2 * p, 2 * p + 1, 2 * q + 1, 2 * q])
                fham += of.FermionOperator(ba, coefficient=h2[2 * p + 1, 2 * p, 2 * q, 2 * q + 1])
        return fham

    def get_qubit_hamiltonian(self) -> Tuple[of.QubitOperator, float]:
        r"""sum_{p}e_{p}~N_{p} + g sum_{pq} P_{p}^P_{q}"""
        spatial_orbs = self.norbs
        qham = of.QubitOperator()
        h1 = np.diag(np.arange(spatial_orbs) + 1)
        # constant term
        constant = sum(np.diagonal(h1))
        for pp in range(spatial_orbs):
            qham += of.QubitOperator((pp, 'Z'), coefficient=float(-h1[pp, pp]))
            for qq in range(spatial_orbs):
                if pp != qq:
                    qham += of.QubitOperator(((pp, 'X'), (qq, 'X')), coefficient=self.g_value / 4)
                    qham += of.QubitOperator(((pp, 'Y'), (qq, 'Y')), coefficient=self.g_value / 4)
        return qham, float(constant)

    def get_qubit_hamiltonian_ov_basis(self, reverse_order=False) -> Tuple[of.QubitOperator, float]:
        r"""sum_{p}e_{p}~N_{p} + g sum_{pq} P_{p}^P_{q}"""
        spatial_orbs = self.norbs
        qham = of.QubitOperator()
        if reverse_order:
            h1 = np.diag((np.arange(spatial_orbs) + 1)[::-1])
        else:
            h1 = np.diag(np.arange(spatial_orbs) + 1)

        perm_order = [(xx, (self.norbs // 2) + xx) for xx in range(self.norbs // 2)]
        perm_order = [k for pair in perm_order for k in pair]
        h1 = h1[:, perm_order]
        h1 = h1[perm_order, :]
        # constant term
        constant = sum(np.diagonal(h1))
        for pp in range(spatial_orbs):
            qham += of.QubitOperator((pp, 'Z'), coefficient=float(-h1[pp, pp]))
            for qq in range(spatial_orbs):
                if pp != qq:
                    qham += of.QubitOperator(((pp, 'X'), (qq, 'X')), coefficient=self.g_value / 4)
                    qham += of.QubitOperator(((pp, 'Y'), (qq, 'Y')), coefficient=self.g_value / 4)
        return qham, float(constant)

    def get_cirq_operator(self, qubits: List[cirq.Qid], ov_basis=False) -> cirq.PauliSum:
        """Construct the Hamiltonian as a PauliSum object

        This works by calling the self.get_doci_qubit_operator which can be
        in the ov_basis or [occ, virt] basis.  Then this OpenFermion object is
        converted into a PauliSum object.

        The returned object is useful because we can directly use it to compute
        expectation values from cirq simulator objects.

        :param qubits: list of qubits where the order corresponds to basis order
                       used.  if ov_basis flag then [0, N/2, 1, N/2 + 1, ...]
                       else [0, 1, ..., N].
        :param bool ov_basis: flag for which basis ordering to use [ov, occ-virt]
        :return: cirq.PauliSum representing the Hamiltonian
        """
        if len(qubits) != self.norbs:
            raise ValueError("Number of qubits is not consistent with the problem size")
        if ov_basis:
            qubit_operator, constant = self.get_qubit_hamiltonian_ov_basis()
        else:
            qubit_operator, constant = self.get_qubit_hamiltonian()

        self.constant = constant
        qubit_map = dict(zip(range(self.norbs), qubits))
        cirq_pauli_terms = []
        for term, val in qubit_operator.terms.items():
            pauli_term_dict = dict([(qubit_map[xx], yy) for xx, yy in term])
            pauli_term = cirq.PauliString(pauli_term_dict, coefficient=val)
            cirq_pauli_terms.append(pauli_term)
        return cirq.PauliSum().from_pauli_strings(cirq_pauli_terms)
