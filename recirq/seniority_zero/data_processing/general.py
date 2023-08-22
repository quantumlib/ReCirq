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

"""General data processing functions"""
from typing import Dict, List, Optional, Union, Tuple

import cirq
import collections
import numpy as np
import pandas
import scipy
from openfermion import QubitOperator

from recirq.seniority_zero.scheduling import SnakeMeasurementGroup


def multi_measurement_histogram(
    result: Union[cirq.Result, pandas.DataFrame],
    keys: List[str]
) -> collections.Counter:
    """Converts either a cirq.Result or pandas.Dataframe to a histogram

    Note: assumes each key is assigned to a measurement of a single qubit
    """
    if isinstance(result, cirq.Result):
        return result.multi_measurement_histogram(keys=keys)

    fixed_keys = [cirq.study.result._key_to_str(key) for key in keys]
    c: collections.Counter = collections.Counter()
    if len(fixed_keys) == 0:
        c[None] += result.repetitions
        return c
    bit_counts = [1 for key in fixed_keys]
    data_grouped = result.groupby(fixed_keys, as_index=False).size()
    for row_id in data_grouped.index:
        row = data_grouped.loc[row_id]
        sample = tuple(
            np.array(cirq.value.big_endian_int_to_bits(row[key], bit_count=bit_count), dtype=bool)
            for key, bit_count in zip(fixed_keys, bit_counts)
        )
        c[cirq.study.result._tuple_of_big_endian_int(sample)] += row['size']
    return c


def vectorize_expvals(
    num_qubits: int,
    expvals: Dict,
    covars: Optional[Dict] = None,
    distribute_with_covars: Optional[bool] = False,
    zz_flag: Optional[bool] = False,
) -> Tuple[np.ndarray, np.ndarray]:
    '''Convert a set of expectation values to a vector in a fixed basis.

    Args:
        num_qubits [int]: number of qubits in operator
        expvals [dict]: indexed by QubitOperators, contains the corresponding
            expectation value estimates Trace[rho * O]
        covars [dict[dict]]: covariance matrix in dictionary form
        distribute_with_covars [bool]: Whether to distribute term coefficients
            to minimise the covariance of the resulting sum based on the given
            covariances or just on the term sizes. This former is in
            principle optimal, but in practice appears unstable.
        zz_flag [bool]: whether or not this Hamiltonian contains ZZ terms
    '''
    fixed_operator_order = list(expvals.keys())
    trace_vector = np.array([expvals[op] for op in fixed_operator_order])

    num_operators = len(fixed_operator_order)
    if zz_flag:
        num_basis_vectors = num_qubits**2
    else:
        num_basis_vectors = (num_qubits + 1) * num_qubits // 2

    overlap_matrix = np.zeros([num_operators, num_basis_vectors])
    for op_index, op in enumerate(fixed_operator_order):
        coeff_list = indexing_function(num_qubits, QubitOperator(op))
        for index, coeff in coeff_list:
            overlap_matrix[op_index, index] += coeff

    if covars is None:
        inv_matrix = scipy.linalg.pinv(overlap_matrix.T @ overlap_matrix)
        expval_vec = trace_vector.T @ overlap_matrix @ inv_matrix
        new_covariance_matrix = None
    else:
        covar_matrix = np.array(
            [
                [covars[op][op2] if op2 in covars[op] else 0 for op2 in fixed_operator_order]
                for op in fixed_operator_order
            ]
        )
        covar_matrix = covar_matrix + np.identity(len(covar_matrix)) * 1e-10
        if distribute_with_covars:
            covar_matrix_inv = np.linalg.inv(covar_matrix)
            inv_matrix = scipy.linalg.pinv(overlap_matrix.T @ covar_matrix_inv @ overlap_matrix)
            conversion_matrix = covar_matrix_inv @ overlap_matrix @ inv_matrix
        else:
            inv_matrix = scipy.linalg.pinv(overlap_matrix.T @ overlap_matrix)
            conversion_matrix = overlap_matrix @ inv_matrix
        expval_vec = trace_vector.T @ conversion_matrix

        new_covariance_matrix = conversion_matrix.T @ covar_matrix @ conversion_matrix
    return expval_vec, new_covariance_matrix


def indexing_function(num_qubits: int, operator: QubitOperator) -> List:
    """Turn a single operator into sparse vector form.

    We define our vector basis in order:
    all Z_i, then all X_iX_j+Y_iY_j, then all Z_iZ_j
    (if needed), and assume that all operators coming in
    are linear combinations of these terms. This implies that
    in terms of where the basis operators are located

    Note: we assume that the coefficients of XX and YY are the same!
    If this is wrong, it will not be picked up here.

    Z_i -> i
    X_i X_j (i>j) -> num_qubits
                     + i * (i - 1) / 2 + j
    Z_i Z_j (i>j) -> num_qubits
                     + num_qubits * (num_qubits - 1) / 2
                     + i * (i - 1) / 2 + j

    Args:
        num_qubits: number of qubits in system
        operator: operator to transform

    Returns:
        list of [index, coefficient]
    """
    coefficients = []
    for term, coeff in operator.terms.items():
        if len(term) == 0:
            continue
        elif len(term) == 1:
            index = term[0][0]
        else:
            id1 = max(term[1][0], term[0][0])
            id2 = min(term[1][0], term[0][0])
            if term[0][1] == 'X' or term[0][1] == 'Y':
                index = num_qubits + id1 * (id1 - 1) // 2 + id2
            else:
                index = num_qubits * (num_qubits + 1) // 2 + id1 * (id1 - 1) // 2 + id2
        coefficients.append([index, coeff])
    return coefficients


def energy_from_expvals(
    num_qubits: int,
    expvals: Dict,
    hamiltonian: QubitOperator,
    covars: Optional[Dict] = None,
    distribute_with_covars: Optional[bool] = False,
    zz_flag: Optional[bool] = False,
) -> Tuple[float, float]:
    '''Take a set of measured expectation values and estimate an energy

    Args:
        num_qubits [int]: number of qubits in operator
        expvals [dict]: indexed by QubitOperators, contains the corresponding
            expectation value estimates Trace[rho * O]
        hamiltonian [QubitOperator]: The operator to calculate Trace[rho * H].
            We assume that hamiltonian is of the form
            sum_i h_i Z_i + sum_{i,j} h_{i,j} Z_i Z_j
            + sum_{i,j} g_{i,j}(X_i X_j + Y_i Y_j)
        covars [dict[dict]]: covariance matrix in dictionary form
        zz_flag [bool]: whether or not this Hamiltonian contains ZZ terms
    '''
    if zz_flag:
        num_basis_vectors = num_qubits**2
    else:
        num_basis_vectors = (num_qubits + 1) * num_qubits // 2

    expval_vec, new_covars = vectorize_expvals(
        num_qubits=num_qubits,
        expvals=expvals,
        covars=covars,
        zz_flag=zz_flag,
        distribute_with_covars=distribute_with_covars,
    )

    ham_vec = np.zeros([num_basis_vectors])
    coeff_list = indexing_function(num_qubits, hamiltonian)
    for index, coeff in coeff_list:
        ham_vec[index] += coeff

    energy = np.dot(ham_vec, expval_vec)
    if new_covars is not None:
        variance = ham_vec.T @ new_covars @ ham_vec
    else:
        variance = None
    if () in hamiltonian.terms:
        energy += hamiltonian.terms[()]
    return energy, variance


def order_parameter_from_expvals(num_qubits: int, expvals: Dict) -> float:
    """Calculate the superconducting order parameter from expectation values

    Args:
        num_qubits: number of qubits in system
        expvals: dictionary containing expectation values
    """
    order_parameter = 0
    for qid in range(num_qubits):
        key = str(QubitOperator(f'Z{qid}'))
        expz = expvals[key]
        nval = 0.5 - 0.5 * expz
        # If nval > 1, or nval < 0, we set nval = 0, in which case the contribution
        # to the order parameter is zero
        if nval < 1 and nval > 0:
            order_parameter += np.sqrt(nval - nval**2) * 2 / num_qubits
    return order_parameter


def order_parameter_var_from_expvals(num_qubits: int, expvals: Dict, covars: Dict) -> float:
    """Calculate the superconducting order parameter variance

    Args:
        num_qubits: number of qubits in system
        expvals: dictionary containing expectation values
        covars: dictionary containing covariances
    """

    variance = 0
    for qid in range(num_qubits):
        key = str(QubitOperator(f'Z{qid}'))
        expz = expvals[key]
        varz = covars[key][key]
        nvar = 0.25 * varz
        nval = 0.5 - 0.5 * expz
        if nval < 1 and nval > 0:
            variance += (
                nvar
                * (0.5 * (1 - 2 * nval) / np.sqrt(nval - nval**2)) ** 2
                * (2 / num_qubits) ** 2
            )
    return variance


def get_ls_echo_fidelity(
    frame: pandas.DataFrame,
    qubits: List,
    group: SnakeMeasurementGroup,
    postselect: Optional[bool] = False,
) -> Tuple[float, float]:
    """Extract an estimate of lochschmidt echo fidelity.

    Args:
        frame [dataframe]: frame containing experiment result
        qubits [List[cirq.GridQubit]]: Qubits in experiment
        group [SnakeMeasurementGroup]: defines initial state
        postselect [bool]: whether or not to give the fidelity with
            postselection on half-filling.
    """
    if type(frame) == cirq.Result:
        return get_ls_echo_fidelity(frame.data, qubits, group, postselect)

    # Get the number of successful experiments
    num_qubits = len(qubits)
    all_qubit_names = [str(qubit) for qubit in qubits]
    frame['success'] = (frame[all_qubit_names].sum(axis=1) == 0).astype(int)
    total_success = frame['success'].sum(axis=0)
    if postselect is False:
        prob = total_success / frame.shape[0]
        pvar = prob * (1 - prob) / frame.shape[0]
        fidel = np.sqrt(prob)
        fvar = pvar / prob
        return fidel, fvar

    # Postselected fidelity requires that we estimate the
    # number of experiments that pass postselection, which requires
    # 'undoing' the final set of de-excitations.
    if group is None:
        shift = 0
    else:
        shift = group.shift
    initial_state = [(qid + shift) % 2 for qid in range(num_qubits)]
    all_exq_names = [qid + 'i' for qid in all_qubit_names]
    for qid, exqid, ex in zip(all_qubit_names, all_exq_names, initial_state):
        frame[exqid] = frame[qid] ^ ex
    frame['postselected'] = (frame[all_exq_names].sum(axis=1) == len(all_qubit_names) // 2).astype(
        int
    )
    total_postselected = frame['postselected'].sum(axis=0)
    assert total_postselected >= total_success
    prob = total_success / total_postselected
    pvar = prob * (1 - prob) / total_postselected
    fidel = np.sqrt(prob)
    fvar = pvar / prob

    return fidel, fvar


def get_signed_count(
    frame: pandas.DataFrame, msmt_qubits: List[cirq.GridQubit], all_qubits: List[cirq.GridQubit]
) -> Tuple[int, int]:
    """Calculate a signed count from a pandas dataframe of a cirq result

    The signed count is the number of even parity results minus
    the number of odd parity results.

    (I.e. for a single qubit, the signed count is the number of 0's minus
    the number of 1's.)

    result [pandas.DataFrame] : the dataframe to use
    msmt_qubits [list[cirq.GridQubit]] : the qubits to calculate the
        signed count across
    all_qubits [list[cirq.GridQubit]] : list of all qubits
    """
    if isinstance(frame, cirq.Result):
        return get_signed_count(frame.data, msmt_qubits, all_qubits)
    qubit_names = [str(qubit) for qubit in msmt_qubits]
    frame['signed_count'] = 1 - 2 * (frame[qubit_names].sum(axis=1) % 2)
    signed_count = frame['signed_count'].sum(axis=0)
    total_count = frame.shape[0]
    return signed_count, total_count


def get_signed_count_verified(
    frame: pandas.DataFrame,
    msmt_qubits: List[cirq.GridQubit],
    all_qubits: List[cirq.GridQubit],
    correct_number: int,
) -> Tuple[int, int, int]:
    """Calculate a signed count from a pandas dataframe with verification

    The signed count is the number of even parity results minus
    the number of odd parity results.

    (I.e. for a single qubit, the signed count is the number of 0's minus
    the number of 1's.)

    result [pandas.DataFrame] : the dataframe to use
    msmt_qubits [list[cirq.GridQubit]] : the qubits to calculate the
        signed count across
    all_qubits [list[cirq.GridQubit]] : list of all qubits
    correct_number: How many of all_qubits (excluding msmt_qubits)
        should be equal to 1.
    """
    if type(frame) == cirq.Result:
        return get_signed_count_verified(frame.data, msmt_qubits, all_qubits, correct_number)
    other_qubit_names = [str(qubit) for qubit in all_qubits if qubit not in msmt_qubits]
    qubit_names = [str(qubit) for qubit in msmt_qubits]
    frame['verified'] = (frame[other_qubit_names].sum(axis=1) == correct_number).astype(int)
    frame['signed_count'] = 1 - 2 * (frame[qubit_names].sum(axis=1) % 2)
    frame['verified_signed_count'] = frame['verified'] * frame['signed_count']
    signed_count = frame['signed_count'].sum(axis=0)
    signed_count_vf = frame['verified_signed_count'].sum(axis=0)
    total_count = frame.shape[0]
    return signed_count, signed_count_vf, total_count


def get_signed_count_postselected(
    frame: pandas.DataFrame,
    msmt_qubits: List[cirq.GridQubit],
    all_qubits: List[cirq.GridQubit],
    correct_number: int,
) -> Tuple[int, int]:
    """Calculate a signed count from a pandas dataframe with postselection

    The signed count is the number of even parity results minus
    the number of odd parity results.

    (I.e. for a single qubit, the signed count is the number of 0's minus
    the number of 1's.)

    result [pandas.DataFrame] : the dataframe to use
    msmt_qubits [list[cirq.GridQubit]] : the qubits to calculate the
        signed count across
    all_qubits [list[cirq.GridQubit]] : list of all qubits
    correct_number: How many of all_qubits should be equal to 1.
    """
    if type(frame) == cirq.Result:
        return get_signed_count_postselected(frame.data, msmt_qubits, all_qubits, correct_number)
    msmt_qubit_names = [str(qubit) for qubit in msmt_qubits]
    all_qubit_names = [str(qubit) for qubit in all_qubits]
    frame['postselected'] = frame[all_qubit_names].sum(axis=1) == correct_number
    frame['signed_count'] = 1 - 2 * (frame[msmt_qubit_names].sum(axis=1) % 2)
    frame['postselected_signed_count'] = frame['postselected'] * frame['signed_count']
    ps_signed_count = frame['postselected_signed_count'].sum(axis=0)
    ps_total_count = frame['postselected'].sum(axis=0)
    return ps_signed_count, ps_total_count


def get_p0m_count_verified(
    frame: pandas.DataFrame,
    msmt_qubits: List[cirq.GridQubit],
    all_qubits: List[cirq.GridQubit],
    correct_number: int,
) -> Tuple[int, int, int]:
    """Calculate counts of +, -, and 0 for a verification experiment

    result [pandas.DataFrame] : the dataframe to use
    msmt_qubits [list[cirq.GridQubit]] : the qubits to calculate the
        signed count across
    all_qubits [list[cirq.GridQubit]] : list of all qubits
    correct_number: How many of all_qubits (excluding msmt_qubits)
        should be equal to 1.
    """
    if type(frame) == cirq.Result:
        return get_p0m_count_verified(frame.data, msmt_qubits, all_qubits, correct_number)
    other_qubit_names = [str(qubit) for qubit in all_qubits if qubit not in msmt_qubits]
    qubit_names = [str(qubit) for qubit in msmt_qubits]
    frame['verified'] = (frame[other_qubit_names].sum(axis=1) == correct_number).astype(int)
    num_zeros = frame.shape[0] - frame['verified'].sum()
    frame['parity_m'] = frame[qubit_names].sum(axis=1) % 2
    frame['parity_p'] = 1 - frame['parity_m']
    frame['verified_parity_m'] = frame['parity_m'] * frame['verified']
    frame['verified_parity_p'] = frame['parity_p'] * frame['verified']
    num_ps = frame['verified_parity_p'].sum()
    num_ms = frame['verified_parity_m'].sum()
    assert num_ps + num_ms + num_zeros == frame.shape[0]
    return num_ps, num_zeros, num_ms
