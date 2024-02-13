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

"""Analysis routines for loschmidt echo data.

See also the notebooks in this directory demonstrating usage of these analysis routines.
"""
import datetime
from typing import Callable, Dict, cast, Sequence, Any, Tuple, List, TypeVar, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import cirq
import cirq_google as cg

try:
    from cirq_google.workflow import (
        ExecutableGroupResult, SharedRuntimeInfo,
        QuantumRuntimeConfiguration,
        ExecutableResult)
    from recirq.otoc.loschmidt.tilted_square_lattice import TiltedSquareLatticeLoschmidtSpec

    workflow = True
except ImportError as e:
    import os

    if 'RECIRQ_IMPORT_FAILSAFE' in os.environ:
        workflow = False
    else:
        raise ImportError(f"This functionality requires a Cirq >= 0.14: {e}")

CYCLES_PER_MACROCYCLE = 4
"""Each macrocycle has 4 'cycles' for the four directions in the tilted square lattice."""

U_APPLICATION_COUNT = 2
"""In the echo, we apply the random circuit forwards and backwards, for two total applications."""

BASE_GB_COLS = ['run_id', 'processor_str', 'n_repetitions']
"""Basic grouping of runs."""

WH_GB_COLS = BASE_GB_COLS + ['width', 'height']
"""Additionally group by the width and height of the rectangle of qubits."""

A_GB_COLS = BASE_GB_COLS + ['q_area']
"""Additionally group by the quantum area (n_qubits*depth)."""

WHD_GB_COLS = WH_GB_COLS + ['macrocycle_depth']
"""Additionally group by width, height, and depth."""


def to_ground_state_prob(result: cirq.Result) -> float:
    """Compute the fraction of times we return to the state we started from.

    In a loschmidt echo, this is a measure of success since perfect execution of the random
    unitary and its inverse should return the system to its starting state.

    This function computes the fraction of time the all-zeros bitstring was measured. It assumes
    the system was initialized in the all-zeros state.
    """
    return np.mean(np.sum(result.measurements["z"], axis=1) == 0).item()


T = TypeVar('T')


def assert_one_unique_val(vals: Iterable[T]) -> T:
    """Extract one unique value from a column.

    Raises `AssertionError` if there is not exactly one unique value.

    Can be used during groupby aggregation to preserve a column you expect to
    have one consistent value in a given group.
    """
    vals = list(set(vals))
    if len(vals) != 1:
        raise AssertionError("Expected one unique value")
    return vals[0]


def groupby_all_except(df: pd.DataFrame, *, y_cols: Sequence[Any], agg_func: Any) \
        -> Tuple[pd.DataFrame, List[str]]:
    """Group by all columns except the named columns.

    This is a wrapper around `pd.DataFrame.groupby`. Instead of specifying the columns
    to group by, we group by all columns except the named columns. This can make analysis
    more robust: if you add a new parameter it will automatically be included in the
    groupby clauses without modifying the plotting code.

    This method also does "SQL-style" groupby by setting `as_index=False` (preventing
    row MultiIndices) and will flatten column MultiIndices to `_` delimited flat column names.

    Args:
        df: The dataframe
        y_cols: Columnn not to group by. These will be aggregated over.
        agg_func: The aggregation function

    Returns:
        The dataframe and list of columns used in the groupby.
    """
    groupby_cols = list(df.columns)
    for yc in y_cols:
        try:
            groupby_cols.remove(yc)
        except ValueError as e:
            raise ValueError(f"{yc} not in columns {groupby_cols}.")

    agged = df.groupby(groupby_cols, as_index=False).agg(agg_func)

    if isinstance(agged.columns, pd.MultiIndex):
        agged.columns = ['_'.join(x for x in column_tuple if x)
                         for column_tuple in agged.columns]

    return agged, groupby_cols


def _results_to_dataframe(
        results: 'ExecutableGroupResult',
        func: Callable[['ExecutableResult', 'QuantumRuntimeConfiguration', 'SharedRuntimeInfo'], Dict]
) -> pd.DataFrame:
    """Call a function on each result in an `ExecutableGroupResult` to construct a DataFrame."""
    return pd.DataFrame([func(result, results.runtime_configuration, results.shared_runtime_info)
                         for result in results.executable_results])


def loschmidt_results_to_dataframe(results: 'ExecutableGroupResult') -> pd.DataFrame:
    """Process an `ExecutableGroupResult`.

    This function performs the data analysis using `to_ground_state_prob` and
    extracts the most relevant quantities for a dataframe suitable for
    grouping, selecting, and plotting.

    In addition to `TiltedSquareLatticeLoschmidtSpec` input parameters, we also include
    the computed exogenous parameters "n_qubits" (a function of topology width and height)
    and "q_area" (a function of macrocycle_depth and n_qubits). The quantum area (q_area)
    is the number of qubits times the circuit depth. The circuit depth is 4 * 2 * macrocycle_depth
    accounting for the four directions of gates in the tilted square lattice and the two times
    we do U and/or its inverse.
    """

    def _to_record(result: 'ExecutableResult',
                   rt_config: 'QuantumRuntimeConfiguration',
                   shared_rt_info: 'SharedRuntimeInfo') -> Dict:
        success_prob = to_ground_state_prob(result.raw_data)
        spec = cast('TiltedSquareLatticeLoschmidtSpec', result.spec)

        record = {
            'run_id': shared_rt_info.run_id,
            'width': spec.topology.width,
            'height': spec.topology.height,
            'n_qubits': spec.topology.n_nodes,
            'macrocycle_depth': spec.macrocycle_depth,
            # "quantum area" is circuit width (n_qubits) times circuit depth.
            'q_area': ((spec.macrocycle_depth * CYCLES_PER_MACROCYCLE * U_APPLICATION_COUNT)
                       * spec.topology.n_nodes),
            'instance_i': spec.instance_i,
            'n_repetitions': spec.n_repetitions,
            'success_probability': success_prob,
            'processor_str': str(rt_config.processor_record),
        }

        if isinstance(result.raw_data, cg.EngineResult):
            record['job_finished_time'] = result.raw_data.job_finished_time
        else:
            record['job_finished_time'] = datetime.datetime.fromtimestamp(0)

        return record

    return _results_to_dataframe(results, _to_record)


def agg_vs_macrocycle_depth(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Aggregate mean and stddev vs macrocycle depth.

    We use pandas group-by functionality to
        1. Average (and compute the standard deviation) over random circuit
           instances, holding all else constant.
        2. Group these averaged quantities vs. macrocycle_depth for plotting by row.

    This aggregation uses `groupby_all_except` as a wrapper around `pd.DataFrame.groupby` so we
    can specify what we _don't_ want to aggregate over, making this function robust to the
    introduction of new columns.

    Args:
        df: The dataframe from `loschmidt_results_to_dataframe`.

    Returns:
        vs_depth_df: A new, aggregated dataframe.
        vs_depth_gb_cols: The named of the columns used in the final groupby operation.
    """

    # 1. Average over random circuit instances.
    means_y_cols = {
        'success_probability_mean': ('success_probability', 'mean'),
        'success_probability_std': ('success_probability', 'std'),
        'job_finished_time': ('job_finished_time', 'last'),
    }
    means_df = df.groupby(WHD_GB_COLS).agg(**means_y_cols)

    # 2. Group these averaged quantities into lists.
    # (a) first "ungroup" macrocycle_depth
    means_df = means_df.reset_index('macrocycle_depth')

    # (b) now do the list aggregation.
    vs_depth_y_cols = {
        'macrocycle_depth': list,
        'success_probability_mean': list,
        'success_probability_std': list,
        'job_finished_time': 'last',
    }

    vs_depth_df = means_df.groupby(WH_GB_COLS).agg(vs_depth_y_cols)
    return vs_depth_df, WH_GB_COLS


def fit_vs_macrocycle_depth(df):
    """Perform a fitting vs macrocycle depth given a loschmidt results dataframe.

    Args:
        df: The dataframe from `loschmidt_results_to_dataframe`.

    Returns:
        fit_df: A new dataframe containing fit parameters.
        exp_ansatz_vs_macrocycle_depth: The function used for the fit. This is
            a * f^depth. The depth in this expression is the number of macrocycles
            multiplied by four (to give the number of cycles) and multiplied by two
            (to account for the inversion in the echo).
    """

    def exp_ansatz_vs_macrocycle_depth(macrocycle_depths, a, f):
        return a * np.exp(
            np.log(f) * macrocycle_depths * CYCLES_PER_MACROCYCLE * U_APPLICATION_COUNT)

    def _fit(row):
        (a, f), pcov = curve_fit(
            exp_ansatz_vs_macrocycle_depth,
            xdata=row['macrocycle_depth'],
            ydata=row['success_probability'],
        )
        a_err, f_err = np.sqrt(np.diag(pcov))
        row['a'] = a
        row['f'] = f
        row['a_err'] = a_err
        row['f_err'] = f_err
        return row

    vs_size_y_cols = {
        'macrocycle_depth': list,
        'success_probability': list,
        'job_finished_time': 'last',
        'n_qubits': assert_one_unique_val,
    }
    vs_size_df = df.groupby(WH_GB_COLS).agg(vs_size_y_cols)

    fit_df_y_cols = ['job_finished_time', 'n_qubits', 'a', 'f', 'a_err', 'f_err']
    fit_df = vs_size_df.apply(_fit, axis=1).loc[:, fit_df_y_cols]
    return fit_df, exp_ansatz_vs_macrocycle_depth


def agg_vs_q_area(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Aggregate mean and stddev vs quantum area.

    quantum area is the circuit number of qubits multiplied by its depth.

    We use pandas group-by functionality to
        1. Average (and compute the standard deviation) over random circuit
           instances, holding all else constant.
        2. Group these averaged quantities vs. q_area for plotting by row.

    This aggregation uses `groupby_all_except` as a wrapper around `pd.DataFrame.groupby` so we
    can specify what we _don't_ want to aggregate over, making this function robust to the
    introduction of new columns.

    Args:
        df: The dataframe from `loschmidt_results_to_dataframe`.

    Returns:
        vs_q_area_df: A new, aggregated dataframe.
        vs_q_area_gb_cols: The named of the columns used in the final groupby operation.
    """
    means_y_cols = {
        'success_probability_mean': ('success_probability', 'mean'),
        'success_probability_std': ('success_probability', 'std'),
        'job_finished_time': ('job_finished_time', 'last'),
    }
    means_df = df.groupby(A_GB_COLS).agg(**means_y_cols)

    means_df = means_df.reset_index('q_area')
    vs_q_area_y_cols = {
        'q_area': list,
        'success_probability_mean': list,
        'success_probability_std': list,
        'job_finished_time': 'last',
    }
    vs_q_area_df = means_df.groupby(BASE_GB_COLS).agg(vs_q_area_y_cols)
    return vs_q_area_df, BASE_GB_COLS


def fit_vs_q_area(df):
    """Perform a fitting vs "quantum area", i.e. depth*n_qubits.

    Args:
        df: The dataframe from `loschmidt_results_to_dataframe`.

    Returns:
        fitted_df: A new dataframe containing fit parameters.
        exp_ansatz_vs_qarea: The function used for the fit.
    """

    def exp_ansatz_vs_q_area(q_area, a, f):
        return a * np.exp(np.log(f) * q_area)

    def _fit(row):
        (a, f), pcov = curve_fit(
            exp_ansatz_vs_q_area,
            xdata=row['q_area'],
            ydata=row['success_probability'],
        )
        a_err, f_err = np.sqrt(np.diag(pcov))
        row['a'] = a
        row['f'] = f
        row['a_err'] = a_err
        row['f_err'] = f_err
        return row

    vs_q_area_y_cols = {
        'q_area': list,
        'success_probability': list,
        'job_finished_time': 'last',
    }
    vs_q_area_df = df.groupby(BASE_GB_COLS).agg(vs_q_area_y_cols)

    fit_df_y_cols = ['job_finished_time', 'a', 'f', 'a_err', 'f_err']
    fit_df = vs_q_area_df.apply(_fit, axis=1).loc[:, fit_df_y_cols]
    return fit_df, exp_ansatz_vs_q_area
