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

from typing import Callable, Dict, cast, Sequence, Any, Tuple, List

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import cirq
from cirq_google.workflow import (
    ExecutableGroupResult, SharedRuntimeInfo,
    QuantumRuntimeConfiguration,
    ExecutableResult)
from recirq.otoc.loschmidt.tilted_square_lattice import TiltedSquareLatticeLoschmidtSpec


def to_ground_state_prob(result: cirq.Result) -> float:
    """Compute the fraction of times we return to the state we started from.

    In a loschmidt echo, this is a measure of success since perfect execution of the random
    unitary and its inverse should return the system to its starting state.

    This function computes the fraction of time the all-zeros bitstring was measured. It assumes
    the system was initialized in the all-zeros state.
    """
    return np.mean(np.sum(result.measurements["z"], axis=1) == 0).item()


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
        results: ExecutableGroupResult,
        func: Callable[[ExecutableResult, QuantumRuntimeConfiguration, SharedRuntimeInfo], Dict]
) -> pd.DataFrame:
    """Call a function on each result in an `ExecutableGroupResult` to construct a DataFrame."""
    return pd.DataFrame([func(result, results.runtime_configuration, results.shared_runtime_info)
                         for result in results.executable_results])


def loschmidt_results_to_dataframe(results: ExecutableGroupResult) -> pd.DataFrame:
    """Process an `ExecutableGroupResult`.

    This function performs the data analysis using `to_ground_state_prob` and
    extracts the most relevant quantities for a dataframe suitable for
    grouping, selecting, and plotting.
    """

    def _to_record(result: ExecutableResult,
                   rt_config: QuantumRuntimeConfiguration,
                   shared_rt_info: SharedRuntimeInfo) -> Dict:
        success_prob = to_ground_state_prob(result.raw_data)
        spec = cast(TiltedSquareLatticeLoschmidtSpec, result.spec)

        return {
            'run_id': shared_rt_info.run_id,
            'width': spec.topology.width,
            'height': spec.topology.height,
            'n_qubits': spec.topology.n_nodes,
            'macrocycle_depth': spec.macrocycle_depth,
            'q_area': (spec.macrocycle_depth * 4 * 2) * spec.topology.n_nodes,
            'instance_i': spec.instance_i,
            'n_repetitions': spec.n_repetitions,
            'success_probability': success_prob,
            'processor_id': rt_config.processor_record.processor_id,
        }

    return _results_to_dataframe(results, _to_record)


def fit_vs_macrocycle_depth(df):
    """Perform a fitting vs macrocycle depth given a loschmidt results dataframe.

    Args:
        df: The dataframe from `loschmidt_results_to_dataframe`.

    Returns:
        fitted_df: A new dataframe containing fit parameters.
        exp_ansatz_vs_macrocycle_depth: The function used for the fit.
    """

    def exp_ansatz_vs_macrocycle_depth(macrocycle_depths, a, f):
        return a * np.exp(np.log(f) * 4 * 2 * macrocycle_depths)

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

    y_cols = ['instance_i', 'macrocycle_depth', 'q_area', 'success_probability']
    agged, _ = groupby_all_except(
        df,
        y_cols=y_cols,
        agg_func=list,
    )
    fitted_df = agged.apply(_fit, axis=1) \
        .drop(y_cols, axis=1)
    return fitted_df, exp_ansatz_vs_macrocycle_depth


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

    y_cols = ['q_area', 'n_qubits', 'instance_i', 'macrocycle_depth', 'success_probability']
    agged, _ = groupby_all_except(
        df.drop(['width', 'height'], axis=1),
        y_cols=y_cols,
        agg_func=list,
    )
    fit_df = agged.apply(_fit, axis=1).drop(y_cols, axis=1)

    return fit_df, exp_ansatz_vs_q_area
