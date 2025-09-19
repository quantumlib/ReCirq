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

import pandas as pd
import pytest

import recirq.otoc.loschmidt.tilted_square_lattice.analysis as analysis

if not analysis.workflow:
    pytestmark = pytest.mark.skip('Requires Cirq >= 0.14.')


@pytest.fixture
def sample_df():
    # steps to (re-)generate:
    # 1. load an example `df` like in `analysis-walkthrough.ipynb`
    # 2. use print(repr(df.to_dict()))
    # 3. s/Timestamp/pd.Timestamp/g
    # 4. add a `pd.DataFrame.from_dict()` around it

    return pd.DataFrame.from_dict(
        {'run_id': {0: 'simulated-1', 1: 'simulated-1', 2: 'simulated-1', 3: 'simulated-1',
                    4: 'simulated-1', 5: 'simulated-1', 6: 'simulated-1', 7: 'simulated-1',
                    8: 'simulated-1', 9: 'simulated-1', 10: 'simulated-1', 11: 'simulated-1',
                    12: 'simulated-1', 13: 'simulated-1', 14: 'simulated-1', 15: 'simulated-1',
                    16: 'simulated-1', 17: 'simulated-1', 18: 'simulated-1', 19: 'simulated-1',
                    20: 'simulated-1', 21: 'simulated-1', 22: 'simulated-1', 23: 'simulated-1',
                    24: 'simulated-1', 25: 'simulated-1', 26: 'simulated-1', 27: 'simulated-1',
                    28: 'simulated-1', 29: 'simulated-1', 30: 'simulated-1', 31: 'simulated-1',
                    32: 'simulated-1', 33: 'simulated-1', 34: 'simulated-1', 35: 'simulated-1',
                    36: 'simulated-1', 37: 'simulated-1', 38: 'simulated-1', 39: 'simulated-1',
                    40: 'simulated-1', 41: 'simulated-1', 42: 'simulated-1', 43: 'simulated-1',
                    44: 'simulated-1'},
         'width': {0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2, 11: 2,
                   12: 2, 13: 2, 14: 2, 15: 2, 16: 2, 17: 2, 18: 2, 19: 2, 20: 2, 21: 2, 22: 2,
                   23: 2, 24: 2, 25: 2, 26: 2, 27: 2, 28: 2, 29: 2, 30: 3, 31: 3, 32: 3, 33: 3,
                   34: 3, 35: 3, 36: 3, 37: 3, 38: 3, 39: 3, 40: 3, 41: 3, 42: 3, 43: 3, 44: 3},
         'height': {0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2, 11: 2,
                    12: 2, 13: 2, 14: 2, 15: 3, 16: 3, 17: 3, 18: 3, 19: 3, 20: 3, 21: 3, 22: 3,
                    23: 3, 24: 3, 25: 3, 26: 3, 27: 3, 28: 3, 29: 3, 30: 3, 31: 3, 32: 3, 33: 3,
                    34: 3, 35: 3, 36: 3, 37: 3, 38: 3, 39: 3, 40: 3, 41: 3, 42: 3, 43: 3, 44: 3},
         'n_qubits': {0: 5, 1: 5, 2: 5, 3: 5, 4: 5, 5: 5, 6: 5, 7: 5, 8: 5, 9: 5, 10: 5, 11: 5,
                      12: 5, 13: 5, 14: 5, 15: 6, 16: 6, 17: 6, 18: 6, 19: 6, 20: 6, 21: 6,
                      22: 6, 23: 6, 24: 6, 25: 6, 26: 6, 27: 6, 28: 6, 29: 6, 30: 8, 31: 8,
                      32: 8, 33: 8, 34: 8, 35: 8, 36: 8, 37: 8, 38: 8, 39: 8, 40: 8, 41: 8,
                      42: 8, 43: 8, 44: 8},
         'macrocycle_depth': {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3,
                              11: 3, 12: 4, 13: 4, 14: 4, 15: 0, 16: 0, 17: 0, 18: 1, 19: 1,
                              20: 1, 21: 2, 22: 2, 23: 2, 24: 3, 25: 3, 26: 3, 27: 4, 28: 4,
                              29: 4, 30: 0, 31: 0, 32: 0, 33: 1, 34: 1, 35: 1, 36: 2, 37: 2,
                              38: 2, 39: 3, 40: 3, 41: 3, 42: 4, 43: 4, 44: 4},
         'q_area': {0: 0, 1: 0, 2: 0, 3: 40, 4: 40, 5: 40, 6: 80, 7: 80, 8: 80, 9: 120, 10: 120,
                    11: 120, 12: 160, 13: 160, 14: 160, 15: 0, 16: 0, 17: 0, 18: 48, 19: 48,
                    20: 48, 21: 96, 22: 96, 23: 96, 24: 144, 25: 144, 26: 144, 27: 192, 28: 192,
                    29: 192, 30: 0, 31: 0, 32: 0, 33: 64, 34: 64, 35: 64, 36: 128, 37: 128,
                    38: 128, 39: 192, 40: 192, 41: 192, 42: 256, 43: 256, 44: 256},
         'instance_i': {0: 0, 1: 1, 2: 2, 3: 0, 4: 1, 5: 2, 6: 0, 7: 1, 8: 2, 9: 0, 10: 1, 11: 2,
                        12: 0, 13: 1, 14: 2, 15: 0, 16: 1, 17: 2, 18: 0, 19: 1, 20: 2, 21: 0,
                        22: 1, 23: 2, 24: 0, 25: 1, 26: 2, 27: 0, 28: 1, 29: 2, 30: 0, 31: 1,
                        32: 2, 33: 0, 34: 1, 35: 2, 36: 0, 37: 1, 38: 2, 39: 0, 40: 1, 41: 2,
                        42: 0, 43: 1, 44: 2},
         'n_repetitions': {0: 1000, 1: 1000, 2: 1000, 3: 1000, 4: 1000, 5: 1000, 6: 1000,
                           7: 1000, 8: 1000, 9: 1000, 10: 1000, 11: 1000, 12: 1000, 13: 1000,
                           14: 1000, 15: 1000, 16: 1000, 17: 1000, 18: 1000, 19: 1000, 20: 1000,
                           21: 1000, 22: 1000, 23: 1000, 24: 1000, 25: 1000, 26: 1000, 27: 1000,
                           28: 1000, 29: 1000, 30: 1000, 31: 1000, 32: 1000, 33: 1000, 34: 1000,
                           35: 1000, 36: 1000, 37: 1000, 38: 1000, 39: 1000, 40: 1000, 41: 1000,
                           42: 1000, 43: 1000, 44: 1000},
         'success_probability': {0: 0.959, 1: 0.945, 2: 0.947, 3: 0.703, 4: 0.692, 5: 0.697,
                                 6: 0.488, 7: 0.508, 8: 0.493, 9: 0.368, 10: 0.367, 11: 0.387,
                                 12: 0.258, 13: 0.255, 14: 0.256, 15: 0.936, 16: 0.94, 17: 0.938,
                                 18: 0.657, 19: 0.624, 20: 0.641, 21: 0.454, 22: 0.45, 23: 0.452,
                                 24: 0.284, 25: 0.291, 26: 0.305, 27: 0.19, 28: 0.182, 29: 0.178,
                                 30: 0.924, 31: 0.929, 32: 0.92, 33: 0.564, 34: 0.529, 35: 0.555,
                                 36: 0.306, 37: 0.296, 38: 0.319, 39: 0.194, 40: 0.172,
                                 41: 0.192, 42: 0.098, 43: 0.097, 44: 0.105},
         'processor_str': {0: 'rainbow-depol(5.000e-03)', 1: 'rainbow-depol(5.000e-03)',
                           2: 'rainbow-depol(5.000e-03)', 3: 'rainbow-depol(5.000e-03)',
                           4: 'rainbow-depol(5.000e-03)', 5: 'rainbow-depol(5.000e-03)',
                           6: 'rainbow-depol(5.000e-03)', 7: 'rainbow-depol(5.000e-03)',
                           8: 'rainbow-depol(5.000e-03)', 9: 'rainbow-depol(5.000e-03)',
                           10: 'rainbow-depol(5.000e-03)', 11: 'rainbow-depol(5.000e-03)',
                           12: 'rainbow-depol(5.000e-03)', 13: 'rainbow-depol(5.000e-03)',
                           14: 'rainbow-depol(5.000e-03)', 15: 'rainbow-depol(5.000e-03)',
                           16: 'rainbow-depol(5.000e-03)', 17: 'rainbow-depol(5.000e-03)',
                           18: 'rainbow-depol(5.000e-03)', 19: 'rainbow-depol(5.000e-03)',
                           20: 'rainbow-depol(5.000e-03)', 21: 'rainbow-depol(5.000e-03)',
                           22: 'rainbow-depol(5.000e-03)', 23: 'rainbow-depol(5.000e-03)',
                           24: 'rainbow-depol(5.000e-03)', 25: 'rainbow-depol(5.000e-03)',
                           26: 'rainbow-depol(5.000e-03)', 27: 'rainbow-depol(5.000e-03)',
                           28: 'rainbow-depol(5.000e-03)', 29: 'rainbow-depol(5.000e-03)',
                           30: 'rainbow-depol(5.000e-03)', 31: 'rainbow-depol(5.000e-03)',
                           32: 'rainbow-depol(5.000e-03)', 33: 'rainbow-depol(5.000e-03)',
                           34: 'rainbow-depol(5.000e-03)', 35: 'rainbow-depol(5.000e-03)',
                           36: 'rainbow-depol(5.000e-03)', 37: 'rainbow-depol(5.000e-03)',
                           38: 'rainbow-depol(5.000e-03)', 39: 'rainbow-depol(5.000e-03)',
                           40: 'rainbow-depol(5.000e-03)', 41: 'rainbow-depol(5.000e-03)',
                           42: 'rainbow-depol(5.000e-03)', 43: 'rainbow-depol(5.000e-03)',
                           44: 'rainbow-depol(5.000e-03)'},
         'job_finished_time': {0: pd.Timestamp('1969-12-31 16:00:00'),
                               1: pd.Timestamp('1969-12-31 16:00:00'),
                               2: pd.Timestamp('1969-12-31 16:00:00'),
                               3: pd.Timestamp('1969-12-31 16:00:00'),
                               4: pd.Timestamp('1969-12-31 16:00:00'),
                               5: pd.Timestamp('1969-12-31 16:00:00'),
                               6: pd.Timestamp('1969-12-31 16:00:00'),
                               7: pd.Timestamp('1969-12-31 16:00:00'),
                               8: pd.Timestamp('1969-12-31 16:00:00'),
                               9: pd.Timestamp('1969-12-31 16:00:00'),
                               10: pd.Timestamp('1969-12-31 16:00:00'),
                               11: pd.Timestamp('1969-12-31 16:00:00'),
                               12: pd.Timestamp('1969-12-31 16:00:00'),
                               13: pd.Timestamp('1969-12-31 16:00:00'),
                               14: pd.Timestamp('1969-12-31 16:00:00'),
                               15: pd.Timestamp('1969-12-31 16:00:00'),
                               16: pd.Timestamp('1969-12-31 16:00:00'),
                               17: pd.Timestamp('1969-12-31 16:00:00'),
                               18: pd.Timestamp('1969-12-31 16:00:00'),
                               19: pd.Timestamp('1969-12-31 16:00:00'),
                               20: pd.Timestamp('1969-12-31 16:00:00'),
                               21: pd.Timestamp('1969-12-31 16:00:00'),
                               22: pd.Timestamp('1969-12-31 16:00:00'),
                               23: pd.Timestamp('1969-12-31 16:00:00'),
                               24: pd.Timestamp('1969-12-31 16:00:00'),
                               25: pd.Timestamp('1969-12-31 16:00:00'),
                               26: pd.Timestamp('1969-12-31 16:00:00'),
                               27: pd.Timestamp('1969-12-31 16:00:00'),
                               28: pd.Timestamp('1969-12-31 16:00:00'),
                               29: pd.Timestamp('1969-12-31 16:00:00'),
                               30: pd.Timestamp('1969-12-31 16:00:00'),
                               31: pd.Timestamp('1969-12-31 16:00:00'),
                               32: pd.Timestamp('1969-12-31 16:00:00'),
                               33: pd.Timestamp('1969-12-31 16:00:00'),
                               34: pd.Timestamp('1969-12-31 16:00:00'),
                               35: pd.Timestamp('1969-12-31 16:00:00'),
                               36: pd.Timestamp('1969-12-31 16:00:00'),
                               37: pd.Timestamp('1969-12-31 16:00:00'),
                               38: pd.Timestamp('1969-12-31 16:00:00'),
                               39: pd.Timestamp('1969-12-31 16:00:00'),
                               40: pd.Timestamp('1969-12-31 16:00:00'),
                               41: pd.Timestamp('1969-12-31 16:00:00'),
                               42: pd.Timestamp('1969-12-31 16:00:00'),
                               43: pd.Timestamp('1969-12-31 16:00:00'),
                               44: pd.Timestamp('1969-12-31 16:00:00')}})


def test_agg_vs_macrocycle_depth(sample_df):
    vs_depth_df, gb_cols = analysis.agg_vs_macrocycle_depth(sample_df)
    assert 'width' in gb_cols
    assert 'height' in gb_cols
    assert 'macrocycle_depth' not in gb_cols
    assert vs_depth_df.index.names == gb_cols
    assert sorted(vs_depth_df.columns) == sorted(['macrocycle_depth', 'success_probability_mean',
                                                  'success_probability_std', 'job_finished_time'])


def test_agg_vs_q_area(sample_df):
    vs_q_area_df, gb_cols = analysis.agg_vs_q_area(sample_df)
    assert 'run_id' in gb_cols
    assert 'q_area' not in gb_cols
    assert vs_q_area_df.index.names == gb_cols
    assert sorted(vs_q_area_df.columns) == sorted(['q_area', 'success_probability_mean',
                                                   'success_probability_std', 'job_finished_time'])


def test_fit_vs_macrocycle_depth(sample_df):
    fit_df, exp_ansatz = analysis.fit_vs_macrocycle_depth(sample_df)
    assert len(fit_df) == 3, '3 different topo shapes in sample df'
    assert sorted(fit_df['n_qubits']) == [5, 6, 8]
    for c in ['a', 'f', 'a_err', 'f_err']:
        assert c in fit_df.columns, c


def test_fit_vs_q_area(sample_df):
    fit_df, exp_ansatz = analysis.fit_vs_q_area(sample_df)
    assert len(fit_df) == 1, '1 processor/run'
    for c in ['a', 'f', 'a_err', 'f_err']:
        assert c in fit_df.columns, c
