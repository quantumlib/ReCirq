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

"""Utilities to "correct" measured probability distributions using "unfolding."

Based on Nachman et al., "Unfolding quantum computer readout noise"
npj Quantum Information 6 84 (2020) https://doi.org/10.1038/s41534-020-00309-7

This is similar to using "readout matrix inversion" but more numerically stable.
Note it does NOT take into account state preparation error, such as stray |1) population
and pi pulse error.

The readout matrix (R_ij in Nachman et al.) is:
 - Nachman: R_ij = Prob(measured=i | prepared=j) (beginning of "Methods" section)
    → Columns sum to 1

Consistent bitstring ordering and endianness is needed for compatibility
with other libraries.

This is an iterative technique. The key equation is Nachman (5). The number of iterations
to use is not obvious (see Fig. 6), but in practice N=20 iterations works well.

Nachman uses a uniform prior ("guess") probability distribution. This is the most "agnostic"
choice possible. However, starting from the measured distribution works better empircally.
"""

import numpy as np


def _get_initial_guess(
    measured: np.ndarray, use_uniform_prior: bool = False
) -> np.ndarray:
    """Initial (prior) probability distribution."""
    if use_uniform_prior:
        return np.ones_like(measured) / len(measured)
    return np.array(measured)  # Copy to avoid side effects


def correct_readout_distribution(
    readout_matrix: np.ndarray,
    measured: np.ndarray,
    iterations: int = 20,
    use_uniform_prior: bool = False,
) -> np.ndarray:
    """Use iterative unfolding to correct the measured probabilities.

    This implements Nachman eqn. (5).

    Args:
        readout_matrix: Measured readout correction matrix.
            readout_matrix[row, col] = prob(measured=col | prepared=row); rows sum to 1
            (transpose of "R" in Nachman; see beginning of "methods" section).
            Must be a square matrix, shape (2**n, 2**n)
        measured: Probability distribution to correct, 2**n entries
        iterations: Number of iterations ("N" in Nachman; see eqn. (5))
        use_uniform_prior: If True, the initial guess probability distribution is
            uniform over all bitstrings (as in Nachman). If False, the initial
            guess is the measured distribution. See module docstring.

    Returns:
        Corrected version of measured_probabilities
    """
    r = readout_matrix.transpose()
    corrected = _get_initial_guess(measured, use_uniform_prior)

    for _ in range(iterations):
        denominators = r @ corrected
        # By construction, if denominator is zero, so is the numerator; skip 0 / 0 cases
        nonzero = np.logical_not(np.isclose(denominators, 0))

        for idx in range(len(corrected)):
            updates = r[nonzero, idx] * measured[nonzero] / denominators[nonzero]
            corrected[idx] *= np.sum(updates)

    return corrected / np.sum(corrected)
