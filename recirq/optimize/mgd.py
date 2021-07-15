# Copyright 2018 The Cirq Developers
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

from typing import Callable, List, Optional, Tuple

import numpy as np
import scipy
from scipy.optimize import OptimizeResult
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from recirq.optimize._util import wrap_function


def _get_least_squares_model_gradient(
        xs: List[np.ndarray],
        ys: List[float],
        xopt: np.ndarray,
) -> Tuple[np.ndarray, Pipeline]:
    """Fit a least squares quadratic model and return its gradient.

    Given a list of points and their function values, fits a least squares
    quadratic model to the function using polynomial features. Then,
    return the gradient of the model at a given point, and the model itself.
    This function is used by `model_gradient_descent` to estimate the gradient
    of the objective function.

    Args:
        xs: The points at which function values are given.
        ys: The function values at the points given by `xs`.
        xopt: The point at which to evaluate the gradient.

    Returns:
        A tuple whose first element is the gradient and second element is the
        regression model.
    """
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('linear_model', LinearRegression(fit_intercept=False)),
    ])
    shifted_xs = [x - xopt for x in xs]
    model = model.fit(shifted_xs, ys)
    fitted_coeffs = model.named_steps['linear_model'].coef_
    n = len(xs[0])
    linear_coeffs = np.array(fitted_coeffs[1:n + 1])
    return linear_coeffs, model


def _random_point_in_ball(n: int, radius: float) -> np.ndarray:
    """Return a point uniformly at random from a ball centered at the origin.

    Args:
        n: The dimension of the point to return.
        radius: The radius of the ball to sample from.

    Returns:
        The randomly sampled point.
    """
    point_on_sphere = np.random.randn(n)
    point_on_sphere /= np.linalg.norm(point_on_sphere)
    length = np.random.uniform()
    length = radius * length**(1 / n)
    return length * point_on_sphere


def model_gradient_descent(
        f: Callable[..., float],
        x0: np.ndarray,
        *,
        args=(),
        rate: float = 1e-1,
        sample_radius: float = 1e-1,
        n_sample_points: int = 100,
        n_sample_points_ratio: Optional[float] = None,
        rate_decay_exponent: float = 0.0,
        stability_constant: float = 0.0,
        sample_radius_decay_exponent: float = 0.0,
        tol: float = 1e-8,
        known_values: Optional[Tuple[List[np.ndarray], List[float]]] = None,
        max_iterations: Optional[int] = None,
        max_evaluations: Optional[int] = None) -> scipy.optimize.OptimizeResult:
    """Model gradient descent algorithm for black-box optimization.

    The idea of this algorithm is to perform gradient descent, but estimate
    the gradient using a surrogate model instead of, say, by
    finite-differencing. The surrogate model is a least-squared quadratic
    fit to points sampled from the vicinity of the current iterate.
    This algorithm works well when you have an initial guess which is in the
    convex neighborhood of a local optimum and you want to converge to that
    local optimum. It's meant to be used when the function is stochastic.

    Args:
        f: The function to minimize.
        x0: An initial guess.
        args: Additional arguments to pass to the function.
        rate: The learning rate for the gradient descent.
        sample_radius: The radius around the current iterate to sample
            points from to build the quadratic model.
        n_sample_points: The number of points to sample in each iteration.
        n_sample_points_ratio: This specifies the number of points to sample
            in each iteration as a coefficient of the number of points
            required to exactly determine a quadratic model. The number
            of sample points will be this coefficient times (n+1)(n+2)/2,
            rounded up, where n is the number of parameters.
            Setting this overrides n_sample_points.
        rate_decay_exponent: Controls decay of learning rate.
            In each iteration, the learning rate is changed to the
            base learning rate divided by (i + 1 + S)**a, where S
            is the stability constant and a is the rate decay exponent
            (this parameter).
        stability_constant: Affects decay of learning rate.
            In each iteration, the learning rate is changed to the
            base learning rate divided by (i + 1 + S)**a, where S
            is the stability constant (this parameter) and a is the rate decay
            exponent.
        sample_radius_decay_exponent: Controls decay of sample radius.
        tol: The algorithm terminates when the difference between the current
            iterate and the next suggested iterate is smaller than this value.
        known_values: Any prior known values of the objective function.
            This is given as a tuple where the first element is a list
            of points and the second element is a list of the function values
            at those points.
        max_iterations: The maximum number of iterations to allow before
            termination.
        max_evaluations: The maximum number of function evaluations to allow
            before termination.

    Returns:
        Scipy OptimizeResult
    """

    if known_values is not None:
        known_xs, known_ys = known_values
        known_xs = [np.copy(x) for x in known_xs]
        known_ys = [np.copy(y) for y in known_ys]
    else:
        known_xs, known_ys = [], []

    if max_iterations is None:
        max_iterations = np.inf
    if max_evaluations is None:
        max_evaluations = np.inf

    n = len(x0)
    if n_sample_points_ratio is not None:
        n_sample_points = int(
            np.ceil(n_sample_points_ratio * (n + 1) * (n + 2) / 2))

    _, f = wrap_function(f, args)
    res = OptimizeResult()
    current_x = np.copy(x0)
    res.x_iters = []  # initializes as lists
    res.xs_iters = []
    res.ys_iters = []
    res.func_vals = []
    res.model_vals = [None]
    res.fun = 0
    total_evals = 0
    num_iter = 0
    converged = False
    message = None

    while num_iter < max_iterations:
        current_sample_radius = (sample_radius /
                                 (num_iter + 1)**sample_radius_decay_exponent)

        # Determine points to evaluate
        # in ball around current point
        new_xs = [np.copy(current_x)] + [
            current_x + _random_point_in_ball(n, current_sample_radius)
            for _ in range(n_sample_points)
        ]

        if total_evals + len(new_xs) > max_evaluations:
            message = 'Reached maximum number of evaluations.'
            break

        # Evaluate points
        res.xs_iters.append(new_xs)
        new_ys = [f(x) for x in new_xs]
        res.ys_iters.append(new_ys)
        total_evals += len(new_ys)
        known_xs.extend(new_xs)
        known_ys.extend(new_ys)

        # Save function value
        res.func_vals.append(new_ys[0])
        res.x_iters.append(np.copy(current_x))
        res.fun = res.func_vals[-1]

        # Determine points to use to build model
        model_xs = []
        model_ys = []
        for x, y in zip(known_xs, known_ys):
            if np.linalg.norm(x - current_x) < current_sample_radius:
                model_xs.append(x)
                model_ys.append(y)
        # Build and solve model
        model_gradient, model = _get_least_squares_model_gradient(
            model_xs, model_ys, current_x)

        # calculate the gradient and update the current point
        gradient_norm = np.linalg.norm(model_gradient)
        decayed_rate = (
            rate / (num_iter + 1 + stability_constant)**rate_decay_exponent)
        # Convergence criteria
        if decayed_rate * gradient_norm < tol:
            converged = True
            message = 'Optimization converged successfully.'
            break
        # Update
        current_x -= decayed_rate * model_gradient
        res.model_vals.append(
            model.predict([-decayed_rate * model_gradient])[0])

        num_iter += 1

    if converged:
        final_val = res.func_vals[-1]
    else:
        final_val = f(current_x)
        res.func_vals.append(final_val)

    if message is None:
        message = 'Reached maximum number of iterations.'

    res.x_iters.append(current_x)
    total_evals += 1
    res.x = current_x
    res.fun = final_val
    res.nit = num_iter
    res.nfev = total_evals
    res.message = message
    return res
