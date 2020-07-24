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
from scipy.optimize.optimize import wrap_function
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


def _get_quadratic_model(
    xs: List[np.ndarray], ys: List[float], xopt: np.ndarray
) -> Pipeline:
    """Fit a least squares quadratic model.

    Given a list of points and their function values, fits a least squares
    quadratic model to the function using polynomial features. Then,
    return the model itself.
    This function is used by `model_policy_gradient` to estimate the gradient
    of the objective function.

    Args:
        xs: The points at which function values are given.
        ys: The function values at the points given by `xs`.
        xopt: The point at which to center.

    Returns:
        the regression model.
    """
    linear_model = LinearRegression(fit_intercept=False)
    model = Pipeline(
        [("poly", PolynomialFeatures(degree=2)), ("linear_model", linear_model),]
    )
    shifted_xs = [(x - xopt) for x in xs]
    model = model.fit(shifted_xs, ys)
    return model


def clip_by_norm(t, clip_norm):
    """Clips tensor values to a maximum L2-norm.
    returns a tensor of the same type and shape as `t` with its values set to:
    `t * clip_norm / l2norm(t)`

    Args:
        t: The tensor values at which we want to clip the norm.
        clip_norm: The scalar to give a maximal upper bound for the L2-norm of the tensor.

    Returns:
        the tensor with the same shape of the input, but with norm clipped.
    """
    l2_norm = np.linalg.norm(t, 2)
    return t * clip_norm / np.maximum(l2_norm, clip_norm)


class ConstantSchedule(object):
    def __init__(self, value):
        """The constant schedule for some hyperparameter (e.g. learning_rate)
        
        Value remains constant over time.

        Args: 
            value: Constant value of the schedule
        
        Returns: 
            a class of the schedule
        """
        self._v = value

    def value(self, t):
        """Return the value of the schedule at time step t 

        Args: 
            t: the time step for the schedule 

        Returns: 
            the schedule value
        """
        return self._v


class ExponentialSchedule(object):
    def __init__(self, learning_rate, decay_steps, decay_rate, staircase=False):
        """The Exponential schedule for some hyperparameter (e.g. learning_rate)
        
        Exponential decay for the `learning rate`. For each `decay_steps`, the learning 
        rate is scheduled to decay at the `decay_rate`. The `staircase` controls whether 
        to decay smoothly or discontinuously. After this many timesteps pass, the final
        learning rate is returned.

        Args: 
            learning rate: the initial learning rate 
            decay_steps: the learning rate is scheduled to decay every such number of steps 
            decay_rate: the learning rate is scheduled to decay at the such rate
            staircase: if True, the learning rate keeps the same before every decay steps; 
                       otherwise, the learning rate decays smoothly according 
                       to exponential interpolation.
        
        Returns: 
            a class of the schedule
        """
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    def value(self, t):
        """Return the value of the schedule at time step t 

        Args: 
            t: the time step for the schedule 

        Returns: 
            the schedule value
        """
        m = t / self.decay_steps
        if self.staircase == True:
            m = np.floor(m)

        return self.learning_rate * self.decay_rate ** m


class Adam:
    def __init__(
        self, lr_schedule=ConstantSchedule(0.001), b1=0.9, b2=0.999, eps=10 ** -8
    ):
        """The optimizer Adam: a method for stochastic gradient descent  

        Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
        adapted from https://github.com/HIPS/autograd/blob/master/autograd/misc/optimizers.py#L57-L71

        Args:
            lr_schedule: the class of learning rate decay schedule. Defaults to ConstantSchedule(0.001).
            b1 (float, optional): coefficients used for computing
                running averages of gradient. Defaults to 0.9.
            b2 (float, optional): coefficients used for computing
                running averages of gradient's square. Defaults to 0.999.
            eps (float, optional): term added to the denominator to improve
                numerical stability. Defaults to 10**-8.
        """
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.step = 0
        self.lr_schedule = lr_schedule

    def ascend(self, grad, x):
        """Performs a single optimization step.


        Args:
            grad: the gradient computed at the current update.  
            x: the current value of the parameters. 

        Returns:
            the updated parameter values. 
        """
        if self.step == 0:
            self.m = np.zeros(len(x))
            self.v = np.zeros(len(x))
        self.lr = self.lr_schedule.value(self.step)

        # First  moment estimate.
        self.m = (1 - self.b1) * grad + self.b1 * self.m
        # Second moment estimate.
        self.v = (1 - self.b2) * (grad ** 2) + self.b2 * self.v

        # Bias correction.
        mhat = self.m / (1 - self.b1 ** (self.step + 1))
        vhat = self.v / (1 - self.b2 ** (self.step + 1))
        x = x + self.lr * mhat / (np.sqrt(vhat) + self.eps)
        self.step += 1

        return x


def model_policy_gradient(
    f: Callable[..., float],
    x0: np.ndarray,
    *,
    args=(),
    learning_rate: float = 1e-2,
    decay_rate: float = 0.96,
    decay_steps: int = 5,
    log_sigma_init: float = -5.0,
    max_iterations: int = 1000,
    batch_size: int = 10,
    radius_coeff: float = 3.0,
    warmup_steps: int = 10,
    batch_size_model: int = 65536,
    save_func_vals: bool = False, 
    known_values: Optional[Tuple[List[np.ndarray], List[float]]] = None,
    max_evaluations: Optional[int] = None
) -> scipy.optimize.OptimizeResult:

    """Model policy gradient algorithm for black-box optimization.

    The idea of this algorithm is to perform policy gradient, but estimate
    the function values using a surrogate model. 
    The surrogate model is a least-squared quadratic
    fit to points sampled from the vicinity of the current iterate.

    Args:
        f: The function to minimize.
        x0: An initial guess.
        args: Additional arguments to pass to the function.
        learning_rate: The learning rate for the policy gradient.
        decay_rate: the learning decay rate for the Adam optimizer.
        decay_steps: the learning decay steps for the Adam optimizer.
        log_sigma_init: the intial value for the sigma of the policy
            in the log scale. 
        max_iterations: The maximum number of iterations to allow before
            termination.
        batch_size: The number of points to sample in each iteration. The cost 
            of evaluation of these samples are computed through the 
            quantum computer cost model.
        radius_coeff: The ratio determining the size of the radius around 
            the current iterate to sample points from to build the quadratic model.
            The ratio is with respect to the maximal ratio of the samples 
            from the current policy. 
        warmup_steps: The number of steps before the model policy gradient is performed. 
            before these steps, we use the policy gradient without the model. 
        batch_size_model: The model sample batch size. 
            After we fit the quadratic model, we use the model to evaluate 
            on big enough batch of samples.
        save_func_vals: whether to compute and save the function values for 
            the current value of parameter.   
        known_values: Any prior known values of the objective function.
            This is given as a tuple where the first element is a list
            of points and the second element is a list of the function values
            at those points.
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

    if max_evaluations is None:
        max_evaluations = np.inf

    n = len(x0)
    log_sigma = np.ones(n) * log_sigma_init
    sigma = np.exp(log_sigma)

    # set up lr schedule and optimizer
    lr_schedule1 = ExponentialSchedule(
        learning_rate, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True
    )
    lr_schedule2 = ExponentialSchedule(
        learning_rate, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True
    )
    optimizer1 = Adam(lr_schedule1)
    optimizer2 = Adam(lr_schedule2)

    _, f = wrap_function(f, args)
    res = OptimizeResult()
    current_x = np.copy(x0)
    res.x_iters = []  # initializes as lists
    res.xs_iters = []
    res.ys_iters = []
    res.func_vals = []
    res.fun = 0
    total_evals = 0
    num_iter = 0
    message = None

    # stats
    history_max = -np.inf

    while num_iter < max_iterations:
        # get samples from the current policy to evaluate
        z = np.random.randn(batch_size, n)
        new_xs = sigma * z + current_x

        if total_evals + batch_size > max_evaluations:
            message = "Reached maximum number of evaluations."
            break

        # Evaluate points
        res.xs_iters.append(new_xs)
        new_ys = [f(x) for x in new_xs]
        res.ys_iters.append(new_ys)
        total_evals += batch_size
        known_xs.extend(new_xs)
        known_ys.extend(new_ys)

        # Save function value
        if save_func_vals:
            res.func_vals.append(f(current_x))
            res.x_iters.append(np.copy(current_x))
            res.fun = res.func_vals[-1]

        # current sampling radius (maximal)
        max_radius = 0
        for x in new_xs:
            if np.linalg.norm(x - current_x) > max_radius:
                max_radius = np.linalg.norm(x - current_x)

        reward = [-y for y in new_ys]

        # warmup steps control whether to use the model to estimate the f
        if num_iter >= warmup_steps:
            # Determine points to use to build model
            model_xs = []
            model_ys = []
            # safer way without the `SVD` not converging
            try:
                for x, y in zip(known_xs, known_ys):
                    if np.linalg.norm(x - current_x) < radius_coeff * max_radius:
                        model_xs.append(x)
                        model_ys.append(y)
                model = _get_quadratic_model(model_xs, model_ys, x)

                # get samples (from model)
                z = np.random.randn(batch_size_model, n)
                new_xs = sigma * z + current_x

                # use the model for prediction
                new_ys = model.predict(new_xs - current_x)
                reward = [-y for y in new_ys]
            except:
                pass

        reward = np.array(reward)

        # stats
        reward_mean = np.mean(reward)
        reward_max = np.max(reward)

        if reward_max > history_max:
            history_max = reward_max

        # subtract baseline
        reward = reward - np.mean(reward)

        # analytic derivatives (natural gradient policy gradient)
        delta_mean = np.dot(z.T, reward) * sigma
        delta_log_sigma = np.dot(z.T ** 2, reward) / np.sqrt(2)

        delta_mean_norm = np.linalg.norm(np.dot(z.T, reward))
        delta_log_sigma_norm = np.linalg.norm(np.dot(z.T ** 2, reward))

        delta_mean = delta_mean / delta_mean_norm
        delta_log_sigma = delta_log_sigma / delta_log_sigma_norm

        # gradient ascend to update the parameters
        current_x = optimizer1.ascend(delta_mean, current_x)
        log_sigma = optimizer2.ascend(delta_log_sigma, log_sigma)

        log_sigma = np.clip(log_sigma, -20.0, 2.0)
        sigma = np.exp(log_sigma)

        num_iter += 1

    final_val = f(current_x)
    res.func_vals.append(final_val)

    if message is None:
        message = "Reached maximum number of iterations."

    res.x_iters.append(current_x)
    total_evals += 1
    res.x = current_x
    res.fun = final_val
    res.nit = num_iter
    res.nfev = total_evals
    res.message = message
    return res
