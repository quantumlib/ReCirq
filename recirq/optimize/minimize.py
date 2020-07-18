from typing import Callable, Optional, Tuple

import numpy as np
import scipy.optimize
from recirq.optimize.mgd import model_gradient_descent
from recirq.optimize.mpg import model_policy_gradient


OPTIMIZERS = {'mgd': model_gradient_descent, 'mpg': model_policy_gradient}


def minimize(fun: Callable[..., float],
             x0: np.ndarray,
             args: Tuple = (),
             method: Optional[str] = None,
             **kwargs) -> scipy.optimize.OptimizeResult:
    """Minimize a scalar function of one or more variables.

    Attempts to match `method` with an optimization method implemented in
    ReCirq. If there is no match, then delegates to `scipy.optimize.minimize`.
    The optimization method is called with the arguments
    (fun, x0, args=args, **kwargs).

    Args:
        fun: The objective function to minimize.
        x0: The initial guess.
        args: Extra arguments passed to the objective function.
        method: The name of the optimization method to use. Can be any method
            supported by `scipy.optimize.minimize`, or else a method implemented
            in ReCirq. Currently the methods implemented in ReCirq are 'MGD' and 'mpg'.
    """
    if method.lower() in OPTIMIZERS:
        optimizer = OPTIMIZERS[method.lower()]
        return optimizer(fun, x0, args=args, **kwargs)
    return scipy.optimize.minimize(fun, x0, args=args, method=method, **kwargs)
