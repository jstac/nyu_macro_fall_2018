import numpy as np
from numba import jit

@jit(nopython=True)
def interp1d(grid, vals, x):
    """
    Linearly interpolate (grid, vals) to evaluate at x.

    Parameters
    ----------
        grid and vals are numpy arrays, x is a float

    Returns
    -------
        a float, the interpolated value

    """

    a, b, G = np.min(grid), np.max(grid), len(grid)

    s = (x - a) / (b - a)

    q_0 = max(min(int(s * (G - 1)), (G - 2)), 0)
    v_0 = vals[q_0]
    v_1 = vals[q_0 + 1]

    λ = s * (G - 1) - q_0

    return (1 - λ) * v_0 + λ * v_1


@jit(nopython=True)
def interp1d_vectorized(grid, vals, x_vec):
    """
    Linearly interpolate (grid, vals) to evaluate at x_vec.

    All inputs are numpy arrays.

    Return value is a numpy array of length len(x_vec).
    """

    out = np.empty_like(x_vec)

    for i, x in enumerate(x_vec):
        out[i] = interp1d(grid, vals, x)

    return out


