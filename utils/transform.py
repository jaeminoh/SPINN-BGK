import jax.numpy as jnp
import numpy as np

def to_column(lst: list[jnp.ndarray]) -> list[jnp.ndarray]:
    # Converting the elements into the column vectors. (N,1)
    return [elt.reshape(-1, 1) for elt in lst]


def to_vec(lst: list[jnp.ndarray]) -> list[jnp.ndarray]:
    # Converting the elements into the vectors. (N,)
    return [elt.ravel() for elt in lst]


def Lp_norm(x: np.ndarray, h: float = None, p: int = 2, d: int = None, plus_one=False):
    integrand = np.abs(x)
    integral = (np.power(integrand, p) * h**d).sum()
    norm = np.power(integral, 1 / p)
    if plus_one:
        norm += 1
    return norm


def trapezoidal_rule(n: int, a: float = -1.0, b: float = 1.0) -> list[jnp.ndarray]:
    """
    1d trapezoidal rule.
    Very efficient for periodic functions, or peak functions like gaussian.
    n: the number of intervals
    a: left end point
    b: right end point
    """
    x = jnp.linspace(a, b, n + 1)
    h = (b - a) / n
    w = jnp.stack([h / 2] + [h] * (n - 1) + [h / 2])
    return x, w


def gaussian(v: jnp.ndarray, mu: float = 0.0, tau: float = 1.0) -> jnp.ndarray:
    """
    ijnputs:
        v: query point(s)
        mu: mean
        tau: precision, = 1/std
    outputs:
        N(mu, 1/tau^2)(v) without normalizing constant.
    """
    pdf = jnp.exp(-0.5 * (tau * (v - mu)) ** 2)
    return pdf


def maxwellian3d(rho, u, temp, vx, vy, vz) -> jnp.ndarray:
    """
    Maxwellian, 3d veloicty space. (..., Nv,Nv,Nv) shaped.
    """
    rho, (ux, uy, uz), temp = [jnp.expand_dims(m, (-3, -2, -1)) for m in (rho, u, temp)]
    vx, vy, vz = vx[:, None, None], vy[None, :, None], vz[None, None, :]
    maxwellian = (
        rho
        / jnp.power(2 * jnp.pi * temp, 1.5)
        * jnp.exp(-((vx - ux) ** 2 + (vy - uy) ** 2 + (vz - uz) ** 2) / (2 * temp))
    )
    return maxwellian
