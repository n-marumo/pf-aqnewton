from .. import jax_config  # noqa: F401
import jax.numpy as jnp
import jax.random as jrandom

from .base import ProblemBase


class DixonPrice(ProblemBase):
    def __init__(self, dim: int, seed: int = 0, sigma_init: float = 1.0):
        assert dim >= 2
        self.dim = dim
        self.seed = seed
        self.sigma_init = sigma_init

    def f(self, x: jnp.ndarray) -> jnp.ndarray:
        term1 = (x[0] - 1.0) ** 2
        idx = jnp.arange(2, self.dim + 1)
        xi = x[1:]
        xi_prev = x[:-1]
        term2 = jnp.sum(idx * (2.0 * xi**2 - xi_prev) ** 2)
        return term1 + term2

    def initial_point(self) -> jnp.ndarray:
        key = jrandom.PRNGKey(self.seed)
        exponent = 2.0 ** (-jnp.arange(self.dim))
        optimum = 2.0 ** (exponent - 1.0)
        noise = self.sigma_init * jrandom.normal(key, (self.dim,))
        return optimum + noise
