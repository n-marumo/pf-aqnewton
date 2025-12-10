from .. import jax_config  # noqa: F401
import jax.numpy as jnp
import jax.random as jrandom

from .base import ProblemBase


class Qing(ProblemBase):
    def __init__(self, dim: int, seed: int = 0, sigma_init: float = 1.0):
        assert dim >= 1
        self.dim = dim
        self.seed = seed
        self.sigma_init = sigma_init

    def f(self, x: jnp.ndarray) -> jnp.ndarray:
        idx = jnp.arange(1, self.dim + 1)
        return jnp.sum((x[: self.dim] ** 2 - idx) ** 2)

    def initial_point(self) -> jnp.ndarray:
        key = jrandom.PRNGKey(self.seed)
        optimum = jnp.sqrt(jnp.arange(1, self.dim + 1))
        noise = self.sigma_init * jrandom.normal(key, (self.dim,))
        return optimum + noise
