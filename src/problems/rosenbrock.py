from .. import jax_config  # noqa: F401
import jax.numpy as jnp
import jax.random as jrandom
from .base import ProblemBase


class Rosenbrock(ProblemBase):
    def __init__(self, dim: int, a: float = 1.0, b: float = 100.0, seed: int = 0, sigma_init: float = 1.0):
        assert dim >= 2
        self.dim = dim
        self.a = a
        self.b = b
        self.seed = seed
        self.sigma_init = sigma_init

    def f(self, x: jnp.ndarray) -> jnp.ndarray:
        x0 = x[:-1]
        x1 = x[1:]
        return jnp.sum(self.b * (x1 - x0**2) ** 2 + (self.a - x0) ** 2)

    def initial_point(self) -> jnp.ndarray:
        key = jrandom.PRNGKey(self.seed)
        optimum = jnp.ones(self.dim)
        noise = self.sigma_init * jrandom.normal(key, (self.dim,))
        return optimum + noise
