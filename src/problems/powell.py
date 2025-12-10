from .. import jax_config  # noqa: F401
import jax.numpy as jnp
import jax.random as jrandom

from .base import ProblemBase


class Powell(ProblemBase):
    def __init__(self, dim: int, seed: int = 0, sigma_init: float = 1.0):
        assert dim >= 4, "Powell function requires dimension >= 4"
        assert dim % 4 == 0, "Powell function dimension must be divisible by 4"
        self.dim = dim
        self.seed = seed
        self.sigma_init = sigma_init

    def f(self, x: jnp.ndarray) -> jnp.ndarray:
        groups = self.dim // 4
        usable = groups * 4
        x1 = x[:usable:4]
        x2 = x[1:usable:4]
        x3 = x[2:usable:4]
        x4 = x[3:usable:4]

        term1 = (x1 + 10.0 * x2) ** 2
        term2 = 5.0 * (x3 - x4) ** 2
        term3 = (x2 - 2.0 * x3) ** 4
        term4 = 10.0 * (x1 - x4) ** 4

        return jnp.sum(term1 + term2 + term3 + term4)

    def initial_point(self) -> jnp.ndarray:
        key = jrandom.PRNGKey(self.seed)
        optimum = jnp.zeros(self.dim)
        noise = self.sigma_init * jrandom.normal(key, (self.dim,))
        return optimum + noise
