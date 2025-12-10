# src/problems/base.py

from abc import ABC, abstractmethod

from .. import jax_config  # noqa: F401
import jax.numpy as jnp


class ProblemBase(ABC):
    @abstractmethod
    def f(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return the objective value."""
        ...

    @abstractmethod
    def initial_point(self) -> jnp.ndarray:
        """Return an initial point (problem parameters include randomness)."""
        ...
