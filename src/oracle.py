from dataclasses import dataclass
from typing import Tuple

from . import jax_config  # noqa: F401
import jax
import jax.numpy as jnp


@dataclass
class JaxOracle:
    problem: object  # ProblemBase

    def __post_init__(self):
        # Wrap problem.f for JIT and autograd
        def f_wrapped(x):
            return self.problem.f(x)

        # JIT-compiled functions
        self._f_jit = jax.jit(f_wrapped)
        self._grad_jit = jax.jit(jax.grad(f_wrapped))
        self._f_and_grad_jit = jax.jit(jax.value_and_grad(f_wrapped))

        # Counters
        self.nfev = 0
        self.ngev = 0
        self.nfgev = 0

    # ---------- Counter utilities ----------
    def reset_counts(self):
        self.nfev = 0
        self.ngev = 0
        self.nfgev = 0

    # ---------- Core oracle API ----------
    def f(self, x: jnp.ndarray) -> jnp.ndarray:
        self.nfev += 1
        return self._f_jit(x)

    def grad(self, x: jnp.ndarray) -> jnp.ndarray:
        self.ngev += 1
        return self._grad_jit(x)

    def f_and_grad(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        self.nfgev += 1
        return self._f_and_grad_jit(x)

    # ---------- Logging-only API (counter OFF) ----------
    def log_values(self, x: jnp.ndarray) -> Tuple[float, float]:
        f_val, grad = self._f_and_grad_jit(x)
        gnorm = jnp.linalg.norm(grad)
        return float(f_val), float(gnorm)
