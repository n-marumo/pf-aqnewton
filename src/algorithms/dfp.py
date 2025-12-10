from dataclasses import dataclass
from itertools import count
from typing import Any, Dict, Iterator, Tuple

from .. import jax_config  # noqa: F401
import jax.numpy as jnp

from ..oracle import JaxOracle
from .base import AlgorithmBase


@dataclass
class DFP(AlgorithmBase):
    init_step: float = 1e0
    shrink: float = 0.5
    c1: float = 1e-4

    def build_info(self) -> Dict[str, Any]:
        return {
            "step_size": getattr(self, "step_size", None),
        }

    def _armijo_backtracking(
        self,
        oracle: JaxOracle,
        x: jnp.ndarray,
        f: float,
        grad: jnp.ndarray,
        p: jnp.ndarray,
        start_step: float,
    ) -> Tuple[float, float]:
        step = start_step
        directional_derivative = jnp.dot(grad, p)

        while True:
            x_trial = x + step * p
            f_trial = oracle.f(x_trial)
            if f_trial <= f + self.c1 * step * directional_derivative:
                return step, f_trial
            step *= self.shrink

    def iterate(self, oracle: JaxOracle, x0: jnp.ndarray) -> Iterator[Tuple[int, jnp.ndarray, Dict[str, Any]]]:
        x = x0
        dim = x0.shape[0]
        H = jnp.eye(dim)

        f, grad = oracle.f_and_grad(x)
        self.step_size = None
        prev_step = None
        yield 0, x, self.build_info()

        for it in count(1):
            p = -H @ grad

            start_step = prev_step / self.shrink if prev_step is not None else self.init_step
            step, f_new = self._armijo_backtracking(oracle, x, f, grad, p, start_step)
            self.step_size = step
            prev_step = step

            s = step * p
            x_new = x + s
            grad_new = oracle.grad(x_new)
            y = grad_new - grad

            sy = jnp.dot(s, y)
            if sy > 0:
                Hy = H @ y
                yHy = jnp.dot(y, Hy)
                H = H + jnp.outer(s, s) / sy - jnp.outer(Hy, Hy) / yHy

            x, grad, f = x_new, grad_new, f_new
            yield it, x, self.build_info()
