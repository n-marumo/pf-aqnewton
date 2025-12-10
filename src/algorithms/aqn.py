from dataclasses import dataclass
from itertools import count
from typing import Iterator, Tuple, Dict, Any
import math

from .. import jax_config  # noqa: F401
import jax.numpy as jnp
import scipy.optimize as opt

from ..oracle import JaxOracle
from .base import AlgorithmBase


# Root-finder helpers to stop once |phi| is below tolerance.
class ValueToleranceReached(Exception):
    def __init__(self, x):
        self.x = x


def stop_on_value_tolerance(f, delta):
    def wrapped(x):
        v = f(x)
        if abs(v) <= delta:
            raise ValueToleranceReached(x)
        return v

    return wrapped


@dataclass
class AcceleratedQuasiNewton(AlgorithmBase):
    c_kappa: float = 1e1
    c_sigma: float = 1e5
    c_delta: float = 1e0
    use_xbar: bool = True
    INFO_KEYS = (
        "t",
        "k",
        "kappa",
        "sigma",
        "delta",
        "theta",
        "mu",
        "rs_ratio",
        "B_normfro",
        "B_normop",
    )

    def build_info(self) -> Dict[str, Any]:
        return {key: getattr(self, key, None) for key in self.INFO_KEYS}

    def _solve_subproblem(self, g: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
        lam, U = jnp.linalg.eigh(B)
        g_tilde = U.T @ g

        def phi(mu: float) -> float:
            return self.sigma * jnp.sum((g_tilde / (lam + mu)) ** 2) - mu

        # Find interval [l, r] containing root of phi
        c = self.sigma * g_tilde[0] ** 2
        if lam[0] >= 0:
            l = c / (lam[0] + c ** (1 / 3)) ** 2
        else:
            l = (c / (c ** (1 / 3) - lam[0])) ** (1 / 2) - lam[0]
        r = l + phi(l)

        # Solve phi(mu) = 0 with early exit on tolerance
        wrapped = stop_on_value_tolerance(phi, self.delta)
        try:
            mu = opt.brentq(wrapped, l, r)
        except ValueToleranceReached as e:
            mu = e.x

        # compute s
        s_tilde = -g_tilde / (lam + mu)
        s = U @ s_tilde
        self.mu = mu
        return s

    def iterate(self, oracle: JaxOracle, x0: jnp.ndarray) -> Iterator[Tuple[int, jnp.ndarray, Dict[str, Any]]]:
        # Initialize x and B
        d = x0.shape[0]
        x = x0
        B = jnp.zeros((d, d))
        self.B_normfro = 0.0
        self.B_normop = 0.0
        it = 0
        yield it, x, self.build_info()
        g = oracle.grad(x)

        for t in count(1):
            self.t = t

            # Update parameters
            self.kappa = self.c_kappa * t ** (1 / 12)
            self.sigma = self.c_sigma * t ** (2 / 3)
            self.delta = self.c_delta * t ** (-5 / 24)
            self.K = math.floor(self.kappa)
            self.theta = d / self.kappa**5

            if self.kappa <= d ** (1 / 5):
                return

            g_sum = jnp.zeros_like(x0)
            x_sum = jnp.zeros_like(x0)

            for k in range(self.K):
                self.k = k
                it += 1
                x_sum += (2 * k + 1) * x
                g_sum += (2 * k + 1) * g
                s = self._solve_subproblem(g + g_sum / (k + 1), B)
                x = x + s
                # yield it, x, self.build_info()

                g_new = oracle.grad(x)
                y = g_new - g
                g = g_new
                r = y - B @ s

                s_norm = jnp.linalg.norm(s)
                r_norm = jnp.linalg.norm(r)
                self.rs_ratio = float(r_norm / s_norm)

                # Scaled PSB update for B
                s /= s_norm
                r /= s_norm
                rs = jnp.outer(r, s)
                B += (rs + rs.T) - jnp.dot(r, s) * jnp.outer(s, s)
                B *= (1 - self.theta) / (1 + self.theta)
                self.B_normfro = float(jnp.linalg.norm(B, ord="fro"))
                self.B_normop = float(jnp.linalg.norm(B, ord=2))

            # Compute x_bar
            x_sum += self.K * x
            x_bar = x_sum / (self.K * (self.K + 1))
            yield it, x_bar, self.build_info()

            if self.use_xbar:
                f_xbar, g_xbar = oracle.f_and_grad(x_bar)
                if f_xbar < oracle.f(x):
                    x = x_bar
                    g = g_xbar
