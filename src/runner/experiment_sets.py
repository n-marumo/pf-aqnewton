from typing import Callable, List, Tuple

from ..algorithms.gd import GradientDescent
from ..algorithms.bfgs import BFGS
from ..algorithms.dfp import DFP
from ..algorithms.aqn import AcceleratedQuasiNewton

from ..problems.base import ProblemBase
from ..problems.rosenbrock import Rosenbrock
from ..problems.dixon_price import DixonPrice
from ..problems.powell import Powell
from ..problems.qing import Qing


def default_iterative_algorithms() -> List[Callable[[], object]]:
    factories: List[Callable[[], object]] = [
        lambda: BFGS(),
        lambda: DFP(),
        lambda: GradientDescent(),
    ]
    for c_kappa in (1e1, 3e1, 1e2):
        for c_sigma in (1e3, 1e4, 1e5, 1e6):
            for c_delta in (1e-5,):
                factories.append(
                    lambda c_kappa=c_kappa, c_sigma=c_sigma, c_delta=c_delta: AcceleratedQuasiNewton(
                        c_kappa=c_kappa,
                        c_sigma=c_sigma,
                        c_delta=c_delta,
                        use_xbar=False,
                    )
                )
    return factories


def default_problems() -> List[Tuple[ProblemBase, int]]:
    """Return list of (problem, max_oc) tuples."""
    return [
        (DixonPrice(dim=100, seed=0, sigma_init=1), 1000),
        (Powell(dim=100, seed=0, sigma_init=1), 3000),
        (Qing(dim=100, seed=0, sigma_init=1), 3000),
        (Rosenbrock(dim=100, seed=0, sigma_init=1), 2000),
    ]
