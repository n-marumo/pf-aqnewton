from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Dict, Any

from .. import jax_config  # noqa: F401
import jax.numpy as jnp


class AlgorithmBase(ABC):
    @abstractmethod
    def build_info(self) -> Dict[str, Any]:
        """Return a dict of algorithm-specific parameters."""
        ...

    @abstractmethod
    def iterate(
        self,
        oracle,
        x0: jnp.ndarray
    ) -> Iterator[Tuple[int, jnp.ndarray, Dict[str, Any]]]:
        """
        Yield (iteration_index, x, info_dict) for each iteration.
        All computations must use jnp arrays.
        """
        ...
