import time
from pathlib import Path
from typing import Any, Dict

from ..algorithms.base import AlgorithmBase
from ..oracle import JaxOracle

LOG_COLUMNS = (
    "iter",
    "f_value",
    "grad_norm",
    "elapsed",
    "nfev",
    "ngev",
    "nfgev",
    "algo_info",
)

F_THRESHOLD_MULTIPLIER = 10.0
ABSOLUTE_F_THRESHOLD = 1e-10
ABSOLUTE_GRAD_NORM_THRESHOLD = 1e-10

NAME_ABBREVIATIONS = {
    "AcceleratedQuasiNewton": "aqn",
    "GradientDescent": "gd",
    "BFGS": "bfgs",
    "DFP": "dfp",
}


def ensure_out_dir(base_path: Path | None = None) -> Path:
    out_dir = base_path or (Path(__file__).resolve().parents[2] / "results" / "raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def log_iterations(
    iterator,
    writer,
    oracle: JaxOracle,
    max_oc: int,
    f_threshold: float | None = None,
) -> None:
    """Iterate while logging metrics, stopping on thresholds."""
    elapsed = 0.0
    t0 = time.perf_counter()
    for it, x_jnp, info in iterator:
        t1 = time.perf_counter()
        elapsed += t1 - t0
        f_val, gnorm = oracle.log_values(x_jnp)
        row_data: Dict[str, Any] = {
            "iter": it,
            "f_value": f_val,
            "grad_norm": gnorm,
            "elapsed": elapsed,
            "nfev": oracle.nfev,
            "ngev": oracle.ngev,
            "nfgev": oracle.nfgev,
            "algo_info": info,
        }
        writer.writerow([row_data[key] for key in LOG_COLUMNS])
        if f_val <= ABSOLUTE_F_THRESHOLD or gnorm <= ABSOLUTE_GRAD_NORM_THRESHOLD:
            break
        if f_threshold is not None and f_val >= f_threshold:
            break
        total_calls = oracle.nfev + oracle.ngev + oracle.nfgev
        if total_calls >= max_oc:
            break
        t0 = time.perf_counter()


def format_component(obj: object) -> str:
    """Format an object identifier with scalar fields as key=value pairs."""
    base_name = obj.__class__.__name__
    base = NAME_ABBREVIATIONS.get(base_name, base_name).lower()
    params = []
    if hasattr(obj, "__dataclass_fields__"):
        iterable = ((name, getattr(obj, name)) for name in obj.__dataclass_fields__)
    else:
        iterable = vars(obj).items()
    for key, val in sorted(iterable):
        if key.startswith("_"):
            continue
        if isinstance(val, (int, float, str)):
            params.append(f"{key}={_format_value(val)}")
    label = f"{base}@{','.join(params)}" if params else base
    return label


def _format_value(val) -> str:
    if isinstance(val, float):
        return f"{val:g}"
    return str(val)
