from pathlib import Path
import math
import csv
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Sequence, Dict, Any

X_LABELS = {
    "oc": r"\# Oracle calls",
    "time": "Elapsed time (s)",
}

Y_LABELS = {
    "f": r"$f(x)$",
    "grad_norm": r"$\|\nabla f(x)\|$",
}

XKEY_SUBDIRS = {
    "oc": "oc",
    "time": "time",
}


def setup_plot_style() -> None:
    """Apply a consistent plotting style across scripts."""
    sns.set_theme(context="notebook")
    plt.rcParams["text.usetex"] = True


def load_records_from_csv(csv_path: Path):
    """Return the list of iteration records stored in a CSV file."""
    records = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            nfev = int(row["nfev"])
            ngev = int(row["ngev"])
            nfgev = int(row["nfgev"])
            records.append(
                {
                    "oc": nfev + ngev + nfgev,
                    "time": float(row["elapsed"]),
                    "f": float(row["f_value"]),
                    "grad_norm": float(row["grad_norm"]),
                }
            )
    return records


def create_legend(
    handles: Sequence[object],
    labels: Sequence[str],
    out_dir: Path,
    figsize: tuple[float, float],
    legend_kwargs: dict,
    filename: str = "legend.pdf",
) -> None:
    """Save legend-only figure."""
    if not handles or not labels:
        return
    legend_fig = plt.figure(figsize=figsize)
    legend_fig.legend(handles, labels, ncol=len(labels), frameon=True, loc="center", **legend_kwargs)
    legend_fig.gca().axis("off")
    legend_path = out_dir / filename
    legend_fig.savefig(legend_path, bbox_inches="tight", pad_inches=0)
    plt.close(legend_fig)


def parse_algo_params(algo_name: str) -> Dict[str, Any]:
    """Parse parameter tokens from algo name of form 'name@k1=v1,k2=v2'."""
    params: Dict[str, Any] = {}
    if "@" not in algo_name:
        return params
    param_str = algo_name.split("@", 1)[1]
    for token in param_str.split(","):
        if "=" not in token:
            continue
        key, val = token.split("=", 1)
        lower = val.lower()
        if lower in ("true", "false"):
            params[key] = lower == "true"
            continue
        try:
            params[key] = float(val)
        except ValueError:
            params[key] = val
    return params


def format_power_of_ten(value: float, threshold: float = 1e-9) -> str:
    """Return '10^{k}' if value is an exact power of ten, else '{value:g}'."""
    if value > 0:
        exp = math.log10(value)
        if abs(exp - round(exp)) < threshold:
            return rf"10^{int(round(exp))}"
    return f"{value:g}"
