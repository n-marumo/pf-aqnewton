from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from src.visualization.utils import (  # type: ignore
    X_LABELS,
    Y_LABELS,
    setup_plot_style,
    load_records_from_csv,
    create_legend,
    parse_algo_params,
)
from src.visualization.config import (
    RAW_DIR,
    FIG_DIR_ALGOS,
    FIG_SIZE_ALGOS,
    LEGEND_FIGSIZE,
    LEGEND_KWARGS,
    XKEYS,
    YKEYS,
    LINE_STYLES,
    LINEWIDTH_AQN,
    LINEWIDTH_DEFAULT,
    COLOR_AQN,
    AQN_ALLOWED_PARAMS,
)


setup_plot_style()

# Custom legend labels per algorithm (full algo string or base name)
LEGEND_LABELS: dict[str, str] = {
    "aqn": r"\textbf{Proposed}",
    "bfgs": "BFGS",
    "dfp": "DFP",
    "gd": "GD",
}

PLOT_ORDER = ["aqn", "bfgs", "dfp", "gd"]  # Edit to change which algorithms are plotted and their order.
PLOT_ORDER_INDEX = {name: idx for idx, name in enumerate(PLOT_ORDER)}

RecordsByAlgo = Dict[str, List[Dict[str, float]]]
AllRecords = Dict[str, RecordsByAlgo]


def _algo_base(algo_name: str) -> str:
    """Return lowercase algorithm base name before any @params suffix."""
    return algo_name.partition("@")[0].lower()


def load_all_records(raw_dir: Path) -> AllRecords:
    data: AllRecords = {}
    for problem_dir in raw_dir.iterdir():
        if not problem_dir.is_dir():
            continue
        problem = problem_dir.name
        for csv_file in problem_dir.glob("*.csv"):
            algo = csv_file.stem
            data.setdefault(problem, {}).setdefault(algo, []).extend(load_records_from_csv(csv_file))
    return data


def short_label(algo_name: str) -> str:
    """Return a concise legend label."""
    base = _algo_base(algo_name)
    if base in LEGEND_LABELS:
        return LEGEND_LABELS[base]
    if "@method=" in algo_name:
        return algo_name.split("@method=", 1)[1]
    return base


def _is_aqn_allowed(problem: str, algo_name: str) -> bool:
    allowed = AQN_ALLOWED_PARAMS.get(problem)
    if not allowed:
        return True
    params = parse_algo_params(algo_name)
    try:
        combo = (
            float(params["c_kappa"]),
            float(params["c_sigma"]),
            float(params["c_delta"]),
            bool(params.get("use_xbar", False)),
        )
    except KeyError:
        return False
    return combo in allowed


def plot_xy(
    problem: str, algo_data: RecordsByAlgo, out_dir: Path, xkey: str, ykey: str
) -> Tuple[list[object], list[str]]:
    plt.figure(figsize=FIG_SIZE_ALGOS)
    ordered = sorted(algo_data.items(), key=lambda item: PLOT_ORDER_INDEX[_algo_base(item[0])])
    for idx, (algo, records) in enumerate(ordered):
        xs = [r[xkey] for r in records]
        ys = [r[ykey] for r in records]
        base = _algo_base(algo)
        linewidth = LINEWIDTH_AQN if base == "aqn" else LINEWIDTH_DEFAULT
        linestyle = LINE_STYLES[idx % len(LINE_STYLES)]
        color = COLOR_AQN if base == "aqn" else None
        plt.plot(xs, ys, label=short_label(algo), linewidth=linewidth, linestyle=linestyle, color=color)

    plt.xlabel(X_LABELS[xkey])
    plt.ylabel(Y_LABELS[ykey])
    plt.yscale("log")
    legend = plt.legend(loc="best", **LEGEND_KWARGS)
    handles, labels = legend.legend_handles, [text.get_text() for text in legend.get_texts()]
    legend.remove()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{problem}_{xkey}_{ykey}.pdf"
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    return handles, labels


def save_legend(out_dir: Path, handles, labels) -> None:
    create_legend(handles, labels, out_dir, figsize=LEGEND_FIGSIZE, legend_kwargs=LEGEND_KWARGS)


def main() -> None:
    fig_dir = FIG_DIR_ALGOS
    fig_dir.mkdir(parents=True, exist_ok=True)

    data = load_all_records(RAW_DIR)

    legend_handles: list[object] | None = None
    legend_labels: list[str] | None = None
    for problem, algo_data in data.items():
        filtered: RecordsByAlgo = {}
        for algo, records in algo_data.items():
            base = _algo_base(algo)
            if base not in PLOT_ORDER:
                continue
            if base == "aqn" and not _is_aqn_allowed(problem, algo):
                continue
            filtered[algo] = records
        if not filtered:
            continue
        for xkey in XKEYS:
            for ykey in YKEYS:
                handles, labels = plot_xy(problem, filtered, fig_dir, xkey, ykey)
                if legend_handles is None and handles and labels:
                    legend_handles, legend_labels = handles, labels
        print(f"[OK] {problem} plots saved.")
    if legend_handles and legend_labels:
        save_legend(fig_dir, legend_handles, legend_labels)


if __name__ == "__main__":
    main()
