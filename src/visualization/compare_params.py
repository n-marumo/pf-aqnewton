from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

from src.visualization.utils import (  # type: ignore
    X_LABELS,
    Y_LABELS,
    setup_plot_style,
    load_records_from_csv,
    create_legend,
    parse_algo_params,
    format_power_of_ten,
)
from src.visualization.config import (
    RAW_DIR,
    FIG_DIR_PARAMS,
    FIG_SIZE_PARAMS,
    LEGEND_FIGSIZE,
    LEGEND_KWARGS,
    LINEWIDTH_DEFAULT,
    AQN_SIGMA_VALUES,
    AQN_KAPPA_VALUES,
    AQN_DELTA_VALUES,
    XKEYS,
    YKEYS,
)


setup_plot_style()

FIG_DIR = FIG_DIR_PARAMS


def load_parameter_records(raw_dir: Path):
    data = {}
    aqn_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for problem_dir in raw_dir.iterdir():
        if not problem_dir.is_dir():
            continue
        problem = problem_dir.name
        for csv_file in problem_dir.glob("*.csv"):
            algo = csv_file.stem
            base_algo = algo.split("@")[0]
            key = (problem, base_algo)
            records = load_records_from_csv(csv_file)
            data.setdefault(key, {}).setdefault(algo, []).extend(records)

            if base_algo.lower() == "aqn":
                params = parse_algo_params(algo)
                c_kappa = float(params.get("c_kappa", 0.0))
                c_delta = float(params.get("c_delta", 0.0))
                c_sigma = float(params.get("c_sigma", 0.0))
                use_xbar = bool(params.get("use_xbar", False))
                aqn_data[problem][use_xbar][(c_kappa, c_delta)][c_sigma].extend(records)
    return data, aqn_data


def plot_algo_params(problem: str, base_algo: str, algo_data: dict, out_dir: Path, xkey: str, ykey: str):
    plt.figure(figsize=FIG_SIZE_PARAMS)
    linestyles = ["-", "--", "-.", ":"]
    for idx, (algo_name, records) in enumerate(sorted(algo_data.items())):
        label = algo_name
        if "@" in algo_name:
            label = algo_name.split("@", 1)[1].replace(",", "\n")
        xs = [r[xkey] for r in records]
        ys = [r[ykey] for r in records]
        linestyle = linestyles[idx % len(linestyles)]
        plt.plot(xs, ys, label=label, linestyle=linestyle, linewidth=LINEWIDTH_DEFAULT)

    plt.xlabel(X_LABELS[xkey])
    plt.ylabel(Y_LABELS[ykey])
    plt.yscale("log")
    legend = plt.legend(loc="best", **LEGEND_KWARGS)
    handles, labels = legend.legend_handles, [text.get_text() for text in legend.get_texts()]
    legend.remove()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{problem}_{base_algo}_{xkey}_{ykey}_params.pdf"
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    return handles, labels


def plot_aqn_sigma(
    problem: str, c_kappa: float, c_delta: float, sigma_map: dict, out_dir: Path, xkey: str, ykey: str, use_xbar: bool
):
    plt.figure(figsize=FIG_SIZE_PARAMS)
    linestyles = ["-", "--", "-.", ":"]
    sigma_items = sorted((sigma, records) for sigma, records in sigma_map.items() if sigma in AQN_SIGMA_VALUES)
    if not sigma_items:
        plt.close()
        return
    ax = plt.gca()
    for idx, (sigma, records) in enumerate(sigma_items):
        xs = [r[xkey] for r in records]
        ys = [r[ykey] for r in records]
        linestyle = linestyles[idx % len(linestyles)]
        plt.plot(
            xs,
            ys,
            label=rf"$c_\sigma = {format_power_of_ten(sigma)}$",
            linestyle=linestyle,
            linewidth=LINEWIDTH_DEFAULT,
        )

    plt.xlabel(X_LABELS[xkey])
    plt.ylabel(Y_LABELS[ykey])
    plt.yscale("log")
    ax.set_title(rf"$c_\kappa = {c_kappa:g}$", fontsize="large")
    legend = plt.legend(loc="best", **LEGEND_KWARGS)
    handles, labels = legend.legend_handles, [text.get_text() for text in legend.get_texts()]
    legend.remove()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        out_dir
        / f"{problem}_aqn_ckappa={c_kappa:g}_cdelta={c_delta:g}_usexbar={str(use_xbar).lower()}_{xkey}_{ykey}.pdf"
    )
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    return handles, labels


def save_legend(out_dir: Path, handles, labels) -> None:
    create_legend(handles, labels, out_dir, figsize=LEGEND_FIGSIZE, legend_kwargs=LEGEND_KWARGS)


def main() -> None:
    fig_dir = FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)

    data, aqn_data = load_parameter_records(RAW_DIR)

    legend_saved = False
    legend_handles = None
    legend_labels = None

    for (problem, base_algo), algo_data in data.items():
        if base_algo.lower() == "aqn" or len(algo_data) <= 1:
            continue
        for xkey in XKEYS:
            for ykey in YKEYS:
                handles, labels = plot_algo_params(problem, base_algo, algo_data, fig_dir, xkey, ykey)
                if handles and labels and not legend_saved:
                    legend_handles, legend_labels = handles, labels
        print(f"[OK] {problem} | {base_algo} parameter plots saved.")

    for problem, use_xbar_map in aqn_data.items():
        for use_xbar, ck_map in use_xbar_map.items():
            for c_kappa in AQN_KAPPA_VALUES:
                for c_delta in AQN_DELTA_VALUES:
                    sigma_map = ck_map.get((c_kappa, c_delta))
                    if not sigma_map:
                        continue
                    for xkey in XKEYS:
                        for ykey in YKEYS:
                            handles, labels = plot_aqn_sigma(
                                problem, c_kappa, c_delta, sigma_map, fig_dir, xkey, ykey, use_xbar
                            )
                            if handles and labels and not legend_saved:
                                legend_handles, legend_labels = handles, labels
                    print(
                        f"[OK] {problem} | AQN c_kappa={c_kappa:g}, c_delta={c_delta:g}, use_xbar={use_xbar} sigma sweep saved."
                    )

    if legend_handles and legend_labels:
        save_legend(fig_dir, legend_handles, legend_labels)


if __name__ == "__main__":
    main()
