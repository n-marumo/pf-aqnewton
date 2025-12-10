import csv
from pathlib import Path
from itertools import islice
from typing import Any, Dict

from ..problems.base import ProblemBase
from .experiment_sets import default_iterative_algorithms, default_problems

from ..algorithms.base import AlgorithmBase
from ..oracle import JaxOracle
from .utils import F_THRESHOLD_MULTIPLIER, LOG_COLUMNS, ensure_out_dir, format_component, log_iterations


def run_experiment():
    out_dir = ensure_out_dir()
    for prob_entry in default_problems():
        prob, max_oc = prob_entry
        for algo_factory in default_iterative_algorithms():
            run_one_setting(prob, algo_factory(), out_dir, max_oc=max_oc)


def run_one_setting(
    problem: ProblemBase,
    algo: AlgorithmBase,
    out_dir: Path,
    max_oc: int,
):
    prob_name = format_component(problem)
    algo_name = format_component(algo)
    prob_dir = out_dir / prob_name
    prob_dir.mkdir(parents=True, exist_ok=True)
    out_path = prob_dir / f"{algo_name}.csv"

    oracle = JaxOracle(problem)
    x0 = problem.initial_point()
    f_initial = float(oracle.f(x0))

    warmup_algorithm(algo, oracle, x0)
    iterator = algo.iterate(oracle, x0)

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(LOG_COLUMNS)

        f_threshold = F_THRESHOLD_MULTIPLIER * f_initial
        log_iterations(iterator, writer, oracle, f_threshold=f_threshold, max_oc=max_oc)

    print(f"Saved: {out_path}")


def warmup_algorithm(algo: AlgorithmBase, oracle: JaxOracle, x0, steps: int = 2) -> None:
    """Run a few iterations to compile JIT paths before timing."""
    iterator = algo.iterate(oracle, x0)
    for _, warmup_x, _ in islice(iterator, steps):
        oracle.log_values(warmup_x)
    oracle.reset_counts()


if __name__ == "__main__":
    run_experiment()
