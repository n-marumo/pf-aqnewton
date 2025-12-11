# Parameter-Free Accelerated Quasi-Newton Experiments

This repository provides the implementation for the numerical experiments presented in our paper, *Parameter-Free Accelerated Quasi-Newton Method for Nonconvex Optimization* (arXiv: [2512.09439](https://arxiv.org/abs/2512.09439)).


## Setup
- Python 3.13.9 environment (e.g., `python -m venv .venv && source .venv/bin/activate`)
- `pip install -r requirements.txt`
- LaTeX toolchain recommended for Matplotlib `text.usetex` in the plotting scripts.

## Run experiments
- `python -m src.runner.run`
- CSV logs are written to `results/raw/<problem>/<algo>.csv` with columns: `iter, f_value, grad_norm, elapsed, nfev, ngev, nfgev, algo_info`.

## Visualize
- `python -m src.visualization.compare_params`
- `python -m src.visualization.compare_algorithms`
- Figures are saved under `results/figures/compare_params` and `results/figures/compare_algorithms`.

## Layout
- `src/algorithms` — AQN, BFGS, DFP, GD implementations
- `src/problems` — benchmark definitions and initial points
- `src/runner` — experiment driver and logging utilities
- `src/visualization` — plotting scripts and config
