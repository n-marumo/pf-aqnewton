from pathlib import Path

# Common paths
RAW_DIR = Path("results/raw")
FIG_DIR_ALGOS = Path("results/figures") / "compare_algorithms"
FIG_DIR_PARAMS = Path("results/figures") / "compare_params"

# Plot styling
FIG_SIZE_ALGOS = (5.0, 3.2)
FIG_SIZE_PARAMS = (4.4, 2.8)
LEGEND_FIGSIZE = (6, 0.35)
LEGEND_KWARGS = {"fontsize": "small"}
XKEYS = ("oc",)
YKEYS = ("f", "grad_norm")
LINEWIDTH_AQN = 3
LINEWIDTH_DEFAULT = 2
LINE_STYLES = ["-", "--", "-.", ":"]
COLOR_AQN = "black"

# AQN settings
AQN_SIGMA_VALUES = (1e3, 1e4, 1e5, 1e6)
AQN_KAPPA_VALUES = (1e1, 3e1, 1e2)
AQN_DELTA_VALUES = (1e-5,)

# Allowed AQN parameter tuples per problem (ckappa, csigma, cdelta, use_xbar)
AQN_ALLOWED_PARAMS: dict[str, set[tuple[float, float, float, bool]]] = {
    "dixonprice@dim=100,seed=0,sigma_init=1": {(1e1, 1e4, 1e-5, False)},
    "powell@dim=100,seed=0,sigma_init=1": {(3e1, 1e6, 1e-5, False)},
    "qing@dim=100,seed=0,sigma_init=1": {(3e1, 1e4, 1e-5, False)},
    "rosenbrock@a=1,b=100,dim=100,seed=0,sigma_init=1": {(1e1, 1e5, 1e-5, False)},
}
