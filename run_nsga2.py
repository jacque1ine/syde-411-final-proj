"""Run binary NSGA-II on the frozen Toronto CSV. No station-capacity model.
Run python3 run_nsga2.py. Results are saved under results/.
"""

from pathlib import Path
import json
import time
import platform
import numpy as np
import pandas as pd
import pymoo
import scipy
from scipy.spatial.distance import cdist
from scipy.optimize import milp, Bounds, LinearConstraint
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
active_cells = pd.read_csv(PROJECT_DIR / "toronto_225_cells.csv")
active_cells = active_cells[active_cells.active_demand].reset_index(drop=True)
xy = active_cells[["easting_epsg26917_m", "northing_epsg26917_m"]].to_numpy()
distance_km = cdist(xy, xy) / 1000
coverage = distance_km <= 4 + 1e-9
risk_weights = active_cells.risk_per_1000_over_6_years.to_numpy()
cell_count = len(active_cells)
scale = np.array([cell_count, 4 * risk_weights.sum() + 1])
hv = HV(ref_point=np.array([1.01, 1.01]))


class FireProblem(Problem):
    def __init__(self):
        super().__init__(
            n_var=cell_count, n_obj=2, n_ieq_constr=1, xl=0, xu=1, vtype=bool
        )

    def _evaluate(self, X, out, *args, **kwargs):
        objectives = []
        violations = []
        for layout in X:
            dist = (
                distance_km[:, layout.astype(bool)].min(axis=1)
                if layout.any()
                else np.full(cell_count, 1e6)
            )
            objectives.append([layout.sum(), risk_weights @ dist])
            violations.append([np.count_nonzero(dist > 4 + 1e-9)])
        out["F"] = np.asarray(objectives)
        out["G"] = np.asarray(violations)


def initial_layouts(seed, size):
    rng = np.random.default_rng(seed)
    X = []
    for k in range(size):
        layout = np.zeros(cell_count, dtype=bool)
        layout[rng.choice(cell_count, size=int(rng.integers(0, 35)), replace=False)] = (
            True
        )
        uncovered = ~coverage[:, layout].any(axis=1)
        while uncovered.any():
            scores = coverage[uncovered].sum(axis=0).astype(float)
            scores[layout] = -1
            candidates = np.flatnonzero(scores >= max(1, scores.max() * 0.8))
            layout[rng.choice(candidates)] = True
            uncovered = ~coverage[:, layout].any(axis=1)
        X.append(layout)
    return np.array(X)


class History(Callback):
    def __init__(self):
        super().__init__()
        self.rows = []

    def notify(self, a):
        feasible = a.pop.get("G").ravel() <= 0
        F = a.pop.get("F")[feasible]
        self.rows.append(
            {
                "generation": a.n_gen,
                "evaluations": a.evaluator.n_eval,
                "hypervolume": float(hv(F / scale)) if len(F) else 0,
                "min_stations": float(F[:, 0].min()) if len(F) else None,
                "feasible_fraction": float(feasible.mean()),
            }
        )


pooled_objectives = []
pooled_layouts = []
runs = []
for seed in [411, 412, 413, 414, 415]:
    history = History()
    t = time.perf_counter()
    result = minimize(
        FireProblem(),
        NSGA2(
            pop_size=120,
            sampling=initial_layouts(seed, 120),
            crossover=TwoPointCrossover(prob=0.9),
            mutation=BitflipMutation(prob=1.0, prob_var=1 / cell_count),
            eliminate_duplicates=True,
        ),
        ("n_gen", 200),
        seed=seed,
        callback=history,
        verbose=False,
    )
    elapsed = time.perf_counter() - t
    F = np.atleast_2d(result.F)
    X = np.atleast_2d(result.X)
    for layout, obj in zip(X, F):
        nearest_km = distance_km[:, layout].min(axis=1)
        assert nearest_km.max() <= 4 + 1e-9
        assert np.allclose(obj, [layout.sum(), risk_weights @ nearest_km])
    pooled_objectives.extend(F)
    pooled_layouts.extend(X)
    pd.DataFrame(history.rows).to_csv(RESULTS_DIR / f"history_{seed}.csv", index=False)
    runs.append(
        {
            "seed": seed,
            "seconds": elapsed,
            "evaluations": int(result.algorithm.evaluator.n_eval),
            "min_stations": int(F[:, 0].min()),
            "max_stations_on_front": int(F[:, 0].max()),
            "final_hypervolume": float(hv(F / scale)),
            "initial_hypervolume": history.rows[0]["hypervolume"],
        }
    )
    print(runs[-1], flush=True)
F = np.array(pooled_objectives)
X = np.array(pooled_layouts)
front_indices = NonDominatedSorting().do(F, only_non_dominated_front=True)
F = F[front_indices]
X = X[front_indices]
# One best layout per station count; ties resolved by the first saved realization.
order = np.lexsort((F[:, 1], F[:, 0]))
idx = []
seen = set()
for k in order:
    if int(F[k, 0]) not in seen:
        idx.append(k)
        seen.add(int(F[k, 0]))
F = F[idx]
X = X[idx]
pd.DataFrame(F, columns=["stations", "risk_weighted_distance"]).to_csv(
    RESULTS_DIR / "pooled_front.csv", index=False
)
np.savez_compressed(
    RESULTS_DIR / "layouts.npz",
    X=X,
    F=F,
    cell_ids=active_cells.cell_id.to_numpy(dtype=str),
)
# Independent exact set-cover benchmark, 15-second cap. Bound is saved even if not optimal.
t = time.perf_counter()
exact = milp(
    np.ones(cell_count),
    integrality=np.ones(cell_count),
    bounds=Bounds(0, 1),
    constraints=LinearConstraint(
        coverage.astype(float), np.ones(cell_count), np.full(cell_count, np.inf)
    ),
    options={"time_limit": 15},
)
if exact.x is not None:
    z_exact = exact.x > 0.5
    assert np.all(coverage[:, z_exact].any(axis=1))
    pd.DataFrame({"cell_id": active_cells.cell_id, "selected": z_exact}).to_csv(
        RESULTS_DIR / "exact_set_cover_layout.csv", index=False
    )
benchmark = {
    "status": int(exact.status),
    "message": exact.message,
    "station_count": float(exact.fun) if exact.fun is not None else None,
    "dual_bound": (
        float(exact.mip_dual_bound)
        if getattr(exact, "mip_dual_bound", None) is not None
        else None
    ),
    "gap": (
        float(exact.mip_gap) if getattr(exact, "mip_gap", None) is not None else None
    ),
    "seconds": time.perf_counter() - t,
}
# Select a transparent presentation example: lowest station count found, best pooled distance at that count.
layout = X[0]
dist = distance_km[:, layout].min(axis=1)
grid = pd.read_csv(PROJECT_DIR / "toronto_225_cells.csv")
grid["nsga2_station_selected"] = grid.cell_id.isin(active_cells.cell_id[layout])
grid["nearest_nsga2_station_km"] = np.nan
grid.loc[grid.active_demand, "nearest_nsga2_station_km"] = dist
grid.to_csv(RESULTS_DIR / "selected_layout_225_cells.csv", index=False)
summary = {
    "runs": runs,
    "configuration": {
        "population": 120,
        "generations": 200,
        "seeds": [411, 412, 413, 414, 415],
        "crossover": "two-point, probability 0.9",
        "mutation": "bit flip, individual probability 1, bit probability 1/210",
        "constraint": "uncovered cell count <= 0; feasibility-first",
        "initialization": "random 0-34 sites, greedy randomized coverage completion",
        "hypervolume": "F normalized by [210, 4*sum(w)+1], reference [1.01,1.01]",
    },
    "exact_set_cover": benchmark,
    "selected": {
        "stations": int(layout.sum()),
        "risk_weighted_distance": float(risk_weights @ dist),
        "max_distance_km": float(dist.max()),
        "coverage_percent": 100,
        "cell_ids": active_cells.cell_id[layout].tolist(),
    },
    "pooled_points": len(F),
    "sum_risk_weights": float(risk_weights.sum()),
    "environment": {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pymoo": pymoo.__version__,
        "scipy": scipy.__version__,
        "machine": platform.machine(),
        "processor": platform.processor(),
    },
}
(RESULTS_DIR / "experiment_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary["exact_set_cover"]), flush=True)
