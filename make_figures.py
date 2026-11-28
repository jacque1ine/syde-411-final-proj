"""Make the report figures from saved results.

Run after run_nsga2.py:
    python make_figures.py
Outputs: figures/opening_heatmaps.png (data maps), figures/pareto.png
(trade-off), figures/convergence.png (hypervolume) and slide_data.json.
"""

from pathlib import Path
import pandas as pd, numpy as np, json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = Path(__file__).resolve().parent
R = P / "results"
FIG = P / "figures"
FIG.mkdir(exist_ok=True)
f = pd.read_csv(R / "selected_layout_225_cells.csv")
a = f.active_demand.to_numpy().reshape(15, 15)
s = json.loads((R / "experiment_summary.json").read_text())
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Data maps: population, fires and risk weight for each cell.
fig, axs = plt.subplots(1, 3, figsize=(12, 4.1))
fig.subplots_adjust(left=0.05, right=0.97, bottom=0.25, top=0.86, wspace=0.25)
for ax, col, title, cmap in zip(
    axs,
    [
        "population_2021_sample_based_area_estimate",
        "recorded_fires_2018_2023",
        "risk_per_1000_over_6_years",
    ],
    ["Estimated population", "Recorded fires 2018–2023", "Risk weight"],
    ["YlGnBu", "YlOrRd", "magma_r"],
):
    z = f[col].to_numpy().reshape(15, 15)
    im = ax.imshow(
        np.ma.masked_where(~a, z),
        origin="lower",
        extent=[0, 15, 0, 15],
        cmap=cmap,
        vmin=0,
    )
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xticks([0, 5, 10, 15])
    ax.set_yticks([0, 5, 10, 15])
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
fig.text(
    0.05,
    0.09,
    "Population: estimated residents  |  Fires: six-year counts  |  Risk: six-year fires per 1,000 estimated residents",
    fontsize=9,
)
fig.text(
    0.05,
    0.025,
    "Toronto study window: 225 cells, 210 included. Blank cells: outside-city areas or a negligible boundary sliver.",
    fontsize=8,
)
fig.savefig(FIG / "opening_heatmaps.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# Trade-off: best risk-weighted distance found at each station count.
F = pd.read_csv(R / "pooled_front.csv")
fig, ax = plt.subplots(figsize=(7.5, 3.5))
ax.plot(F.stations, F.risk_weighted_distance, "o-", ms=3, color="#087e8b")
ax.scatter(
    [8],
    [s["selected"]["risk_weighted_distance"]],
    color="#d14d2e",
    s=50,
    zorder=4,
    label="Mapped 8-station example",
)
ax.set_xlabel("Number of stations")
ax.set_ylabel("Risk-weighted distance")
ax.set_title("Pooled nondominated solutions from five runs")
ax.legend(fontsize=9)
ax.grid(alpha=0.15)
fig.tight_layout()
fig.savefig(FIG / "pareto.png", dpi=200)
plt.close(fig)

# Convergence: hypervolume per generation for each seed.
fig, ax = plt.subplots(figsize=(7.5, 3.5))
for seed in [411, 412, 413, 414, 415]:
    h = pd.read_csv(R / f"history_{seed}.csv")
    ax.plot(h.generation, h.hypervolume, label=str(seed), lw=1.5)
ax.set_xlabel("Generation")
ax.set_ylabel("Normalized hypervolume")
ax.set_title("Search progress by random seed; higher is better")
ax.legend(ncol=5, fontsize=8, loc="lower right")
ax.grid(alpha=0.15)
fig.tight_layout()
fig.savefig(FIG / "convergence.png", dpi=200)
plt.close(fig)

# Data for native charts in the presentation slides.
(P / "slide_data.json").write_text(
    json.dumps(
        {
            "summary": s,
            "front": F.to_dict("list"),
            "histories": {
                str(seed): pd.read_csv(R / f"history_{seed}.csv")
                .iloc[::10]
                .to_dict("list")
                for seed in [411, 412, 413, 414, 415]
            },
        }
    )
)