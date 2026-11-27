"""Compare the three eight-station layouts on the same risk map.

Panels: original NSGA-II layout (seed 411), the layout after the relocation
test, and the exact HiGHS optimum. Reads only files already in results/ and
does not change any saved results.

Run after run_nsga2.py and audit_solution.py:
    python make_comparison_figure.py
Output: figures/layout_comparison.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

P = Path(__file__).resolve().parent
R = P / "results"
(P / "figures").mkdir(exist_ok=True)

cells = pd.read_csv(R / "selected_layout_225_cells.csv")
active = cells[cells.active_demand].reset_index(drop=True)

# Station selections for the 210 included cells, in the same order as `active`.
original = active.nsga2_station_selected.to_numpy()
layouts = []
for name, sel in [
    ("NSGA-II layout (seed 411)", original),
    ("After relocation test", pd.read_csv(R / "local_search_layout.csv").station_selected.to_numpy()),
    ("Exact optimum (HiGHS)", pd.read_csv(R / "fixed_count_milp_layout.csv").station_selected.to_numpy()),
]:
    layouts.append((name, sel.astype(bool)))

# Recompute each layout's score so the labels come from the data, not hard-coded values.
xy = active[["easting_epsg26917_m", "northing_epsg26917_m"]].to_numpy()
d = cdist(xy, xy) / 1000
w = active.risk_per_1000_over_6_years.to_numpy()

mask = ~cells.active_demand.to_numpy().reshape(15, 15)
risk = np.ma.masked_where(mask, cells.risk_per_1000_over_6_years.to_numpy().reshape(15, 15))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(1, 3, figsize=(12, 4.3))
fig.subplots_adjust(left=0.05, right=0.92, bottom=0.2, top=0.85, wspace=0.25)

orig_x = active.column_from_west[original] - 0.5
orig_y = active.row_from_south[original] - 0.5

for i, (ax, (name, sel)) in enumerate(zip(axs, layouts)):
    nearest = d[:, sel].min(axis=1)
    assert sel.sum() == 8 and nearest.max() <= 4 + 1e-9
    score = w @ nearest
    im = ax.imshow(risk, origin="lower", extent=[0, 15, 0, 15], cmap="magma_r", vmin=0)
    if i > 0:  # show where the original NSGA-II stations were
        ax.scatter(orig_x, orig_y, s=60, facecolors="none", edgecolors="#2b6cb0",
                   linewidths=1.2, label="Original NSGA-II site")
    ax.scatter(active.column_from_west[sel] - 0.5, active.row_from_south[sel] - 0.5,
               c="#13d7e7", edgecolor="#17374a", s=90, marker="*", zorder=3, label="Station")
    ax.set_title(f"{name}\nRisk-weighted distance: {score:,.2f}", fontsize=10.5)
    ax.set_xticks([0, 5, 10, 15]); ax.set_yticks([0, 5, 10, 15])
    ax.set_xlabel("Easting (km)")
    if i == 0:
        ax.set_ylabel("Northing (km)")

cax = fig.add_axes([0.935, 0.2, 0.012, 0.65])
fig.colorbar(im, cax=cax, label="Six-year fires per 1,000 residents")
handles, labels = axs[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
fig.savefig(P / "figures" / "layout_comparison.png", dpi=220, bbox_inches="tight")
print("Saved figures/layout_comparison.png")