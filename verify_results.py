"""Independently check saved inputs, pooled layouts and the exact coverage layout."""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

project_dir = Path(__file__).resolve().parent
grid = pd.read_csv(project_dir / "toronto_225_cells.csv")
assert len(grid) == 225 and grid.cell_id.nunique() == 225
positive = grid.population_2021_sample_based_area_estimate > 0
assert np.allclose(
    grid.loc[positive, "risk_per_1000_over_6_years"],
    1000
    * grid.loc[positive, "recorded_fires_2018_2023"]
    / grid.loc[positive, "population_2021_sample_based_area_estimate"],
)
active_cells = grid[grid.active_demand].reset_index(drop=True)
assert len(active_cells) == 210
distance_km = (
    cdist(
        active_cells[["easting_epsg26917_m", "northing_epsg26917_m"]],
        active_cells[["easting_epsg26917_m", "northing_epsg26917_m"]],
    )
    / 1000
)
risk_weights = active_cells.risk_per_1000_over_6_years.to_numpy()
z = np.load(project_dir / "results/layouts.npz")
assert np.array_equal(z["cell_ids"], active_cells.cell_id.to_numpy())
for layout, objectives in zip(z["X"], z["F"]):
    d = distance_km[:, layout].min(axis=1)
    assert d.max() <= 4 + 1e-9
    assert np.allclose(objectives, [layout.sum(), risk_weights @ d])
benchmark = pd.read_csv(project_dir / "results/exact_set_cover_layout.csv")
assert np.array_equal(benchmark.cell_id, active_cells.cell_id)
selected = benchmark.selected.to_numpy()
assert selected.sum() == 7 and (distance_km[:, selected].min(axis=1) <= 4 + 1e-9).all()
print(
    f'PASS: 225 cells, all risk formulas, {len(z["X"])} saved layouts and seven-station benchmark coverage verified.'
)
