"""Build the Toronto study grid and greedy example from the archived raw files.
Dependencies: numpy pandas openpyxl matplotlib requests shapely pyproj scipy.
"""

from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from shapely.geometry import shape, box
from shapely.ops import transform, unary_union
from pyproj import Transformer
from scipy.spatial.distance import cdist
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SNAPSHOT_DIR = Path(__file__).resolve().parent
RAW_DIR = SNAPSHOT_DIR / "raw"
to_utm = Transformer.from_crs(4326, 26917, always_xy=True)
to_geographic = Transformer.from_crs(26917, 4326, always_xy=True)
cx, cy = to_utm.transform(-79.40, 43.75)
x0, y0 = round(cx / 1000) * 1000 - 7500, round(cy / 1000) * 1000 - 7500
N = 15
cell_polygons = [
    box(x0 + c * 1000, y0 + r * 1000, x0 + (c + 1) * 1000, y0 + (r + 1) * 1000)
    for r in range(N)
    for c in range(N)
]
xy = np.array([[p.centroid.x, p.centroid.y] for p in cell_polygons])
profile = pd.read_excel(RAW_DIR / "population.xlsx", header=None)
assert (
    profile.iloc[3, 0].strip()
    == "Total - Age groups of the population - 25% sample data"
)
poplookup = {
    str(int(profile.iloc[1, c])): float(profile.iloc[3, c]) for c in range(1, 159)
}
features = json.loads((RAW_DIR / "neighbourhoods.geojson").read_text())["features"]
nb = [
    (
        transform(to_utm.transform, shape(f["geometry"])),
        poplookup[str(int(f["properties"]["AREA_SHORT_CODE"]))],
    )
    for f in features
]
city = unary_union([p for p, n in nb])
# Uniform allocation within neighbourhood: an estimate, not observed 1 km counts.
pop = np.array(
    [
        sum(n * p.intersection(q).area / q.area for q, n in nb if p.intersects(q))
        for p in cell_polygons
    ]
)
land = np.array([p.intersection(city).area / 1e6 for p in cell_polygons])
fire = pd.read_csv(RAW_DIR / "fires.csv", low_memory=False)
f = fire[
    fire.TFS_Alarm_Time.str[:4].between("2018", "2023")
    & fire.Final_Incident_Type.str.startswith(("01 -", "03 -"))
].copy()
assert not f.Incident_Number.duplicated().any()
missing = int(f[["Longitude", "Latitude"]].isna().any(axis=1).sum())
f = f.dropna(subset=["Longitude", "Latitude"])
valid_geo = f.Longitude.between(-81, -78) & f.Latitude.between(43, 45)
invalid_geo = int((~valid_geo).sum())
f = f[valid_geo]
fx, fy = to_utm.transform(f.Longitude.to_numpy(), f.Latitude.to_numpy())
col = np.floor((fx - x0) / 1000).astype(int)
row = np.floor((fy - y0) / 1000).astype(int)
inside = (row >= 0) & (row < N) & (col >= 0) & (col < N)
counts = np.bincount((row[inside] * N + col[inside]), minlength=N * N)
w = np.divide(counts * 1000.0, pop, out=np.full(225, np.nan), where=pop > 0)
# Zero/undefined population cells are excluded explicitly, never assigned zero risk.
active = (pop > 0) & (land > 0.001)
assert np.isfinite(w[active]).all()
sfeatures = json.loads((RAW_DIR / "stations.geojson").read_text())["features"]
st = []
for ft in sfeatures:
    if ft["properties"]["TYPE_DESC"] == "Fire Station":
        pt = transform(to_utm.transform, shape(ft["geometry"])).centroid
        st.append((ft["properties"]["STATION"], pt.x, pt.y))
stxy = np.array([[a, b] for _, a, b in st])
actualdist = cdist(xy, stxy) / 1000
nearest = actualdist.min(axis=1)
# Simple deterministic greedy coverage followed by one-for-one risk-distance swaps.
# All active cell centres are hypothetical candidates, as in the paper.
d = cdist(xy, xy) / 1000
cover = d <= 4 + 1e-9
chosen = []
uncovered = active.copy()
while uncovered.any():
    scores = cover[uncovered].sum(axis=0).astype(float)
    scores[~active] = -1
    scores[chosen] = -1
    j = int(scores.argmax())
    assert scores[j] > 0
    chosen.append(j)
    uncovered &= ~cover[:, j]


def score(s):
    distances = d[:, s].min(axis=1)
    return (
        float(w[active] @ distances[active])
        if np.all(distances[active] <= 4 + 1e-9)
        else np.inf
    )


initial = score(chosen)
iterations = 0
while iterations < 40:
    best = score(chosen)
    replacement = None
    for k in range(len(chosen)):
        for j in np.flatnonzero(active):
            if j in chosen:
                continue
            trial = chosen.copy()
            trial[k] = int(j)
            v = score(trial)
            if v < best - 1e-9:
                best = v
                replacement = trial
    if replacement is None:
        break
    chosen = replacement
    iterations += 1
modeldist = d[:, chosen].min(axis=1)
lon, lat = to_geographic.transform(xy[:, 0], xy[:, 1])
frame = pd.DataFrame(
    {
        "cell_id": [f"R{r+1:02d}C{c+1:02d}" for r in range(N) for c in range(N)],
        "row_from_south": np.repeat(np.arange(1, 16), 15),
        "column_from_west": np.tile(np.arange(1, 16), 15),
        "easting_epsg26917_m": xy[:, 0],
        "northing_epsg26917_m": xy[:, 1],
        "longitude": lon,
        "latitude": lat,
        "area_km2": 1,
        "toronto_overlap_km2": land,
        "population_2021_sample_based_area_estimate": pop,
        "recorded_fires_2018_2023": counts,
        "risk_per_1000_over_6_years": w,
        "active_demand": active,
        "nearest_published_station_km": nearest,
        "illustrative_station_selected": np.isin(np.arange(225), chosen),
        "nearest_illustrative_station_km": modeldist,
    }
)
frame.to_csv(SNAPSHOT_DIR / "toronto_225_cells.csv", index=False)
sf = pd.DataFrame(
    st, columns=["station", "easting_epsg26917_m", "northing_epsg26917_m"]
)
sf["inside_study_square"] = (
    (sf.iloc[:, 1] >= x0)
    & (sf.iloc[:, 1] < x0 + 15000)
    & (sf.iloc[:, 2] >= y0)
    & (sf.iloc[:, 2] < y0 + 15000)
)
sf.to_csv(SNAPSHOT_DIR / "published_stations.csv", index=False)
frame.iloc[chosen].to_csv(SNAPSHOT_DIR / "illustrative_station_layout.csv", index=False)
summary = {
    "study_centre_lon_lat": [-79.4, 43.75],
    "southwest_epsg26917_m": [x0, y0],
    "population_proxy_sum": float(pop.sum()),
    "recorded_fires_in_square": int(counts.sum()),
    "citywide_selected_fire_records_missing_coordinates": missing,
    "citywide_selected_fire_records_invalid_coordinates": invalid_geo,
    "active_cells": int(active.sum()),
    "published_stations_in_square": int(sf.inside_study_square.sum()),
    "published_stations_used_for_baseline_including_outside_square": len(st),
    "illustrative_stations": len(chosen),
    "illustrative_max_distance_km": float(modeldist[active].max()),
    "illustrative_weighted_distance": score(chosen),
    "heuristic_swap_iterations": iterations,
    "method": "Greedy cover + best one-for-one swaps; no optimality claim; not NSGA-II",
    "period": "2018-2023 inclusive, 6 years; not annualized",
    "population_variable": str(profile.iloc[3, 0]).strip(),
    "raw_sha256": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in RAW_DIR.iterdir()
    },
}
(SNAPSHOT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
# Four panels with equal metric axes; stations in panel 3 are published locations.
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelcolor": "#43556a",
        "text.color": "#172b42",
    }
)
fig, axes = plt.subplots(2, 2, figsize=(14, 13))
fig.patch.set_facecolor("#f7f9fc")
fig.subplots_adjust(
    left=0.07, right=0.94, top=0.86, bottom=0.18, hspace=0.39, wspace=0.25
)
fig.text(
    0.07,
    0.958,
    "Toronto: a real-data fire-station teaching model",
    fontsize=22,
    weight="bold",
)
fig.text(
    0.07,
    0.923,
    "North York study window  |  15 × 15 cells, each 1 km²  |  fires recorded in 2018–2023",
    fontsize=12,
)
fig.text(
    0.07,
    0.893,
    "Real incident records + estimated grid population + published stations + an illustrative redesign",
    fontsize=10,
    color="#915122",
)
items = [
    (
        pop,
        "01  Estimated population",
        "2021 sample-based residents, area allocated",
        "YlGnBu",
    ),
    (
        counts,
        "02  Recorded fire incidents",
        "Fires over 6 years; intersection-level locations",
        "YlOrRd",
    ),
    (
        w,
        "03  Risk + published station locations",
        "Recorded fires / estimated residents × 1,000",
        "magma_r",
    ),
    (
        w,
        "04  Risk + illustrative station layout",
        "Same risk; greedy cover + local improvement",
        "magma_r",
    ),
]
for i, (ax, (z, title, label, cmap)) in enumerate(zip(axes.flat, items)):
    data = np.ma.masked_where(~active, z).reshape(N, N)
    im = ax.imshow(
        data,
        origin="lower",
        extent=[0, 15, 0, 15],
        cmap=cmap,
        vmin=0,
        vmax=float(np.nanmax(z[active])),
        interpolation="nearest",
    )
    ax.set_title(title, loc="left", weight="bold", fontsize=12, pad=11)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_yticks([0, 5, 10, 15])
    ax.set_xlabel("Easting from study origin (km)")
    ax.set_ylabel("Northing from study origin (km)")
    ax.set_xticks(np.arange(16), minor=True)
    ax.set_yticks(np.arange(16), minor=True)
    ax.grid(which="minor", color="white", lw=0.35, alpha=0.45)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, pad=0.025, fraction=0.046)
    cb.ax.tick_params(labelsize=8)
    ax.text(0, -0.19, label, transform=ax.transAxes, fontsize=9)
    if i == 2:
        s = sf[sf.inside_study_square]
        ax.scatter(
            (s.iloc[:, 1] - x0) / 1000,
            (s.iloc[:, 2] - y0) / 1000,
            s=60,
            marker="^",
            c="#23d6e8",
            edgecolor="#102d40",
            linewidth=0.7,
            label=f"{len(s)} published stations in window",
        )
        ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    if i == 3:
        ax.scatter(
            (xy[chosen, 0] - x0) / 1000,
            (xy[chosen, 1] - y0) / 1000,
            s=75,
            marker="*",
            c="#23d6e8",
            edgecolor="#102d40",
            linewidth=0.7,
            label=f"{len(chosen)} illustrative stations",
        )
        ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
fig.text(
    0.07,
    0.075,
    f"{counts.sum():,} recorded fires in the square  |  {len(chosen)} hypothetical stations cover all {active.sum()} demand-cell centres within 4 km.",
    fontsize=11,
    weight="bold",
)
fig.text(
    0.07,
    0.046,
    "Blank cells have no or negligible Toronto overlap. Population is uniform within neighbourhoods; fire locations are intersection-rounded.",
    fontsize=9,
)
fig.text(
    0.07,
    0.024,
    "Sources: City of Toronto Open Data, 2021 neighbourhood profiles, fire incidents, station locations and neighbourhood boundaries.",
    fontsize=9,
)
fig.text(
    0.07,
    0.006,
    "Illustrative layout is a heuristic, not an optimal solution, NSGA-II result, historical reconstruction, capacity model or operational recommendation.",
    fontsize=8,
    color="#915122",
)
fig.savefig(
    SNAPSHOT_DIR / "toronto_real_data_model.png", dpi=190, facecolor=fig.get_facecolor()
)
fig.savefig(SNAPSHOT_DIR / "toronto_real_data_model.svg", facecolor=fig.get_facecolor())
print(json.dumps({k: v for k, v in summary.items() if k != "raw_sha256"}, indent=2))
