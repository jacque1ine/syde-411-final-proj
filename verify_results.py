"""Independently check saved inputs, pooled layouts and the exact coverage layout."""
from pathlib import Path
import numpy as np,pandas as pd
from scipy.spatial.distance import cdist
p=Path(__file__).resolve().parent
f=pd.read_csv(p/'toronto_225_cells.csv');assert len(f)==225 and f.cell_id.nunique()==225
positive=f.population_2021_sample_based_area_estimate>0
assert np.allclose(f.loc[positive,'risk_per_1000_over_6_years'],1000*f.loc[positive,'recorded_fires_2018_2023']/f.loc[positive,'population_2021_sample_based_area_estimate'])
a=f[f.active_demand].reset_index(drop=True);assert len(a)==210
D=cdist(a[['easting_epsg26917_m','northing_epsg26917_m']],a[['easting_epsg26917_m','northing_epsg26917_m']])/1000
w=a.risk_per_1000_over_6_years.to_numpy();z=np.load(p/'results/layouts.npz')
assert np.array_equal(z['cell_ids'],a.cell_id.to_numpy())
for x,v in zip(z['X'],z['F']):
 d=D[:,x].min(axis=1);assert d.max()<=4+1e-9;assert np.allclose(v,[x.sum(),w@d])
e=pd.read_csv(p/'results/exact_set_cover_layout.csv');assert np.array_equal(e.cell_id,a.cell_id);b=e.selected.to_numpy();assert b.sum()==7 and (D[:,b].min(axis=1)<=4+1e-9).all()
print(f'PASS: 225 cells, all risk formulas, {len(z["X"])} saved layouts and seven-station benchmark coverage verified.')
