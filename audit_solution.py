"""Audit the saved eight-station NSGA-II example without overwriting it.
Run after run_nsga2.py. Output: results/solution_audit.json and local_search_layout.csv.
"""
from pathlib import Path
import json,time
import numpy as np,pandas as pd
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix
from scipy.optimize import milp, Bounds, LinearConstraint
p=Path(__file__).resolve().parent
f=pd.read_csv(p/'toronto_225_cells.csv');f=f[f.active_demand].reset_index(drop=True)
d=cdist(f[['easting_epsg26917_m','northing_epsg26917_m']],f[['easting_epsg26917_m','northing_epsg26917_m']])/1000
w=f.risk_per_1000_over_6_years.to_numpy();n=len(f)
z=np.load(p/'results/layouts.npz')['X'][0].copy();k=int(z.sum())
def evaluate(a):
 dist=d[:,a].min(axis=1)
 return float(w@dist) if np.all(dist<=4+1e-9) else np.inf
initial=evaluate(z);steps=[];first_improving=0;t=time.perf_counter()
while True:
 old=evaluate(z);best=old;chosen=None;improving=0;tested=0
 for j in np.flatnonzero(z):
  for h in np.flatnonzero(~z):
   candidate=z.copy();candidate[j]=False;candidate[h]=True;v=evaluate(candidate);tested+=1
   if v<old-1e-8:improving+=1
   if v<best-1e-8:best=v;chosen=candidate
 if not steps:first_improving=improving
 if chosen is None:break
 z=chosen;steps.append({'step':len(steps)+1,'distance':best,'improving_neighbors_before_move':improving})
local={'starting_stations':k,'starting_distance':initial,'first_neighborhood_tested':k*(n-k),'first_improving_neighbors':first_improving,'moves':len(steps),'final_distance':evaluate(z),'relative_improvement_percent':100*(initial-evaluate(z))/initial,'final_neighborhood_tested':tested,'final_improving_neighbors':improving,'tolerance':1e-8,'seconds':time.perf_counter()-t,'history':steps}
assert z.sum()==k and (d[:,z].min(axis=1)<=4+1e-9).all()
pd.DataFrame({'cell_id':f.cell_id,'station_selected':z}).to_csv(p/'results/local_search_layout.csv',index=False)
# Same fixed station ceiling, exact formulation with binary y and continuous assignments x.
r,c=np.where(d<=4+1e-9);m=len(r);q=np.arange(m)
rows=np.concatenate([r,n+q,n+q,np.full(n,n+m)])
cols=np.concatenate([n+q,n+q,c,np.arange(n)])
values=np.concatenate([np.ones(m),np.ones(m),-np.ones(m),np.ones(n)])
A=coo_matrix((values,(rows,cols)),shape=(n+m+1,n+m)).tocsc()
lb=np.concatenate([np.ones(n),np.full(m,-np.inf),[-np.inf]])
ub=np.concatenate([np.ones(n),np.zeros(m),[k]])
objective=np.concatenate([np.zeros(n),w[r]*d[r,c]])
t=time.perf_counter();res=milp(objective,integrality=np.concatenate([np.ones(n),np.zeros(m)]),bounds=Bounds(0,1),constraints=LinearConstraint(A,lb,ub),options={'time_limit':15,'mip_rel_gap':0})
benchmark={'station_ceiling':k,'status':int(res.status),'message':res.message,'seconds':time.perf_counter()-t,'incumbent':float(res.fun) if res.fun is not None else None,'lower_bound':float(res.mip_dual_bound) if getattr(res,'mip_dual_bound',None) is not None else None,'relative_gap':float(res.mip_gap) if getattr(res,'mip_gap',None) is not None else None}
if res.x is not None:
 a=res.x[:n]>.5;assert a.sum()<=k and np.all(d[:,a].min(axis=1)<=4+1e-9)
 assert np.isclose(evaluate(a),res.fun,rtol=1e-5)
 pd.DataFrame({'cell_id':f.cell_id,'station_selected':a}).to_csv(p/'results/fixed_count_milp_layout.csv',index=False)
traces=[]
for seed in [411,412,413,414,415]:
 h=pd.read_csv(p/f'results/history_{seed}.csv')
 traces.append({'seed':seed,'hv_generation_100':float(h.iloc[99].hypervolume),'hv_generation_200':float(h.iloc[-1].hypervolume),'gain_100_to_200':float(h.iloc[-1].hypervolume-h.iloc[99].hypervolume),'min_feasible_fraction_in_surviving_populations':float(h.feasible_fraction.min())})
out={'local_search':local,'fixed_count_milp':benchmark,'late_progress':traces}
(p/'results/solution_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
