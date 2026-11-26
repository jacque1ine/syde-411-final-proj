"""Run binary NSGA-II on the frozen Toronto CSV. No station-capacity model.
Run python3 run_nsga2.py. Results are saved under results/.
"""
from pathlib import Path
import json,time,platform
import numpy as np,pandas as pd,pymoo,scipy
from scipy.spatial.distance import cdist
from scipy.optimize import milp,Bounds,LinearConstraint
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
P=Path(__file__).resolve().parent;O=P/'results';O.mkdir(exist_ok=True)
f=pd.read_csv(P/'toronto_225_cells.csv');f=f[f.active_demand].reset_index(drop=True)
xy=f[['easting_epsg26917_m','northing_epsg26917_m']].to_numpy();d=cdist(xy,xy)/1000;cover=d<=4+1e-9;w=f.risk_per_1000_over_6_years.to_numpy();n=len(f)
scale=np.array([n,4*w.sum()+1]);hv=HV(ref_point=np.array([1.01,1.01]))
class FireProblem(Problem):
 def __init__(self):super().__init__(n_var=n,n_obj=2,n_ieq_constr=1,xl=0,xu=1,vtype=bool)
 def _evaluate(self,X,out,*args,**kwargs):
  vals=[];viol=[]
  for z in X:
   dist=d[:,z.astype(bool)].min(axis=1) if z.any() else np.full(n,1e6)
   vals.append([z.sum(),w@dist]);viol.append([np.count_nonzero(dist>4+1e-9)])
  out['F']=np.asarray(vals);out['G']=np.asarray(viol)
def initial(seed,size):
 rng=np.random.default_rng(seed);X=[]
 for k in range(size):
  z=np.zeros(n,dtype=bool);z[rng.choice(n,size=int(rng.integers(0,35)),replace=False)]=True
  unc=~cover[:,z].any(axis=1)
  while unc.any():
   scores=cover[unc].sum(axis=0).astype(float);scores[z]=-1
   candidates=np.flatnonzero(scores>=max(1,scores.max()*.8))
   z[rng.choice(candidates)]=True;unc=~cover[:,z].any(axis=1)
  X.append(z)
 return np.array(X)
class History(Callback):
 def __init__(self):super().__init__();self.rows=[]
 def notify(self,a):
  feasible=a.pop.get('G').ravel()<=0;F=a.pop.get('F')[feasible]
  self.rows.append({'generation':a.n_gen,'evaluations':a.evaluator.n_eval,'hypervolume':float(hv(F/scale)) if len(F) else 0,'min_stations':float(F[:,0].min()) if len(F) else None,'feasible_fraction':float(feasible.mean())})
allF=[];allX=[];runs=[]
for seed in [411,412,413,414,415]:
 hist=History();t=time.perf_counter()
 res=minimize(FireProblem(),NSGA2(pop_size=120,sampling=initial(seed,120),crossover=TwoPointCrossover(prob=.9),mutation=BitflipMutation(prob=1.0,prob_var=1/n),eliminate_duplicates=True),('n_gen',200),seed=seed,callback=hist,verbose=False)
 elapsed=time.perf_counter()-t
 F=np.atleast_2d(res.F);X=np.atleast_2d(res.X)
 for z,obj in zip(X,F):
  dd=d[:,z].min(axis=1);assert dd.max()<=4+1e-9;assert np.allclose(obj,[z.sum(),w@dd])
 allF.extend(F);allX.extend(X)
 pd.DataFrame(hist.rows).to_csv(O/f'history_{seed}.csv',index=False)
 runs.append({'seed':seed,'seconds':elapsed,'evaluations':int(res.algorithm.evaluator.n_eval),'min_stations':int(F[:,0].min()),'max_stations_on_front':int(F[:,0].max()),'final_hypervolume':float(hv(F/scale)),'initial_hypervolume':hist.rows[0]['hypervolume']})
 print(runs[-1],flush=True)
F=np.array(allF);X=np.array(allX);nd=NonDominatedSorting().do(F,only_non_dominated_front=True);F=F[nd];X=X[nd]
# One best layout per station count; ties resolved by the first saved realization.
order=np.lexsort((F[:,1],F[:,0]));idx=[];seen=set()
for k in order:
 if int(F[k,0]) not in seen:idx.append(k);seen.add(int(F[k,0]))
F=F[idx];X=X[idx]
pd.DataFrame(F,columns=['stations','risk_weighted_distance']).to_csv(O/'pooled_front.csv',index=False)
np.savez_compressed(O/'layouts.npz',X=X,F=F,cell_ids=f.cell_id.to_numpy(dtype=str))
# Independent exact set-cover benchmark, 15-second cap. Bound is saved even if not optimal.
t=time.perf_counter();exact=milp(np.ones(n),integrality=np.ones(n),bounds=Bounds(0,1),constraints=LinearConstraint(cover.astype(float),np.ones(n),np.full(n,np.inf)),options={'time_limit':15})
if exact.x is not None:
 z_exact=exact.x>.5
 assert np.all(cover[:,z_exact].any(axis=1))
 pd.DataFrame({'cell_id':f.cell_id,'selected':z_exact}).to_csv(O/'exact_set_cover_layout.csv',index=False)
benchmark={'status':int(exact.status),'message':exact.message,'station_count':float(exact.fun) if exact.fun is not None else None,'dual_bound':float(exact.mip_dual_bound) if getattr(exact,'mip_dual_bound',None) is not None else None,'gap':float(exact.mip_gap) if getattr(exact,'mip_gap',None) is not None else None,'seconds':time.perf_counter()-t}
# Select a transparent presentation example: lowest station count found, best pooled distance at that count.
z=X[0];dist=d[:,z].min(axis=1)
ff=pd.read_csv(P/'toronto_225_cells.csv');ff['nsga2_station_selected']=ff.cell_id.isin(f.cell_id[z]);ff['nearest_nsga2_station_km']=np.nan;ff.loc[ff.active_demand,'nearest_nsga2_station_km']=dist
ff.to_csv(O/'selected_layout_225_cells.csv',index=False)
summary={'runs':runs,'configuration':{'population':120,'generations':200,'seeds':[411,412,413,414,415],'crossover':'two-point, probability 0.9','mutation':'bit flip, individual probability 1, bit probability 1/210','constraint':'uncovered cell count <= 0; feasibility-first','initialization':'random 0-34 sites, greedy randomized coverage completion','hypervolume':'F normalized by [210, 4*sum(w)+1], reference [1.01,1.01]'},'exact_set_cover':benchmark,'selected':{'stations':int(z.sum()),'risk_weighted_distance':float(w@dist),'max_distance_km':float(dist.max()),'coverage_percent':100,'cell_ids':f.cell_id[z].tolist()},'pooled_points':len(F),'sum_risk_weights':float(w.sum()),'environment':{'python':platform.python_version(),'numpy':np.__version__,'pymoo':pymoo.__version__,'scipy':scipy.__version__,'machine':platform.machine(),'processor':platform.processor()}}
(O/'experiment_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary['exact_set_cover']),flush=True)
