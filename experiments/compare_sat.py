"""Compare three GM-QAOA variants on random Max-3/4-SAT ensembles."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/adaptive-grover-mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/workspace/.cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
from adaptive_grover_qaoa.sat import compare_sat, to_dimacs

p = argparse.ArgumentParser()
p.add_argument('--n',type=int,default=10)
p.add_argument('--depth',type=int,default=20)
p.add_argument('--seeds',type=int,default=5)
p.add_argument('--alpha3',type=float,default=4.25)
p.add_argument('--alpha4',type=float,default=9.75)
p.add_argument('--maxiter',type=int,default=100)
p.add_argument('--output',default='results/sat')
a=p.parse_args()
if a.seeds < 1 or a.depth < 0:
    p.error('seeds must be positive and depth nonnegative')
out=Path(a.output)
out.mkdir(parents=True,exist_ok=True)
runs=[]
summary={}
fig,axes=plt.subplots(2,2,figsize=(11,8))
for row,(k,alpha) in enumerate([(3,a.alpha3),(4,a.alpha4)]):
    group=[]
    for seed in range(a.seeds):
        run=compare_sat(n=a.n,k=k,m=round(alpha*a.n),seed=seed,depth=a.depth,maxiter=a.maxiter)
        (out/f"max{k}_n{a.n}_m{run['m']}_seed{seed}.cnf").write_text(to_dimacs(a.n, run["clauses"]))
        group.append(run)
        runs.append(run)
        print(f"k={k} seed={seed} min={run['minimum']:.0f} EVT={run['estimated_minimum']:.3f} degeneracy={run['ground_degeneracy']}",flush=True)
    stats={}
    for method in ('constant','analytical','exact_greedy'):
        records=[[r for r in run['records'] if r['method']==method] for run in group]
        success=np.array([[r['success'] for r in rr] for rr in records])
        energy=np.array([[r['mean_energy'] for r in rr] for rr in records])
        # Gap to each instance's optimal number of satisfied clauses, divided by m.
        gap=(energy-np.array([run['minimum'] for run in group])[:,None])/group[0]['m']
        stats[method]=dict(final_mean=float(success[:,-1].mean()),
            final_std=float(success[:,-1].std(ddof=1)) if a.seeds>1 else None,
            initial_mean=float(success[:,0].mean()), final_values=success[:,-1].tolist(),
            mean_success_by_layer=success.mean(axis=0).tolist(),
            mean_normalized_gap_by_layer=gap.mean(axis=0).tolist())
        for ax,values in [(axes[row,0],success),(axes[row,1],gap)]:
            line,=ax.plot(range(a.depth+1),values.mean(axis=0),label=method)
            ax.fill_between(range(a.depth+1),values.min(axis=0),values.max(axis=0),alpha=.12,color=line.get_color())
    summary[str(k)]=stats
    axes[row,0].set(ylabel='P(optimal assignment)',ylim=(0,1))
    axes[row,1].set(ylabel='Mean excess violations / m')
    for ax in axes[row]:
        ax.set(xlabel='Layers',title=f'Max-{k}-SAT: n={a.n}, m={group[0]["m"]}, {a.seeds} seeds')
        ax.legend()
fig.suptitle('Lines: ensemble mean; shading: min–max across instances')
fig.tight_layout()
fig.savefig(out/'comparison.png',dpi=160)
payload=dict(arguments=vars(a),numpy=np.__version__,scipy=scipy.__version__,summary=summary,runs=runs)
(out/'comparison.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps(summary,indent=2))
