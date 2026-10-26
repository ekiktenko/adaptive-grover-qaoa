"""Paired target-energy ablation using the saved Max-SAT ensemble."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/adaptive-grover-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/workspace/.cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
from adaptive_grover_qaoa.sat import sat_energies, sat_moments, analytical_trajectory

p=argparse.ArgumentParser()
p.add_argument('--input',default='results/sat/comparison.json')
p.add_argument('--output',default='results/sat/targets')
a=p.parse_args()
source=json.loads(Path(a.input).read_text())
out=Path(a.output)
out.mkdir(parents=True,exist_ok=True)
runs=[]
for old in source['runs']:
    energies=sat_energies(old['n'],old['clauses'])
    mean,sigma=sat_moments(old['clauses'])
    np.testing.assert_allclose([mean,sigma,energies.min()],
                               [old['energy_mean'],old['sigma'],old['minimum']])
    variants={}
    targets=dict(evt=old['estimated_minimum'],clipped=max(0.,old['estimated_minimum']),
                 oracle=float(energies.min()))
    for name,target in targets.items():
        records=analytical_trajectory(energies,mean,sigma,target,
                seed=old['seed'],depth=old['depth'],maxiter=old['optimizer']['maxiter'])
        variants[name]=records
        # EVT rerun must reproduce the original experiment before comparison.
        if name=='evt':
            expected=[r['success'] for r in old['records'] if r['method']=='analytical']
            np.testing.assert_allclose([r['success'] for r in records],expected,atol=1e-10,rtol=1e-10)
    oracle_greedy=[r for r in old['records'] if r['method']=='exact_greedy']
    run={key:old[key] for key in ('n','k','m','seed','clauses','depth','minimum','energy_mean','sigma','ground_degeneracy','optimizer')}
    run.update(targets=targets,variants=variants,exact_greedy=oracle_greedy)
    runs.append(run)
    print(f"k={old['k']} seed={old['seed']} targets={targets} final="+
          str({key:round(value[-1]['success'],5) for key,value in variants.items()}),flush=True)

fig,axes=plt.subplots(2,2,figsize=(11,8))
summary={}
for row,k in enumerate((3,4)):
    group=[r for r in runs if r['k']==k]
    stats={}
    for name in ('evt','clipped','oracle','exact_greedy'):
        rr=[r['exact_greedy'] if name=='exact_greedy' else r['variants'][name] for r in group]
        success=np.array([[x['success'] for x in records] for records in rr])
        energy=np.array([[x['mean_energy'] for x in records] for records in rr])
        gap=(energy-np.array([r['minimum'] for r in group])[:,None])/group[0]['m']
        stats[name]=dict(final_success_mean=float(success[:,-1].mean()),
            final_success_std=float(success[:,-1].std(ddof=1)) if len(group)>1 else None,
            final_success_values=success[:,-1].tolist(),
            final_mean_energy=float(energy[:,-1].mean()),
            final_normalized_gap=float(gap[:,-1].mean()),
            best_ensemble_layer=int(success.mean(axis=0).argmax()),
            best_ensemble_success=float(success.mean(axis=0).max()))
        for ax,data in [(axes[row,0],success),(axes[row,1],gap)]:
            line,=ax.plot(range(group[0]['depth']+1),data.mean(axis=0),label=name)
            ax.fill_between(range(group[0]['depth']+1),data.min(axis=0),data.max(axis=0),color=line.get_color(),alpha=.1)
    summary[str(k)]=stats
    axes[row,0].set(ylabel='P(optimal assignment)',ylim=(0,1))
    axes[row,1].set(ylabel='Mean excess violations / m')
    for ax in axes[row]:
        ax.set(xlabel='Layers',title=f"Max-{k}-SAT: n={group[0]['n']}, m={group[0]['m']}, {len(group)} seeds")
        ax.legend()
fig.suptitle('Gaussian target ablation; shading: instance min–max')
fig.tight_layout()
fig.savefig(out/'comparison.png',dpi=160)
(out/'comparison.json').write_text(json.dumps(dict(input=a.input,numpy=np.__version__,scipy=scipy.__version__,summary=summary,runs=runs),indent=2)+'\n')
print(json.dumps(summary,indent=2))
