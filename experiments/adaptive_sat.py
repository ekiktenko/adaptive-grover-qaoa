"""Paired 100-shot adaptive/fixed/uniform experiment on saved SAT formulas."""
import argparse
import json
import os
from pathlib import Path
import time
os.environ.setdefault('MPLCONFIGDIR','/tmp/adaptive-grover-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/workspace/.cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
from adaptive_grover_qaoa.adaptive import adaptive_search,fit_angles,circuit_probabilities
from adaptive_grover_qaoa.sat import sat_moments,sat_energies

p=argparse.ArgumentParser()
p.add_argument('--input',default='results/sat/comparison.json')
p.add_argument('--output',default='results/sat/adaptive')
p.add_argument('--repeats',type=int,default=100)
p.add_argument('--budget',type=int,default=100)
p.add_argument('--warmup',type=int,default=10)
p.add_argument('--depth',type=int,default=20)
p.add_argument('--maxiter',type=int,default=100)
a=p.parse_args()
if a.repeats<1 or not 1<=a.warmup<=a.budget:
    p.error('Require repeats>=1 and 1<=warmup<=budget')
out=Path(a.output)
out.mkdir(parents=True,exist_ok=True)
source=json.loads(Path(a.input).read_text())
runs=[]
started=time.perf_counter()
for old in source['runs']:
    energies=sat_energies(old['n'],old['clauses'])
    mean,sigma=sat_moments(old['clauses'])
    # Controller evaluates only measured assignments, directly from clauses.
    def evaluate(assignment):
        return sum(all(not bool((assignment>>(abs(lit)-1))&1) if lit>0
                       else bool((assignment>>(abs(lit)-1))&1) for lit in clause)
                   for clause in old['clauses'])
    assert all(evaluate(i)==energies[i] for i in range(len(energies)))
    cache={}
    def prepare(target):
        key=float(target)
        if key not in cache:
            fitted=fit_angles(mean,sigma,target,depth=a.depth,seed=old['seed'],maxiter=a.maxiter)
            cache[key]=dict(**fitted,probabilities=circuit_probabilities(energies,mean,fitted['angles']))
        return cache[key]
    initial=old['estimated_minimum']
    fixed=prepare(initial)
    trials=[]
    for repeat in range(a.repeats):
        # Paired random streams across methods; fitting uses separate fixed seeds.
        seed=np.random.SeedSequence([2026,old['k'],old['seed'],repeat])
        adaptive=adaptive_search(initial,prepare,evaluate,np.random.default_rng(seed),a.budget,a.warmup)
        trials.append(dict(method='adaptive',repeat=repeat,**adaptive))
        for method,probabilities in [('fixed_evt',fixed['probabilities']),('uniform',np.ones(len(energies))/len(energies))]:
            rng=np.random.default_rng(seed)
            assignments=rng.choice(len(energies),size=a.budget,p=probabilities)
            values=[evaluate(int(i)) for i in assignments]
            best=np.minimum.accumulate(values)
            history=[dict(shot=i+1,assignment=int(x),energy=float(e),best_energy=float(b))
                     for i,(x,e,b) in enumerate(zip(assignments,values,best))]
            trials.append(dict(method=method,repeat=repeat,best_energy=float(best[-1]),
                best_assignment=int(assignments[np.argmin(values)]),shots=a.budget,
                nfev=fixed['nfev'] if method=='fixed_evt' else 0,
                fits=[dict(target=float(initial),before_shot=1,nfev=fixed['nfev'])] if method=='fixed_evt' else [],history=history))
    fits=[{key:value for key,value in fitted.items() if key!='probabilities'} for fitted in cache.values()]
    result=dict(n=old['n'],k=old['k'],m=old['m'],seed=old['seed'],clauses=old['clauses'],
                minimum=int(energies.min()),initial_target=initial,depth=a.depth,
                cached_fits=fits,trials=trials)
    runs.append(result)
    # Incremental artifacts retain completed instances during long experiments.
    (out/f"max{old['k']}_seed{old['seed']}.json").write_text(json.dumps(result)+'\n')
    print(f"k={old['k']} seed={old['seed']} unique_fits={len(cache)}",flush=True)
summary={}
fig,axes=plt.subplots(2,2,figsize=(11,8))
for row,k in enumerate((3,4)):
    group=[run for run in runs if run['k']==k]
    summary[str(k)]={}
    for method in ('adaptive','fixed_evt','uniform'):
        curves=[]
        gaps=[]
        finals=[]
        fits=[]
        calls=[]
        instance_rates=[]
        for run in group:
            trials=[t for t in run['trials'] if t['method']==method]
            best=np.array([[h['best_energy'] for h in t['history']] for t in trials])
            curves.extend(best==run['minimum'])
            gaps.extend(best-run['minimum'])
            finals.extend(best[:,-1])
            fits.extend(len(t['fits']) for t in trials)
            calls.extend(t['nfev'] for t in trials)
            instance_rates.append(float(np.mean(best[:,-1]==run['minimum'])))
        curves=np.asarray(curves)
        gaps=np.asarray(gaps)
        summary[str(k)][method]=dict(success_rate=float(curves[:,-1].mean()),
            mean_best_energy=float(np.mean(finals)),mean_gap=float(gaps[:,-1].mean()),
            mean_fits=float(np.mean(fits)),mean_nfev=float(np.mean(calls)),
            instance_success_rates=instance_rates,
            success_by_shot=curves.mean(axis=0).tolist(),mean_gap_by_shot=gaps.mean(axis=0).tolist())
        axes[row,0].plot(range(1,a.budget+1),curves.mean(axis=0),label=method)
        axes[row,1].plot(range(1,a.budget+1),gaps.mean(axis=0),label=method)
    axes[row,0].set(ylabel='P(optimum found within shot budget)',ylim=(0,1))
    axes[row,1].set(ylabel='Mean best-so-far excess violations')
    for ax in axes[row]:
        ax.set(xlabel='Shots (including warmup)',title=f'Max-{k}-SAT: {len(group)} instances, {a.repeats} repeats each')
        ax.axvline(a.warmup,color='gray',linestyle=':',alpha=.5)
        ax.legend()
fig.tight_layout()
fig.savefig(out/'comparison.png',dpi=160)
payload=dict(arguments=vars(a),numpy=np.__version__,scipy=scipy.__version__,
             seconds=time.perf_counter()-started,summary=summary,
             instance_files=[f"max{r['k']}_seed{r['seed']}.json" for r in runs])
(out/'summary.json').write_text(json.dumps(payload,indent=2)+'\n')
for k,methods in summary.items():
    for method,stats in methods.items():
        print(k,method,{key:value for key,value in stats.items() if not isinstance(value,list)})
