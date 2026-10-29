"""Configuration-driven, reproducible experiment grids: python -m ...runner."""
import argparse
import csv
import hashlib
import itertools
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from math import comb, isfinite
from pathlib import Path

import numpy as np
import scipy
from scipy.special import ndtri

from .adaptive import adaptive_search, fit_angles, circuit_probabilities
from .core import exact_layer, optimize_pair
from .sat import generate_clauses, sat_energies, sat_moments, to_dimacs

METHODS = ('adaptive', 'fixed_evt', 'exact_greedy', 'uniform')
KEYS = {'qubits', 'clause_sizes', 'densities', 'depths', 'instance_seeds',
        'methods', 'repeats', 'shots', 'warmup', 'optimizer_maxiter',
        'sampling_seed', 'save_histories'}


def positive_int(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def validate_config(config):
    if not isinstance(config, dict) or not KEYS <= set(config) or set(config) - KEYS - {"energy_penalty"}:
        missing = KEYS - set(config) if isinstance(config, dict) else KEYS
        extra = set(config) - KEYS - {"energy_penalty"} if isinstance(config, dict) else set()
        raise ValueError(f'Configuration keys: missing={sorted(missing)}, unknown={sorted(extra)}')
    for key, minimum in [('qubits', 2), ('clause_sizes', 2), ('depths', 0), ('instance_seeds', 0)]:
        values = config[key]
        if not isinstance(values, list) or not values:
            raise ValueError(f'{key} must be a nonempty list')
        for value in values:
            positive_int(value, key, minimum)
        if len(set(values)) != len(values):
            raise ValueError(f'{key} contains duplicates')
    if max(config['qubits']) > 20:
        raise ValueError('Exact statevector backend supports at most 20 qubits')
    if max(config['clause_sizes']) > min(config['qubits']):
        raise ValueError('Every clause size must fit every requested qubit count')
    methods = config['methods']
    if not isinstance(methods, list) or not methods or any(m not in METHODS for m in methods):
        raise ValueError(f'methods must be selected from {METHODS}')
    if len(set(methods)) != len(methods):
        raise ValueError('methods contains duplicates')
    for key in ('repeats', 'shots', 'warmup', 'optimizer_maxiter'):
        positive_int(config[key], key)
    positive_int(config['sampling_seed'], 'sampling_seed', 0)
    if config['warmup'] > config['shots']:
        raise ValueError('warmup must not exceed shots')
    if type(config['save_histories']) is not bool:
        raise ValueError('save_histories must be true or false')
    densities = config['densities']
    if not isinstance(densities, dict) or set(densities) != {str(k) for k in config['clause_sizes']}:
        raise ValueError('densities must have exactly one string key per clause size')
    for n, k in itertools.product(config['qubits'], config['clause_sizes']):
        alpha = densities[str(k)]
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not isfinite(alpha) or alpha <= 0:
            raise ValueError('Densities must be positive finite numbers')
        if not 1 <= round(n * alpha) <= comb(n, k) * 2**k:
            raise ValueError(f'Invalid clause count for n={n}, k={k}, alpha={alpha}')
    penalty = config.get("energy_penalty", 0.)
    if isinstance(penalty, bool) or not isinstance(penalty, (int,float)) or not isfinite(penalty) or penalty < 0:
        raise ValueError("energy_penalty must be finite and nonnegative")
    return config


def plan(config):
    validate_config(config)
    return [dict(n=n, k=k, m=round(n*config['densities'][str(k)]), depth=depth,
                 seed=seed, id=f'n{n}_k{k}_m{round(n*config["densities"][str(k)])}_p{depth}_seed{seed}')
            for n,k,depth,seed in itertools.product(config['qubits'], config['clause_sizes'],
                                                     config['depths'], config['instance_seeds'])]


def exact_greedy_fit(energies, mean, sigma, depth, seed, maxiter):
    """Oracle benchmark: optimize only the newest layer against true P(optimum)."""
    centered = energies-mean
    ground = energies == energies.min()
    state = np.ones(len(energies), dtype=complex)/np.sqrt(len(energies))
    angles, nfev = [], 0
    for layer in range(1, depth+1):
        if sigma == 0:
            beta, gamma, calls = 0., 0., 0
        else:
            beta,gamma,calls = optimize_pair(
                lambda b,g:float(np.sum(abs(exact_layer(state,centered,b,g)[ground])**2)),
                sigma,seed+layer,maxiter=maxiter)
        state = exact_layer(state,centered,beta,gamma)
        angles.append((beta,gamma))
        nfev += calls
    probabilities = abs(state)**2
    if not np.isclose(probabilities.sum(),1.,atol=1e-10):
        raise RuntimeError('State normalization lost')
    return dict(angles=angles,nfev=nfev,probabilities=probabilities/probabilities.sum())


def evaluate_assignment(assignment, clauses):
    return sum(all(not bool((assignment>>(abs(lit)-1))&1) if lit>0
                   else bool((assignment>>(abs(lit)-1))&1) for lit in clause)
               for clause in clauses)


def fixed_trial(probabilities, evaluate, rng, shots, nfev, fits):
    assignments = rng.choice(len(probabilities), size=shots, p=probabilities)
    values = [evaluate(int(assignment)) for assignment in assignments]
    best = np.minimum.accumulate(values)
    return dict(shots=shots,best_energy=float(best[-1]),
                best_assignment=int(assignments[np.argmin(values)]),nfev=nfev,fits=fits,
                history=[dict(shot=i+1,assignment=int(x),energy=float(e),best_energy=float(b))
                         for i,(x,e,b) in enumerate(zip(assignments,values,best))])


def run_case(case, config):
    started = time.perf_counter()
    n,k,m,depth,seed = (case[key] for key in ('n','k','m','depth','seed'))
    clauses = generate_clauses(n,k,m,seed)
    mean,sigma = sat_moments(clauses)
    energies = sat_energies(n,clauses)
    target = float(mean+sigma*ndtri(2.**(-n)))
    cache = {}
    def prepare(physical_target):
        key = float(physical_target)
        if key not in cache:
            fitted = (fit_angles(mean,sigma,key,depth,seed,config['optimizer_maxiter'],energy_penalty=config.get('energy_penalty',0.)) if sigma > 0
                      else dict(target=key,angles=[(0.,0.)]*depth,nfev=0))
            cache[key] = dict(**fitted,probabilities=circuit_probabilities(energies,mean,fitted['angles']))
        return cache[key]
    evaluate = lambda assignment:evaluate_assignment(assignment,clauses)
    fixed = prepare(target) if any(method in config['methods'] for method in ('adaptive','fixed_evt')) else None
    greedy = (exact_greedy_fit(energies,mean,sigma,depth,seed,config['optimizer_maxiter'])
              if 'exact_greedy' in config['methods'] else None)
    curves = {method:[] for method in config['methods']}
    trials = []
    for repeat in range(config['repeats']):
        # Deliberately omit depth: paired random streams across the depth grid too.
        rng_seed = [config['sampling_seed'],n,k,m,seed,repeat]
        for method in config['methods']:
            rng = np.random.default_rng(np.random.SeedSequence(rng_seed))
            if method == 'adaptive':
                trial = adaptive_search(target,prepare,evaluate,rng,config['shots'],config['warmup'])
            else:
                fitted = fixed if method == 'fixed_evt' else greedy if method == 'exact_greedy' else None
                probabilities = fitted['probabilities'] if fitted else np.ones(len(energies))/len(energies)
                fits = [dict(before_shot=1,nfev=fitted['nfev'])] if fitted else []
                trial = fixed_trial(probabilities,evaluate,rng,config['shots'],fitted['nfev'] if fitted else 0,fits)
            trial.update(method=method,repeat=repeat)
            curves[method].append([row['best_energy'] for row in trial['history']])
            if not config['save_histories']:
                del trial['history']
            trials.append(trial)
    optimum = int(energies.min())  # Scoring only; not passed to adaptive fitting/controller.
    summaries, trajectories = [], {}
    for method in config['methods']:
        best = np.array(curves[method])
        selected = [t for t in trials if t['method']==method]
        summaries.append(dict(**case,method=method,repeats=config['repeats'],shots=config['shots'],
            optimum=optimum,initial_target=target,success_rate=float(np.mean(best[:,-1]==optimum)),
            mean_best_energy=float(best[:,-1].mean()),mean_gap=float((best[:,-1]-optimum).mean()),
            mean_fits=float(np.mean([len(t['fits']) for t in selected])),
            mean_nfev=float(np.mean([t['nfev'] for t in selected]))))
        trajectories[method] = dict(success_by_shot=np.mean(best==optimum,axis=0).tolist(),
                                    mean_gap_by_shot=np.mean(best-optimum,axis=0).tolist())
    return dict(case=case,clauses=clauses,mean=mean,sigma=sigma,optimum=optimum,
                initial_target=target,summaries=summaries,trajectories=trajectories,trials=trials,
                gaussian_fits=[{key:value for key,value in fit.items() if key!='probabilities'} for fit in cache.values()],
                exact_greedy_fit={key:value for key,value in greedy.items() if key!='probabilities'} if greedy else None,
                seconds=time.perf_counter()-started)


def plot_case(result, path):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/grover-qaoa-matplotlib')
    os.environ.setdefault('XDG_CACHE_HOME','/tmp/grover-qaoa-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for method, values in result['trajectories'].items():
        shots = np.arange(1,len(values['success_by_shot'])+1)
        axes[0].plot(shots,values['success_by_shot'],label=method)
        axes[1].plot(shots,values['mean_gap_by_shot'],label=method)
    axes[0].set(ylabel='P(optimum found)',ylim=(0,1))
    axes[1].set(ylabel='Mean best-so-far gap')
    for ax in axes:
        ax.set(xlabel='Shots')
        ax.legend()
    fig.suptitle(result['case']['id'])
    fig.tight_layout()
    fig.savefig(path,dpi=130)
    plt.close(fig)


def run_grid(config, output):
    cases = plan(config)
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)  # Refuse to overwrite earlier results.
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    git = subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True)
    dirty = subprocess.run(['git','status','--porcelain'],capture_output=True,text=True)
    metadata = dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                    git_commit=git.stdout.strip() if git.returncode==0 else None,
                    git_dirty=bool(dirty.stdout.strip()),created_utc=datetime.now(timezone.utc).isoformat(),
                    config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest())
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    summaries = []
    for i,case in enumerate(cases,1):
        print(f'[{i}/{len(cases)}] {case["id"]}',flush=True)
        result = run_case(case,config)
        path = output/case['id']
        path.mkdir()
        (path/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        (path/'formula.cnf').write_text(to_dimacs(case['n'],result['clauses']))
        plot_case(result,path/'comparison.png')
        summaries.extend(result['summaries'])
        # Refresh the summary after each completed case; completed cases survive failures.
        with (output/'summary.csv').open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(summaries[0]),lineterminator="\n")
            writer.writeheader()
            writer.writerows(summaries)
    write_aggregate(summaries, output)
    return summaries


def write_aggregate(summaries, output):
    """Equal instance weights, sample spread across instances (not a CI)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = {}
    for row in summaries:
        key = (row['n'],row['k'],row['m'],row['depth'],row['method'])
        groups.setdefault(key,[]).append(row)
    aggregate = []
    for (n,k,m,depth,method),rows in groups.items():
        values = [r['success_rate'] for r in rows]
        aggregate.append(dict(n=n,k=k,m=m,depth=depth,method=method,instances=len(rows),
            success_rate=float(np.mean(values)),
            instance_success_std=float(np.std(values,ddof=1)) if len(rows)>1 else None,
            mean_gap=float(np.mean([r['mean_gap'] for r in rows])),
            mean_fits=float(np.mean([r['mean_fits'] for r in rows])),
            mean_nfev=float(np.mean([r['mean_nfev'] for r in rows]))))
    with (output/'aggregate.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(aggregate[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(aggregate)
    panels = sorted({(r['n'],r['k'],r['m']) for r in aggregate})
    fig,axes = plt.subplots(len(panels),1,figsize=(7,3.5*len(panels)),squeeze=False)
    for ax,(n,k,m) in zip(axes[:,0],panels):
        for method in dict.fromkeys(r['method'] for r in aggregate):
            rows = sorted((r for r in aggregate if (r['n'],r['k'],r['m'],r['method'])==(n,k,m,method)),key=lambda r:r['depth'])
            ax.plot([r['depth'] for r in rows],[r['success_rate'] for r in rows],marker='o',label=method)
        ax.set(title=f'Max-{k}-SAT: n={n}, m={m}',xlabel='Layers',ylabel='P(optimum within shot budget)',ylim=(0,1))
        ax.legend()
    fig.tight_layout()
    fig.savefig(output/'depth_comparison.png',dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path)
    parser.add_argument('--output',type=Path,help='New directory; existing paths are rejected')
    parser.add_argument('--dry-run',action='store_true',help='Validate and print grid without fitting')
    args = parser.parse_args()
    try:
        config = validate_config(json.loads(args.config.read_text()))
        cases = plan(config)
        print(f'{len(cases)} cases; {len(cases)*len(config["methods"])*config["repeats"]} sampling trials; '
              f'{len(cases)*len(config["methods"])*config["repeats"]*config["shots"]} total shots')
        print(f'Largest single complex statevector: {16*2**max(config["qubits"])/2**20:.1f} MiB; '
              'actual peak memory includes several arrays. Runtime grows exponentially in qubits.')
        if 'exact_greedy' in config['methods']:
            print('exact_greedy is an oracle benchmark using the exact optimum; its fitting cost is not in the shot budget.')
        if args.dry_run:
            for case in cases:
                print(case['id'])
            return
        output = args.output or Path('runs')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        run_grid(config,output)
        print(f'Results: {output.resolve()}')
    except (ValueError,OSError) as error:
        parser.exit(2,f'Error: {error}\n')


if __name__ == '__main__':
    main()
