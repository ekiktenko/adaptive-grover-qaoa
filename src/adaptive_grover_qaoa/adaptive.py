"""Sample-only adaptive controller and an exact-statevector sampling backend."""
import numpy as np
from .core import GaussianModel, exact_layer, optimize_pair


def fit_angles(mean, sigma, target, depth=20, seed=42, maxiter=100):
    """Refit all layers from scratch; no enumerated spectrum or minimum input."""
    if sigma <= 0 or depth < 0:
        raise ValueError('Require positive sigma and nonnegative depth')
    model = GaussianModel(sigma)
    angles, calls = [], 0
    for layer in range(1,depth+1):
        beta,gamma,nfev = optimize_pair(lambda b,g:model.target_score(target-mean,b,g),
                                       sigma,seed+layer,maxiter=maxiter)
        model.append(beta,gamma)
        angles.append((beta,gamma))
        calls += nfev
    return dict(target=float(target), angles=angles, nfev=calls)


def circuit_probabilities(energies, mean, angles):
    state = np.ones(len(energies),dtype=complex)/np.sqrt(len(energies))
    for beta,gamma in angles:
        state = exact_layer(state,np.asarray(energies)-mean,beta,gamma)
    probabilities = abs(state)**2
    if not np.isclose(probabilities.sum(),1.,atol=1e-10):
        raise RuntimeError('State normalization lost')
    return probabilities/probabilities.sum()


def adaptive_search(initial_target, prepare, evaluate, rng, budget=100, warmup=10):
    """prepare(target) -> {probabilities, nfev, ...}; evaluate(integer bitstring).

    No true minimum or spectrum is accepted by the controller. The backend's
    probabilities represent repeated fresh circuit runs. Warmup replaces the
    analytical estimate unconditionally; later updates require strict decreases.
    """
    if not 1 <= warmup <= budget or not np.isfinite(initial_target):
        raise ValueError('Require finite target and 1 <= warmup <= budget')
    target = float(initial_target)
    prepared = prepare(target)
    fits = [dict(before_shot=1,target=target,nfev=int(prepared['nfev']))]
    history, updates = [], []
    best_energy, best_assignment = float('inf'), None
    dirty = False
    for shot in range(1,budget+1):
        if dirty:
            prepared = prepare(target)
            fits.append(dict(before_shot=shot,target=target,nfev=int(prepared['nfev'])))
            dirty = False
        assignment = int(rng.choice(len(prepared['probabilities']),p=prepared['probabilities']))
        energy = float(evaluate(assignment))
        if not np.isfinite(energy):
            raise ValueError('Sample energy must be finite')
        if energy < best_energy:
            best_energy,best_assignment = energy,assignment
        old_target = target
        if shot == warmup:
            target = best_energy
            updates.append(dict(after_shot=shot,old_target=old_target,target=target,reason='warmup'))
            dirty = True
        elif shot > warmup and energy < target:
            target = energy
            updates.append(dict(after_shot=shot,old_target=old_target,target=target,reason='improvement'))
            dirty = True
        history.append(dict(shot=shot,assignment=assignment,energy=energy,
                            sampling_target=old_target,target_after=target,best_energy=best_energy))
    return dict(best_energy=best_energy,best_assignment=best_assignment,shots=budget,
                initial_target=float(initial_target),final_target=target,
                fits=fits,updates=updates,history=history,
                nfev=sum(f['nfev'] for f in fits))
