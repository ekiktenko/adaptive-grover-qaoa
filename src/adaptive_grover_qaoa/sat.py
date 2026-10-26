"""Uniform random unweighted Max-k-SAT; exact clause-to-Ising moments."""
from itertools import combinations
from math import comb
import numpy as np
from .core import compare_energies


def generate_clauses(n, k, m, seed=42):
    """Unique clauses, distinct variables per clause; signed 1-based literals.

    Positive i means x_i, negative i means NOT x_i. No planted solution,
    tautologies, repeated literals or duplicate clauses. Clauses are sampled
    uniformly without replacement from comb(n,k)*2**k possible clauses.
    """
    if not 2 <= k <= n <= 20 or not 1 <= m <= comb(n, k)*2**k:
        raise ValueError("Require 2 <= k <= n <= 20 and feasible positive m")
    rng = np.random.default_rng(seed)
    clauses, seen = [], set()
    # Near exhaustion, sampling indices avoids expensive rejection.
    if m > comb(n,k)*2**k // 2:
        universe = list(combinations(range(1, n+1), k))
        for index in rng.choice(len(universe)*2**k, size=m, replace=False):
            variables = universe[int(index)//2**k]
            mask = int(index) % 2**k
            clauses.append(tuple(v if (mask >> j)&1 else -v for j,v in enumerate(variables)))
        return clauses
    while len(clauses) < m:
        variables = np.sort(rng.choice(n, k, replace=False)) + 1
        signs = rng.choice([-1, 1], k)
        clause = tuple(int(v*s) for v,s in zip(variables, signs))
        if clause not in seen:
            seen.add(clause)
            clauses.append(clause)
    return clauses


def sat_energies(n, clauses):
    """Count violated clauses for every bit string (LSB corresponds to x_1)."""
    if not 1 <= n <= 20:
        raise ValueError("Exact enumeration requires 1 <= n <= 20")
    assignments = np.arange(2**n)
    energy = np.zeros(2**n, dtype=int)
    for clause in clauses:
        if not clause or any(lit == 0 or abs(lit)>n for lit in clause):
            raise ValueError("Invalid literal")
        unsatisfied = np.ones(2**n, dtype=bool)
        for literal in clause:
            bit = ((assignments >> (abs(literal)-1)) & 1).astype(bool)
            unsatisfied &= ~bit if literal > 0 else bit
        energy += unsatisfied
    return energy


def sat_moments(clauses):
    """Exact uniform mean/std from aggregated Pauli coefficients, no spectrum.

    Violation projector = product_{literal l}(1 + sign(l)*Z_|l|)/2.
    Shared subsets must be aggregated BEFORE squaring their coefficients.
    """
    coefficients = {}
    for clause in clauses:
        if not clause or any(lit == 0 for lit in clause) or len(set(map(abs, clause))) != len(clause):
            raise ValueError("Require nonempty clauses with distinct nonzero variables")
        for d in range(len(clause)+1):
            for subset in combinations(clause, d):
                key = tuple(sorted(abs(lit) for lit in subset))
                value = (-1 if sum(lit<0 for lit in subset)%2 else 1) / 2**len(clause)
                coefficients[key] = coefficients.get(key, 0.) + value
    return coefficients.get((), 0.), float(np.sqrt(sum(v*v for key,v in coefficients.items() if key)))


def compare_sat(n=10, k=3, m=43, seed=42, depth=20, maxiter=100):
    clauses = generate_clauses(n, k, m, seed)
    mean, sigma = sat_moments(clauses)
    result = compare_energies(sat_energies(n, clauses), sigma, seed, depth, maxiter,
        mean=mean, metadata=dict(problem="max_sat", n=n, k=k, m=m, seed=seed,
                                density=m/n, clauses=clauses))
    result["estimated_minimum_below_zero"] = result["estimated_minimum"] < 0
    return result


def to_dimacs(n, clauses):
    """Serialize a CNF formula; optimization is unweighted Max-SAT."""
    return f'p cnf {n} {len(clauses)}\n' + ''.join(
        ' '.join(map(str, clause)) + ' 0\n' for clause in clauses)


def analytical_trajectory(energies, mean, sigma, target_energy, seed=42,
                          depth=20, maxiter=100):
    """Fit Gaussian angles to a supplied PHYSICAL target; evaluate exactly.

    This diagnostic accepts an oracle target but does not infer it internally.
    The optimizer only sees mean, sigma, target and previous model parameters.
    """
    from .core import GaussianModel, exact_layer, optimize_pair
    energies = np.asarray(energies, dtype=float)
    if energies.ndim != 1 or len(energies) == 0 or sigma <= 0 or depth < 0:
        raise ValueError("Require nonempty spectrum, positive sigma and nonnegative depth")
    if not np.isfinite(target_energy):
        raise ValueError("Target must be finite")
    centered = energies - mean
    target = target_energy - mean
    ground = energies == energies.min()  # integer SAT energies
    state = np.ones(len(energies), dtype=complex)/np.sqrt(len(energies))
    model = GaussianModel(sigma)
    records = []
    for layer in range(depth+1):
        beta = gamma = None
        nfev = 0
        if layer:
            beta,gamma,nfev = optimize_pair(
                lambda b,g: model.target_score(target,b,g), sigma,
                seed+layer, maxiter=maxiter)
            model.append(beta,gamma)
            state = exact_layer(state,centered,beta,gamma)
        probabilities = abs(state)**2
        if not np.isclose(probabilities.sum(),1.,atol=1e-10):
            raise RuntimeError("State normalization lost")
        records.append(dict(layer=layer,beta=beta,gamma=gamma,nfev=nfev,
            success=float(probabilities[ground].sum()),
            mean_energy=float(probabilities@energies),
            model_target_amplification=float(abs(model.amplitude(target))**2)))
    return records
