"""Conventions follow PRA 113, 052451 (2026), Eqs. (6), (10), (14).

Model amplitudes are rescaled by sqrt(2**n): the initial amplitude is 1.
Represent a(E) as sum_j c_j exp(-i t_j E). Gaussian expectations use
exp(-sigma**2 * t_j**2 / 2), with squared SUMS of accumulated angles.
"""
from itertools import combinations
import numpy as np
from scipy.optimize import differential_evolution
from scipy.special import ndtri


def sk_problem(n=8, order=4, seed=42):
    """All interactions of degrees 2..order; independent N(0,1) couplings."""
    if not 2 <= order <= n or n > 20:
        raise ValueError("Require 2 <= order <= n <= 20 for exact enumeration")
    rng = np.random.default_rng(seed)
    terms = [t for d in range(2, order + 1) for t in combinations(range(n), d)]
    couplings = rng.normal(size=len(terms))
    spins = 1 - 2 * ((np.arange(2**n)[:, None] >> np.arange(n)) & 1)
    energies = sum(j * np.prod(spins[:, t], axis=1) for t, j in zip(terms, couplings))
    return energies, float(np.linalg.norm(couplings))


def exact_layer(state, energies, beta, gamma):
    v = state * np.exp(-1j * gamma * energies)
    return v + np.expm1(-2j * beta) * np.mean(v)


class GaussianModel:
    def __init__(self, sigma):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.sigma = float(sigma)
        self.coefficients = np.array([1 + 0j])
        self.frequencies = np.array([0.])

    def amplitude(self, energy):
        return np.exp(-1j * np.asarray(energy)[..., None] * self.frequencies) @ self.coefficients

    def candidate(self, beta, gamma):
        shifted = self.frequencies + gamma
        mean = np.dot(self.coefficients, np.exp(-0.5 * (self.sigma * shifted)**2))
        return np.r_[self.coefficients, np.expm1(-2j * beta) * mean], np.r_[shifted, 0.]

    def centered_mean_energy(self, coefficients=None, frequencies=None):
        """Integral E |a(E)|² f(E) dE for E ~ N(0,sigma²).

        E[E exp(-i t E)] = -i sigma² t exp(-sigma² t²/2).
        The model state is normalized under f; a(E)=sqrt(2**n)*psi(E).
        """
        c = self.coefficients if coefficients is None else coefficients
        t = self.frequencies if frequencies is None else frequencies
        delta = t[:, None] - t[None, :]
        moment = -1j * self.sigma**2 * delta * np.exp(-0.5*(self.sigma*delta)**2)
        return float(np.sum(c[:, None]*c.conj()[None, :]*moment).real)

    def target_score(self, energy, beta, gamma, energy_penalty=0.):
        """Maximize target amplification minus penalty * <E_centered>/sigma.

        Amplification is NOT ground-state probability. Penalty is dimensionless;
        zero reproduces the original objective, including its numerical path.
        """
        if not np.isfinite(energy_penalty) or energy_penalty < 0:
            raise ValueError("energy_penalty must be finite and nonnegative")
        c, t = self.candidate(beta, gamma)
        score = float(abs(np.dot(c, np.exp(-1j * t * energy)))**2)
        if energy_penalty:
            score -= energy_penalty * self.centered_mean_energy(c,t) / self.sigma
        return score

    def append(self, beta, gamma):
        self.coefficients, self.frequencies = self.candidate(beta, gamma)


def optimize_pair(score, sigma, seed, gamma_bound=4., maxiter=100):
    """Search beta in [0,pi], dimensionless gamma*sigma in [-4,4].

    Finite search bounds and DE budget are experimental choices, not global
    optimality guarantees. Include identity so greedy scores cannot decrease.
    """
    result = differential_evolution(
        lambda x: -score(x[0], x[1] / sigma),
        [(0., np.pi), (-gamma_bound, gamma_bound)], seed=seed,
        maxiter=maxiter, popsize=12, tol=1e-8, polish=True)
    beta, gamma = float(result.x[0]), float(result.x[1] / sigma)
    if score(0., 0.) > score(beta, gamma):
        return 0., 0., int(result.nfev)
    return beta, gamma, int(result.nfev)


def compare(n=8, order=4, seed=42, depth=20, maxiter=100):
    energies, sigma = sk_problem(n, order, seed)
    return compare_energies(energies, sigma, seed=seed, depth=depth, maxiter=maxiter,
                            metadata=dict(n=n, order=order, seed=seed))


def compare_energies(energies, sigma, seed=42, depth=20, maxiter=100,
                     mean=0., metadata=None):
    """Optimize using centered energies; report physical (uncentered) energies.

    sigma and mean must be obtained independently of the minimum. Gaussian
    EVT targets are deliberately not clipped to a known physical lower bound.
    """
    energies = np.asarray(energies, dtype=float)
    n = int(np.log2(len(energies)))
    if len(energies) != 2**n or sigma < 0 or depth < 0:
        raise ValueError("Require power-of-two spectrum, nonnegative sigma and depth")
    if sigma == 0:
        if not np.allclose(energies, mean, rtol=0, atol=1e-12):
            raise ValueError("Zero variance requires a constant spectrum equal to mean")
        return dict(**(metadata or dict(n=n, seed=seed)), depth=depth, sigma=0.,
            energy_mean=mean, minimum=float(mean), estimated_minimum=float(mean),
            centered_target=0., ground_degeneracy=len(energies), optimizer=None,
            records=[dict(method=method, layer=k, beta=0. if k else None,
                gamma=0. if k else None, success=1., mean_energy=float(mean),
                nfev=0, model_target_amplification=1.)
                for method in ("constant", "analytical", "exact_greedy")
                for k in range(depth+1)])
    physical_energies = energies
    energies = energies - mean
    minimum = float(energies.min())
    ground = np.isclose(energies, minimum, rtol=0, atol=1e-10)
    target = float(sigma * ndtri(2.**(-n)))
    records = []
    for method in ("constant", "analytical", "exact_greedy"):
        state = np.ones(2**n, dtype=complex) / np.sqrt(2**n)
        model = GaussianModel(sigma)
        for k in range(depth + 1):
            beta = gamma = None
            nfev = 0
            if k:
                if method == "constant":
                    beta, gamma = np.pi/2, -np.pi/target
                else:
                    if method == "analytical":
                        score = lambda b, g: model.target_score(target, b, g)
                    else:
                        score = lambda b, g: float(np.sum(abs(exact_layer(state, energies, b, g)[ground])**2))
                    beta, gamma, nfev = optimize_pair(score, sigma, seed + k, maxiter=maxiter)
                state = exact_layer(state, energies, beta, gamma)
                model.append(beta, gamma)
            probabilities = abs(state)**2
            if not np.isclose(probabilities.sum(), 1., atol=1e-10):
                raise RuntimeError("State normalization lost")
            records.append(dict(method=method, layer=k, beta=beta, gamma=gamma,
                success=float(probabilities[ground].sum()),
                mean_energy=float(probabilities @ physical_energies), nfev=nfev,
                model_target_amplification=float(abs(model.amplitude(target))**2)))
    return dict(**(metadata or dict(n=n, seed=seed)), depth=depth, sigma=sigma,
                energy_mean=mean, minimum=minimum + mean, estimated_minimum=target + mean,
                centered_target=target, ground_degeneracy=int(ground.sum()),
                optimizer=dict(maxiter=maxiter, popsize=12, gamma_sigma_bounds=[-4,4]), records=records)
