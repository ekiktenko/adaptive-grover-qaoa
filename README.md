# Grover-mixer QAOA for HUBO

**Свои эксперименты:** [пошаговая инструкция на русском](docs/experiments-ru.md).
Параметры: [`configs/quick.json`](configs/quick.json) и [`configs/sweep.json`](configs/sweep.json).
Запуск: `python -m adaptive_grover_qaoa.runner configs/quick.json`.


Research prototype of GM-QAOA(a) from Kiktenko, Krendeleva and Fedorov,
*Physical Review A* **113**, 052451 (2026), DOI: 10.1103/p2t5-x8l5.
The sample-based adaptive extension and a configuration-driven experiment runner
are implemented; see the Russian guide above for parameter sweeps.

## Run

Validated with Python 3.12.14. From the repository root:

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements-repro.txt
.venv/bin/python -m pip install --no-build-isolation -e .
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python experiments/compare_depth.py
.venv/bin/python experiments/ensemble.py
```

`requirements-repro.txt` pins the three primary numerical packages used here;
it is not a complete transitive lockfile. The cloud base supplies setuptools.
No external service, GPU or credentials are required.

## Model and conventions

`src/adaptive_grover_qaoa/core.py` contains the problem generator, exact
statevector simulation, Gaussian model, and layerwise optimizer. Spin i is
`(-1)**bit_i`, with least-significant-bit indexing. All interactions of degrees
2 through D are included, with independent standard normal couplings.
No linear or constant terms are generated. The initial state is uniform.

The mixer is H_G = 2|sym><sym|, so its phase is exp(-2i beta).
Every layer applies the cost unitary first, then the mixer.
The exact simulator uses O(2**n) state storage, without dense unitary matrices.

The analytical optimizer only receives sigma = sqrt(sum(J**2)) and the EVT
target sigma * scipy.special.ndtri(2**(-n)). It never receives the enumerated
energies or exact minimum. Angles are fitted numerically using the analytical
Gaussian dynamics, not computed in a closed form. All previous angles remain
fixed when a new layer is fitted.

To avoid indexing ambiguities in printed Eqs. (20)-(21), derive the recursion
from Eq. (14). Write the scaled amplitude a(E) = sqrt(2**n)*psi(E) as
sum_j c_j exp(-i t_j E), initially c=[1], t=[0]. For a new layer:

1. Shift every frequency: t'_j = t_j + gamma.
2. Compute m = sum_j c_j exp(-sigma**2 * t'_j**2 / 2).
3. Keep the shifted terms and append the constant term
   (exp(-2i beta)-1)*m with frequency zero.

The characteristic function uses the **square of the accumulated frequency**.
The Gaussian-model score |a(E_target)|**2 is an amplification factor, not a
normalized success probability. The exact probability of all degenerate ground
states is measured separately. No claim that a finite HUBO spectrum is exactly
Gaussian or independent is made; the EVT target is a heuristic, not a bound.

## Comparison and limitations

- `constant`: beta=pi/2, gamma=-pi/E_target, Eq. (27).
- `analytical`: maximize the Gaussian-model target amplitude per layer.
- `exact_greedy`: maximize true ground-state probability per layer, using the
  exact minimum as an oracle benchmark, unavailable in a practical large task.

Both fitted methods use differential evolution with beta in [0, pi] and
**gamma*sigma in [-4, 4]**, population multiplier 12, maximum 100 generations,
tolerance 1e-8, and local polishing. The finite search range and budget are
explicit prototype choices; global optimality is not guaranteed. Identity is
included as a fallback. True success need not increase for analytical angles;
only the fitted model score has the greedy nondecrease guarantee.

The dense exact benchmark is a control, not a rigorous upper bound on other
angle schedules. `nfev` records DE objective calls (including polishing), not
wall time, shots, or the extra identity/final-score evaluations.

## Initial results

For n=8, D=4, seed=42, depth=20:

| Method | Ground-state probability |
|---|---:|
| Constant | 0.007071 |
| Analytical | 0.147648 |
| Exact greedy | 0.482125 |

Exact minimum: -29.124875; EVT target: -28.356742. Initial success: 1/256.
`results/sk_n8_d4_seed42.png` shows the depth curves; the JSON stores angles,
energies, probabilities, and objective-call counts.

Five further instances (seeds 0..4), at depth 20, give mean ± sample standard
deviation:

| Method | Probability |
|---|---:|
| Constant | 0.03848 ± 0.03223 |
| Analytical | 0.29058 ± 0.23496 |
| Exact greedy | 0.61536 ± 0.23924 |

These are diagnostic runs, not reproduction of the paper's 100-instance
statistics or evidence of quantum advantage. The broad spread motivates a
larger ensemble and controlled target-energy sensitivity experiments next.

Four tests cover the exact layer versus a dense unitary, multi-layer Gaussian
recursion versus Gauss-Hermite quadrature (including normalization), exact
spectral moments, and greedy score invariants.

## Random Max-3-SAT and Max-4-SAT

```bash
.venv/bin/python experiments/compare_sat.py
# Adjust ensemble size, depth and clause densities:
.venv/bin/python experiments/compare_sat.py --n 10 --seeds 5 --depth 20 --alpha3 4.25 --alpha4 9.75
```

`sat.py` generates uniform unique signed clauses with k distinct variables,
without a planted solution. Positive literals are true when their bit is 1.
Energy is the integer number of unsatisfied clauses; minimizing it is Max-SAT,
including unsatisfiable formulas. All optimal assignments count toward success.
The generator supports exact enumeration up to 20 variables (large runs can be
expensive). Formulas, parameters and full results are saved as JSON; individual
formulas are also saved as DIMACS CNF files. No SAT solver is required.

A violation projector is product(1 + sign(literal)*Z_variable)/2. Expanding and
aggregating equal Pauli strings gives exact uniform mean mu=m/2**k and variance
sum(nonconstant aggregated coefficients**2), without inspecting the minimum or
enumerating energies. Overlapping clauses mean a naive independent-binomial
variance is generally incorrect. The simulator and Gaussian model use centered
energy E-mu, with target sigma*Phi^-1(2^-n). Reported energies and estimated
minima are shifted back to the physical violation count. This shift only changes
a global phase. The constant-angle baseline uses the centered target as well.
A constant spectrum is handled with identity layers, success 1 and no fitting.

The default small diagnostic ensemble uses n=10, seeds 0..4, depth 20,
m=round(alpha*n): 42 clauses for k=3, 98 for k=4 (Python ties-to-even rounding).
These densities do not establish a finite-size satisfiability threshold.
Results after 20 layers:

| Method | Max-3-SAT mean P(optimum) | Max-4-SAT mean P(optimum) |
|---|---:|---:|
| Initial uniform sampling | 0.00781 | 0.00332 |
| Constant | 0.03408 | 0.01531 |
| Analytical | 0.07476 | 0.02379 |
| Exact greedy oracle | 0.75282 | 0.68323 |

Three of five Max-3-SAT formulas and four of five Max-4-SAT formulas are
satisfiable. The others have optimum one violation. Ensemble spread is large;
see `results/sat/comparison.png` for mean curves and min/max shading (not
confidence intervals), and `comparison.json` for individual results.

**Important limitation:** the unclipped Gaussian EVT target is negative in all
ten runs, although violation counts cannot be negative. The original model is
being tested unchanged after centering, not claimed to be calibrated for SAT.
Analytical angles amplify optimal states relative to uniform sampling on
average but are far behind the oracle. They also increase mean violations by
depth 20 relative to uniform sampling: the objectives P(optimum) and mean energy
are different. Model fitting alone guarantees neither true success monotonicity
nor lower expected energy. Separate sensitivity studies with improved targets
and alternative spectral models are needed to isolate these effects.

Nine tests now include signed-clause truth tables, generator reproducibility
and uniqueness, exact Pauli moments against enumeration, shift invariance, and
an unsatisfiable constant-spectrum formula. The original SK reference run is
unchanged by the refactoring.

## SAT target-energy ablation

```bash
.venv/bin/python experiments/sat_targets.py
```

This reads the saved formulas and optimizer budgets from
`results/sat/comparison.json`; run `experiments/compare_sat.py` first if absent.
It fits Gaussian analytical angles using three physical targets: the original
EVT estimate, max(0, EVT), and the enumerated true minimum (diagnostic oracle).
Every target is centered by subtracting the same exact uniform mean before
model fitting. The comparison changes no Gaussian moments or optimizer bounds,
uses paired seeds and verifies the entire original EVT success trajectory is
reproduced. The exact-greedy reference is reused from the saved paired run.
Targets, angle schedules and evaluated trajectories are saved under
`results/sat/targets/`, together with the comparison figure.

Mean ground-state success after 20 layers (same five instances per k):

| Target/method | Max-3-SAT | Max-4-SAT |
|---|---:|---:|
| EVT | 0.07476 | 0.02379 |
| EVT clipped to zero | 0.23064 | 0.42341 |
| True minimum, Gaussian dynamics | 0.26098 | 0.51422 |
| Exact greedy oracle | 0.75282 | 0.68323 |

For satisfiable instances the clipped and true-minimum targets are identical;
only unsatisfiable instances distinguish these two conditions. Improvement
is an ensemble result, not guaranteed instancewise: Max-3-SAT seed 1 has
success 0.09695 (EVT), 0.08526 (clipped), 0.07776 (true target) at depth 20.
Using the true target is not an implementable estimator for large unknown
instances and this result does not establish a sample-based procedure yet.

For Max-3-SAT, Gaussian true-target success peaks at 0.49710 at layer 11,
then falls to 0.26098. For Max-4-SAT it peaks at 0.53294 at layer 18.
These peaks are post-hoc diagnostics, not validated stopping rules.
A true target does not fix spectral-model mismatch or heuristic fitting.

Final mean violation counts, ordered EVT/clipped/true-target/exact-greedy:
Max-3-SAT: 6.22260 / 5.16503 / 5.15553 / 2.54395 (initial 5.25);
Max-4-SAT: 8.58875 / 5.05201 / 4.71130 / 3.43626 (initial 6.125).
Thus clipping helps both reported metrics on average, but optimal-state
probability and expected violation count remain different objectives.
The paired experiment isolates target changes within this fixed Gaussian
model; it does not by itself isolate all remaining error into the model,
because the finite numerical optimization is also heuristic.

Ten tests pass, including a physical-target energy-shift-invariance test.

## Sample-based adaptive algorithm (100-shot budget)

```bash
.venv/bin/python experiments/adaptive_sat.py
# Shorter sampling study, same algorithm:
.venv/bin/python experiments/adaptive_sat.py --repeats 10 --budget 100 --warmup 10 --depth 20
```

`adaptive.py` separates the sample-only controller from Gaussian angle fitting
and the exact simulation backend. The initial target is the **unclipped** paper
EVT estimate. Fit p=20 layers from scratch and draw ten independent samples.
Replace the target unconditionally with the minimum observed energy, even if
it is above the EVT estimate. Refit from scratch before shot 11. Subsequently
refit only after a sample has strictly lower energy than the current target;
equal or higher energies leave the circuit unchanged. Every sample is from a
fresh circuit initialized uniformly. Stop at exactly 100 total shots, including
warmup, returning the best measured assignment. Continue sampling after finding
zero. No fit is performed after the last shot because no further circuit uses it.

The controller receives no exact minimum or enumerated energy spectrum. It
computes sampled energies directly from the clauses through an evaluator.
Mean and variance are computed from clause-to-Pauli coefficients as before.
The simulation backend enumerates energies solely to execute the exact
statevector circuit; true optima are used outside the controller for scoring.
This is a noise-free classical simulation of quantum sampling, not a hardware
run or a computational speedup claim. Each shot uses p cost/mixer layers;
statevector work and classical angle-fitting cost are separate from shot count.

The default study reuses the ten saved formulas: n=10, k=3 with m=42 or k=4
with m=98, five instances each. For every instance, 100 stochastic repetitions
are run per method with 100 shots each. Fixed EVT and uniform baselines receive
the same total budget. They use paired RNG streams; the adaptive and fixed
methods share identical first-ten samples. The fixed baseline keeps the original
EVT angles for the whole budget (it is not the clipped-target baseline).

| Method | Max-3-SAT: optimum found within 100 shots | Max-4-SAT |
|---|---:|---:|
| Adaptive | 0.984 (492/500) | 0.926 (463/500) |
| Fixed EVT | 0.976 (488/500) | 0.784 (392/500) |
| Uniform random sampling | 0.450 (225/500) | 0.268 (134/500) |

These are **best-of-100 success rates**, not single-shot circuit probabilities.
Mean excess violations above the true optimum are respectively
0.016 / 0.030 / 0.650 for k=3 and 0.074 / 0.246 / 1.020 for k=4
(adaptive / fixed / uniform). The small difference for k=3 should not be
interpreted as a robust advantage: only five problem instances are represented.
Repeated sampling measures randomness on those instances, not generalization
across a broad formula ensemble.

Adaptive mean total fit counts (including the initial fit) are 2.940 for k=3
and 3.674 for k=4, versus one for fixed EVT and zero for uniform. Mean logical
DE objective-evaluation counts per independent run are about 38,417 and 46,700,
versus about 14,163 for fixed EVT. These exclude the extra score checks in
`optimize_pair`, as in earlier experiments. Identical deterministic fits and
statevector distributions are cached across repetitions within each instance
to accelerate the study. Counts charge each trajectory for all logically needed
fits, including cache hits; they are not measured actual repeated CPU work.
The cache key is target energy and its instance-specific depth, moments and
optimizer settings are fixed; cache entries never depend on a true minimum.

Artifacts: `results/sat/adaptive/comparison.png`, `summary.json` and one JSON
file per problem instance containing every measured assignment, energy,
best-so-far value, target change and fit count. Integer assignment bit i-1 is
variable x_i; standard printed binary strings put x_1 on the right.
`example_trace.json` is a post-hoc illustrative Max-4-SAT trajectory (seed 3,
repeat 96): EVT -1.1953 -> 8 after shot 10 -> 7 -> 6 -> 5 -> 4 -> 3 -> 2 -> 0
at shot 45. It was selected to show multiple updates and is not representative
of average behavior.

Fourteen tests pass. New tests cover budget accounting, unconditional warmup
replacement, strict improvements versus ties, final-shot updates without unused
fits, continued sampling after zero, and agreement of fitted circuit sampling
with the previous exact reference. All 3,000 saved trial histories were also
checked for exact budgets, consistent best-so-far values, strict post-warmup
updates and matching paired warmup samples.
