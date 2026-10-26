"""Run from repository root: python experiments/compare_depth.py."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/adaptive-grover-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adaptive_grover_qaoa.core import compare

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=8)
p.add_argument("--order", type=int, default=4)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--depth", type=int, default=20)
p.add_argument("--output", default="results")
a = p.parse_args()
data = compare(a.n, a.order, a.seed, a.depth)
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
stem = f"sk_n{a.n}_d{a.order}_seed{a.seed}"
(out / (stem + ".json")).write_text(json.dumps(data, indent=2) + "\n")
fig, ax = plt.subplots()
for name in ("constant", "analytical", "exact_greedy"):
    rows = [r for r in data["records"] if r["method"] == name]
    ax.plot([r["layer"] for r in rows], [r["success"] for r in rows], label=name)
    print(name, "final success:", rows[-1]["success"])
ax.set(xlabel="Layers", ylabel="Ground-state probability", title=f"SK HUBO: n={a.n}, D={a.order}, seed={a.seed}")
ax.legend()
fig.tight_layout()
fig.savefig(out / (stem + ".png"), dpi=160)
print("Exact / estimated minimum:", data["minimum"], data["estimated_minimum"])
