"""Small diagnostic ensemble, not a reproduction of the paper's 100-instance study."""
import json
from pathlib import Path
import numpy as np
import scipy
from adaptive_grover_qaoa.core import compare

runs = [compare(seed=seed) for seed in range(5)]
summary = {}
for method in ('constant', 'analytical', 'exact_greedy'):
    values = [r['success'] for run in runs for r in run['records']
              if r['method'] == method and r['layer'] == run['depth']]
    summary[method] = dict(mean=float(np.mean(values)), std=float(np.std(values, ddof=1)), values=values)
payload = dict(numpy=np.__version__, scipy=scipy.__version__, summary=summary, runs=runs)
Path('results/ensemble.json').write_text(json.dumps(payload, indent=2) + '\n')
print(json.dumps(summary, indent=2))
