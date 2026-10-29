import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from adaptive_grover_qaoa.runner import plan,validate_config,run_case,run_grid,exact_greedy_fit
from adaptive_grover_qaoa.core import compare_energies
from adaptive_grover_qaoa.sat import sat_moments,sat_energies,generate_clauses


def config():
    return dict(qubits=[5],clause_sizes=[3],densities={'3':2.},depths=[2],instance_seeds=[0],
        methods=['adaptive','fixed_evt','exact_greedy','uniform'],repeats=2,shots=12,warmup=3,
        optimizer_maxiter=5,sampling_seed=2026,save_histories=True)


class RunnerTests(unittest.TestCase):
    def test_grid_and_invalid_config(self):
        c=config()
        c['depths']=[1,2]
        c['instance_seeds']=[0,1]
        self.assertEqual(len(plan(c)),4)
        for key,value in [('qubits',[21]),('methods',['typo']),('warmup',13),('depths',[-1]),('repeats',0),('shots',1.2)]:
            bad=copy.deepcopy(c)
            bad[key]=value
            with self.assertRaises(ValueError):validate_config(bad)
        regularized=copy.deepcopy(c)
        regularized['energy_penalty']=2.0
        validate_config(regularized)
        regularized['energy_penalty']=-1
        with self.assertRaises(ValueError):validate_config(regularized)
        bad=copy.deepcopy(c)
        bad['typo']=1
        with self.assertRaises(ValueError):validate_config(bad)

    def test_pairing_budgets_and_reproducibility(self):
        c=config()
        case=plan(c)[0]
        a=run_case(case,c)
        b=run_case(case,c)
        self.assertEqual(a['trials'],b['trials'])
        for trial in a['trials']:
            self.assertEqual(trial['shots'],12)
            self.assertEqual(len(trial['history']),12)
        paired={t['method']:t for t in a['trials'] if t['repeat']==0}
        self.assertEqual([r['assignment'] for r in paired['adaptive']['history'][:3]],
                         [r['assignment'] for r in paired['fixed_evt']['history'][:3]])
        c['depths']=[3]
        other=run_case(plan(c)[0],c)
        self.assertEqual(a['clauses'],other['clauses'])
        self.assertEqual([t for t in a['trials'] if t['method']=='uniform'],
                         [t for t in other['trials'] if t['method']=='uniform'])

    def test_greedy_reference(self):
        e=sat_energies(5,generate_clauses(5,3,10,0))
        mean,sigma=sat_moments(generate_clauses(5,3,10,0))
        fitted=exact_greedy_fit(e,mean,sigma,2,0,5)
        reference=compare_energies(e,sigma,seed=0,depth=2,maxiter=5,mean=mean)
        expected=[r for r in reference['records'] if r['method']=='exact_greedy'][-1]['success']
        self.assertAlmostEqual(float(fitted['probabilities'][e==e.min()].sum()),expected)

    def test_artifacts_and_overwrite_protection(self):
        c=config()
        c.update(methods=['uniform'],depths=[0],save_histories=False)
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/'run'
            rows=run_grid(c,output)
            self.assertEqual(len(rows),1)
            self.assertTrue((output/'aggregate.csv').exists())
            self.assertTrue((output/'depth_comparison.png').exists())
            result=json.loads(next(output.glob('*/result.json')).read_text())
            self.assertNotIn('history',result['trials'][0])
            self.assertEqual(result['trials'][0]['shots'],12)
            with self.assertRaises(FileExistsError):run_grid(c,output)
