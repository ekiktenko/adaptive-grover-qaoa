import unittest
import numpy as np
from adaptive_grover_qaoa.adaptive import adaptive_search, fit_angles, circuit_probabilities
from adaptive_grover_qaoa.sat import generate_clauses,sat_moments,sat_energies
from adaptive_grover_qaoa.core import compare_energies

class ScriptedRandom:
    def __init__(self,assignments):
        self.assignments=iter(assignments)
    def choice(self,*args,**kwargs):
        return next(self.assignments)

class AdaptiveTests(unittest.TestCase):
    def test_warmup_strict_improvement_and_budget(self):
        targets=[]
        def prepare(target):
            targets.append(target)
            return dict(probabilities=np.ones(8)/8,nfev=3)
        # Initial -2 replaced by 4 despite worsening; ties do not refit.
        result=adaptive_search(-2,prepare,lambda x:x,ScriptedRandom([5,4,6,4,3,3,2]),budget=7,warmup=2)
        self.assertEqual(targets,[-2,4,3])
        self.assertEqual([u['target'] for u in result['updates']],[4,3,2])
        self.assertEqual(result['best_energy'],2)
        self.assertEqual(len(result['history']),7)
        self.assertEqual(result['nfev'],9)
        self.assertEqual([f['before_shot'] for f in result['fits']],[1,3,6])
        # Last-shot improvement incurs no unused fit.
        self.assertEqual(result['final_target'],2)

    def test_all_budget_in_warmup(self):
        calls=[]
        def prepare(t):
            calls.append(t)
            return dict(probabilities=[1.],nfev=0)
        result=adaptive_search(-1,prepare,lambda x:0,np.random.default_rng(0),budget=10,warmup=10)
        self.assertEqual(len(calls),1)
        self.assertEqual(result['shots'],10)
        self.assertEqual(result['best_energy'],0)
        with self.assertRaises(ValueError):
            adaptive_search(0,prepare,lambda x:0,np.random.default_rng(0),budget=9,warmup=10)

    def test_fitted_circuit_matches_reference(self):
        clauses=generate_clauses(5,3,15)
        energies=sat_energies(5,clauses)
        mean,sigma=sat_moments(clauses)
        from scipy.special import ndtri
        target=mean+sigma*ndtri(1/32)
        fitted=fit_angles(mean,sigma,target,depth=3,maxiter=15)
        probabilities=circuit_probabilities(energies,mean,fitted['angles'])
        reference=compare_energies(energies,sigma,mean=mean,depth=3,maxiter=15)
        expected=[r for r in reference['records'] if r['method']=='analytical'][-1]['success']
        self.assertAlmostEqual(float(probabilities[energies==energies.min()].sum()),expected)

    def test_zero_does_not_stop_sampling(self):
        fits=[]
        def prepare(target):
            fits.append(target)
            return dict(probabilities=[.5,.5],nfev=1)
        result=adaptive_search(-1,prepare,lambda assignment:assignment,
            ScriptedRandom([0,1,0,0,1]),budget=5,warmup=2)
        self.assertEqual(fits,[-1,0])
        self.assertEqual(result['shots'],5)
        self.assertEqual(result['best_assignment'],0)
        self.assertEqual(len(result['updates']),1)
