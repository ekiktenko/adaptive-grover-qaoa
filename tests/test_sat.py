import unittest
import numpy as np
from adaptive_grover_qaoa.sat import generate_clauses, sat_energies, sat_moments, compare_sat, analytical_trajectory
from adaptive_grover_qaoa.core import compare_energies


class SatTests(unittest.TestCase):
    def test_truth_table(self):
        # (x1 OR NOT x2 OR x3) is false only at (0,1,0).
        energy = sat_energies(3, [(1,-2,3)])
        expected = np.zeros(8, dtype=int)
        expected[2] = 1
        np.testing.assert_array_equal(energy, expected)
        # Contradictory unit clauses: every assignment violates exactly one.
        np.testing.assert_array_equal(sat_energies(1, [(1,),(-1,)]), [1,1])

    def test_four_literal_clause(self):
        energy = sat_energies(4, [(1,2,3,-4)])
        self.assertEqual(energy.sum(), 1)
        self.assertEqual(energy[8], 1)

    def test_generation_and_moments(self):
        for k,m in [(3,43),(4,98)]:
            clauses = generate_clauses(10,k,m,42)
            self.assertEqual(clauses, generate_clauses(10,k,m,42))
            self.assertEqual(len(set(clauses)), m)
            self.assertTrue(all(len(set(map(abs,c))) == k for c in clauses))
            energies = sat_energies(10,clauses)
            mean,sigma = sat_moments(clauses)
            self.assertAlmostEqual(mean, m/2**k)
            self.assertAlmostEqual(energies.mean(), mean)
            self.assertAlmostEqual(energies.var(), sigma**2)
        self.assertEqual(len(set(generate_clauses(3,3,8))), 8)
        with self.assertRaises(ValueError):
            generate_clauses(3,3,9)

    def test_energy_shift_invariance(self):
        energies = sat_energies(5, generate_clauses(5,3,15))
        mean,sigma = sat_moments(generate_clauses(5,3,15))
        a = compare_energies(energies,sigma,mean=mean,depth=2,maxiter=10)
        b = compare_energies(energies+7,sigma,mean=mean+7,depth=2,maxiter=10)
        np.testing.assert_allclose([r['success'] for r in a['records']],
                                   [r['success'] for r in b['records']], atol=1e-12)
        for x,y in zip(a['records'],b['records']):
            self.assertAlmostEqual(y['mean_energy']-x['mean_energy'],7)

    def test_unsatisfiable_maxsat(self):
        # All eight 3-variable clauses: optimum is one violation, all states optimal.
        result = compare_sat(n=3,k=3,m=8,depth=0)
        self.assertEqual(result['minimum'],1)
        self.assertEqual(result['ground_degeneracy'],8)
        self.assertTrue(all(abs(r['success']-1)<1e-12 for r in result['records']))


class TargetTests(unittest.TestCase):
    def test_physical_target_shift_invariance(self):
        clauses = generate_clauses(5,3,15,7)
        energies = sat_energies(5,clauses)
        mean,sigma = sat_moments(clauses)
        # Physical zero target must be shifted along with spectrum and mean.
        a = analytical_trajectory(energies,mean,sigma,0.,depth=3,maxiter=15)
        b = analytical_trajectory(energies+9,mean+9,sigma,9.,depth=3,maxiter=15)
        np.testing.assert_allclose([r['success'] for r in a], [r['success'] for r in b],atol=1e-12)
        np.testing.assert_allclose([r['mean_energy']+9 for r in a], [r['mean_energy'] for r in b],atol=1e-12)
        self.assertTrue(all(np.isfinite(r['model_target_amplification']) for r in a))
        scores=[r['model_target_amplification'] for r in a]
        self.assertTrue(np.all(np.diff(scores)>=-1e-10))
