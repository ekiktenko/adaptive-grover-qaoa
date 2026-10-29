import unittest
import numpy as np
from numpy.polynomial.hermite import hermgauss
from adaptive_grover_qaoa.core import GaussianModel
from adaptive_grover_qaoa.adaptive import fit_angles


class RegularizationTests(unittest.TestCase):
    def test_mean_against_independent_quadrature(self):
        model=GaussianModel(1.7)
        x,w=hermgauss(140)
        e=np.sqrt(2)*model.sigma*x
        w=w/np.sqrt(np.pi)
        for b,g in [(0.8,.4),(1.9,-.6),(.3,.8),(2.1,.2)]:
            c,t=model.candidate(b,g)
            values=np.exp(-1j*e[:,None]*t)@c
            expected=float(np.dot(w,e*abs(values)**2))
            self.assertAlmostEqual(model.centered_mean_energy(c,t),expected,places=11)
            target=-3.2
            expected_score=float(abs(np.dot(c,np.exp(-1j*t*target)))**2)-2.5*expected/model.sigma
            self.assertAlmostEqual(model.target_score(target,b,g,energy_penalty=2.5),expected_score,places=11)
            model.append(b,g)

    def test_zero_penalty_is_exactly_backward_compatible(self):
        a=fit_angles(4,2,-1,depth=3,maxiter=15)
        b=fit_angles(4,2,-1,depth=3,maxiter=15,energy_penalty=0.)
        self.assertEqual(a,b)
        for bad in (-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):fit_angles(4,2,-1,energy_penalty=bad)

    def test_dimensionless_score_scale_and_shift_invariance(self):
        a=fit_angles(4,2,-1,depth=3,maxiter=15,energy_penalty=3.)
        b=fit_angles(19,6,4,depth=3,maxiter=15,energy_penalty=3.)  # E' = 3E+7
        np.testing.assert_allclose(np.array(a['angles'])[:,0],np.array(b['angles'])[:,0],atol=1e-5)
        np.testing.assert_allclose(np.array(a['angles'])[:,1]/3,np.array(b['angles'])[:,1],atol=1e-5)
        self.assertAlmostEqual(3*a['model_mean_energy']+7,b['model_mean_energy'],places=5)
