import unittest
import numpy as np
from numpy.polynomial.hermite import hermgauss
from adaptive_grover_qaoa.core import GaussianModel, exact_layer, sk_problem, compare


class CoreTests(unittest.TestCase):
    def test_exact_layer_against_dense_unitary(self):
        rng = np.random.default_rng(9)
        state = rng.normal(size=8) + 1j*rng.normal(size=8)
        state /= np.linalg.norm(state)
        energies = rng.normal(size=8)
        sym = np.ones(8)/np.sqrt(8)
        unitary = np.eye(8) + np.expm1(-2j*.7)*np.outer(sym, sym)
        expected = unitary @ (np.exp(-1j*.3*energies)*state)
        np.testing.assert_allclose(exact_layer(state, energies, .7, .3), expected, atol=1e-14)
        self.assertAlmostEqual(np.linalg.norm(expected), 1.)

    def test_gaussian_recursion_against_quadrature(self):
        sigma = 1.7
        x, w = hermgauss(120)
        energies, w = np.sqrt(2)*sigma*x, w/np.sqrt(np.pi)
        values = np.ones_like(x, dtype=complex)
        target, target_value = -3.2, 1+0j
        model = GaussianModel(sigma)
        for beta, gamma in [(0.8, .4), (1.9, -.6), (.3, .8), (2.1, .2)]:
            phased = values*np.exp(-1j*gamma*energies)
            offset = np.expm1(-2j*beta)*np.dot(w, phased)
            values = phased + offset
            target_value = np.exp(-1j*gamma*target)*target_value + offset
            model.append(beta, gamma)
            np.testing.assert_allclose(model.amplitude(energies), values, atol=1e-12)
            np.testing.assert_allclose(model.amplitude(target), target_value, atol=1e-12)
            self.assertAlmostEqual(float(np.dot(w, abs(values)**2)), 1., places=11)

    def test_hubo_moments(self):
        energies, sigma = sk_problem(8, 4, 42)
        self.assertAlmostEqual(float(energies.mean()), 0., places=12)
        self.assertAlmostEqual(float(energies.var()), sigma**2, places=10)

    def test_greedy_and_probabilities(self):
        data = compare(n=5, order=3, depth=3, maxiter=20)
        for method in ("constant", "analytical", "exact_greedy"):
            rows = [r for r in data['records'] if r['method'] == method]
            self.assertTrue(all(0 <= r['success'] <= 1+1e-12 for r in rows))
            if method == 'exact_greedy':
                self.assertTrue(np.all(np.diff([r['success'] for r in rows]) >= -1e-12))
            if method == 'analytical':
                self.assertTrue(np.all(np.diff([r['model_target_amplification'] for r in rows]) >= -1e-10))

if __name__ == '__main__':
    unittest.main()
