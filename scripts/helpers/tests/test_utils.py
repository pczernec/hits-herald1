from unittest import TestCase
import numpy as np

from scripts.helpers.utils import ExponentialAverageCalculator, RunningStd

np.random.seed(1)



class TestExponentialAverageCalculator(TestCase):
    gauss_multiplier = 1
    shift = 3.
    z = np.random.randn(10000) * gauss_multiplier + shift
    mod_signal = np.linspace(0, 5 * np.pi, 10000)
    mod_signal = np.sin(mod_signal)
    # z = z * mod_signal
    exp_avg_calc = ExponentialAverageCalculator(0.999)

    def test_if_correct_average_simple(self):
        avg = [self.exp_avg_calc(elem, i) for i, elem in enumerate(self.z)]

        self.assertAlmostEqual(avg[-1], self.shift, places=1)


class TestStandardDeviationCalculator(TestCase):
    z1 = np.random.normal(0, 1, 1000)
    z2 = np.random.normal(1, 2, 1000)
    z3 = np.random.normal(-2, 20, 1000)
    z = np.concatenate([z1, z2, z3], axis=0)

    exp_avg_calc = ExponentialAverageCalculator(0.999)
    std_calc = RunningStd(alpha=0.9)

    def assertBetween(self, value, min_v, max_v):
        """Fail if value is not between min and max (inclusive)."""
        self.assertGreaterEqual(value, min_v)
        self.assertLessEqual(value, max_v)

    def test_if_correct_std(self):
        avg = [self.exp_avg_calc(elem, i) for i, elem in enumerate(self.z)]
        avg_of_stds = [
            self.std_calc(z_now, avg_now, i) for i, (z_now, avg_now) in enumerate(zip(self.z, avg))
        ]
        self.assertBetween(avg_of_stds[999], 0.95, 1.05)
        self.assertBetween(avg_of_stds[1999], 1.5, 2)
        self.assertBetween(avg_of_stds[2999], 10, 20)
