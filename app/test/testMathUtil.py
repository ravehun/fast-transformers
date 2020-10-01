import sys
import unittest

sys.path.append("..")

from ts_prediction.math_util import *
from mpmath import *

mp.dps = 20;
mp.pretty = True
import numpy as np
import scipy
from ts_prediction.math_util import Loss_functions


class TestMathUtil(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestMathUtil, self).__init__(*args, **kwargs)
        # length = 1500
        # # for _ in range(10):
        # d_kv_2_order(np.array([2.1] * length), np.array([3.1] * length))

    def test_d_kv_2_order_p(self):
        length = 1500
        z = Loss_functions.d_kv_2_order(np.array([2.0] * length), np.linspace(1.000001, 3.0000000000, length))

    def test_pfq(self):
        from ts_prediction.libarb import hypergeometric_pfq
        t = 1500
        a = np.array([[2, 2, 1]]).repeat(t, 0)
        b = np.array([[2, 2, 3, 3]]).repeat(t, 0)
        z = np.array(4).repeat(t)
        res = hypergeometric_pfq(a, b, z)

    def test_hyper(self):
        from mpmath import hyper
        for _ in range(1500):
            hyper([1, 1, 1.2], [2, 2, 2, 1], 4)

    def test_hyper2f3(self):
        from mpmath import hyp2f3
        for _ in range(1500):
            hyp2f3(2, 2, 2, 2, 2, 4)

    def testDigamma_mp(self):
        [digamma(1.0) for _ in range(1500)]

    def testDigamma_scipy(self):
        scipy.special.digamma(np.ones(1500))
        scipy.special.gamma(np.ones(1500))

    def testHypergeometric_pfq(self):
        from ts_prediction.libarb import hypergeometric_pfq
        a = np.array([[1, 2, 1]])
        b = np.array([[2, 2, 2, 1.1]])
        z = np.array([4])
        actual = hypergeometric_pfq(a, b, z)[0]
        expected = 2.40323872104276992090
        self.assertAlmostEqual(expected, actual)

    def test_d_kv_2_order(self):
        from ts_prediction.math_util import Loss_functions
        import scipy.integrate as integrate
        actual = integrate.quad(lambda x: Loss_functions.d_kv_2_order(np.array([2.0]), np.array([x]))[0], 1.1, 3.1)
        expected = 0.57684907204712798645
        self.assertAlmostEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
