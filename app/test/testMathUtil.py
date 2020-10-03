import sys
import unittest

import torch

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

        from scipy.misc import derivative
        def numeric_validation(z=3.1,
                               v=2.1):
            numeric_diff = derivative(lambda v: mp.besselk(v, z), v, dx=1e-5)
            calculated = Loss_functions.d_kv_2_order(np.array([z]), np.array([v]))[0]
            return mp.log(numeric_diff / calculated)

        vs = -np.arange(100) + 0.01 + 50
        z = 5.1

        result = np.array([numeric_validation(z, v) for v in vs])
        print(result)
        eps = 1e-3
        print(result[abs(result) > eps])
        print((abs(result) > eps).sum())
        assert all(abs(result) < eps)

    def test_meijerg(self):
        import mpmath as mp
        from ts_prediction.math_util import Loss_functions
        from scipy.misc import derivative
        def numeric_validation(z=3.1,
                               v=2.1):
            numeric_diff = derivative(lambda v: mp.besselk(v, z), v, dx=1e-10)
            calculated = Loss_functions.d_kv_2_order_plus(v, z)
            return mp.log(numeric_diff / calculated)

        vs = np.arange(100) / 10.0 + 0.01
        z = 5.0
        result = np.array([numeric_validation(z, v) for v in vs])
        print(result)

    def test_nanity(self):
        # indices: (tensor([0, 0, 0, 0]), tensor([241, 843, 932, 1158]))
        # target: tensor([0.0011, 0.0021, -0.0044, -0.0408], dtype=torch.float64)
        # mu: tensor([-0.1098, -0.1621, 0.1961, -0.2194], grad_fn= < IndexBackward >)
        # gamma: tensor([-0.9623, -1.0674, -1.0342, -0.9292], grad_fn= < IndexBackward >)
        # pc: tensor([0.3262, 0.2854, 0.2275, 0.4170], grad_fn= < IndexBackward >)
        # chi: tensor([0.5226, 0.3145, 0.4377, 0.4946], grad_fn= < IndexBackward >)
        # ld: tensor([0.2416, 0.2060, 0.3243, 0.2251], grad_fn= < IndexBackward >)
        target = torch.tensor([0.0011, 0.0021, -0.0044, -0.0408], dtype=torch.float64, requires_grad=True)
        mu = torch.tensor([-0.1098, -0.1621, 0.1961, -0.2194], requires_grad=True)
        gamma = torch.tensor([-0.9623, -1.0674, -1.0342, -0.9292], requires_grad=True)
        pc = torch.tensor([0.3262, 0.2854, 0.2275, 0.4170], requires_grad=True)
        chi = torch.tensor([0.5226, 0.3145, 0.4377, 0.4946], requires_grad=True)
        ld = torch.tensor([0.2416, 0.2060, 0.3243, 0.2251], requires_grad=True)

        res = Loss_functions.nll_dghb6(target, mu, gamma, pc, chi, ld).sum()
        from torch.autograd import grad
        res = grad(res, (target, mu, gamma, pc, chi, ld))

        print(res)


if __name__ == '__main__':
    unittest.main()
