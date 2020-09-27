import unittest

import numpy as np
import torch
from common_utils import TestUtils
from loss import Loss_functions


class MaskLoss(unittest.TestCase):

    def test_t_log_loss(self):
        from scipy.stats import t
        x = np.linspace(-1, 1)
        z = np.linspace(2, 4)
        df = 1.3
        loc = 1.3
        scale = 1
        # m = t(df,loc,scale)
        import scipy
        lg = scipy.special._ufuncs.loggamma
        expected = np.array([lg(i) for i in z])
        actual = torch.lgamma(torch.tensor(z)).numpy()
        self._almost_equals(actual, expected, 1e-12)

        expected = t.logpdf(x, df, loc, scale)
        actual = Loss_functions.nll_logt(torch.tensor(x), torch.tensor(df), torch.tensor(loc),
                                         torch.tensor(scale)).numpy()
        self._almost_equals(actual, -expected, 1e-6)

    def _almost_equals(self, x, y, eps=1e-3):
        res = all(abs(x - y) < eps)
        if not res:
            print(f"expect {x}\n, actual {y}")
        self.assertTrue(res)

    def test_MaskedAPE(self):
        eps = 1e-4
        from loss import MaskedAPE

        metric = MaskedAPE()
        mock1 = {
            "outputs": {
                "pred": torch.tensor([1, 1, 1]).reshape(1, 3),
            },
            "target": torch.tensor([1, 1, 1]).reshape(1, 3),
            "mask": torch.tensor([1, 1, 1]).reshape(1, 3),
        }

        expected = torch.tensor([0])
        self._almost_equals(expected, metric(**mock1))

        # mock1 = [torch.tensor([1, 2, 1]).reshape(1, 3),
        #
        #          torch.tensor([1, 1, -100]).reshape(1, 3),
        #          torch.tensor([1, 1, 0]).reshape(1, 3),
        #          torch.tensor([1, 1, 1]).reshape(1, 3),
        #          ]
        # expected = torch.tensor([50])
        # self._almost_equals(expected, metric(*mock1))
        #
        # mock1 = [torch.tensor([1, 2, 1]).reshape(1, 3),
        #
        #          torch.tensor([1, 1, 1]).reshape(1, 3),
        #          torch.tensor([1, 1, 0]).reshape(1, 3),
        #          torch.tensor([1, 1, 0]).reshape(1, 3),
        #          ]
        # expected = torch.tensor([50])
        # self._almost_equals(expected, metric(*mock1))

    def test_APE(self):
        from loss import APE
        metric = APE()
        mock1 = {
            "outputs": {
                "pred": torch.tensor([1, 1, 1]).reshape(1, 3),
            },
            "target": torch.tensor([1, 1, 1]).reshape(1, 3),
            "group": torch.tensor([1, 2, 0]).reshape((1, 3)),
        }

        expected = {
            "train_loss": torch.tensor([0]),
            "valid_loss": torch.tensor([0]),
        }
        [self._almost_equals(e, a) for (e, a) in zip(expected.values(), metric(**mock1).values())]

    def testLogNorm(self):
        from loss import NegtiveLogNormLoss
        metric = NegtiveLogNormLoss()
        mock = {
            "mu": torch.tensor([1.]),
            "ln_sigma": torch.tensor([1.]).log(),
            "target": torch.tensor([1.])
        }
        import math
        expect = (-torch.distributions.LogNormal(mock["mu"], mock["ln_sigma"].exp()).log_prob(mock["target"])) \
                 - (math.sqrt(2 * math.pi) * mock["target"]).log()
        self._almost_equals(metric.func(**mock), expect)

        mock = {
            "mu": torch.tensor([3.]),
            "ln_sigma": torch.tensor([5.]).log(),
            "target": torch.tensor([7.])
        }
        import math
        expect = (-torch.distributions.LogNormal(mock["mu"], mock["ln_sigma"].exp()).log_prob(mock["target"])) \
                 - (math.sqrt(2 * math.pi) * mock["target"]).log()
        self._almost_equals(metric.func(**mock), expect)

    def test_dghb(self):
        x = torch.tensor(1).float()
        # x = torch.tensor(np.arange(-10, 10)).float()
        y = torch.tensor([0.3759686,
                          ])
        theta = [0.6, 2.0, 1.0, 1.1, 0.1]
        # theta = [torch.tensor(v) for v in theta]
        theta = [torch.tensor(v).repeat([y.shape[0]]) for v in theta]
        print(theta)
        actual = Loss_functions.nll_dghb(x, *theta)
        self._almost_equals(y, (-actual).exp())

        theta = [100.6, 2.0, 1.0, 1.1, 0.1]
        theta = [torch.tensor(v) for v in theta]
        expected = torch.tensor([1.039213e-22])
        actual = Loss_functions.nll_dghb(torch.tensor([200]), *theta)
        self._almost_equals(expected, (-actual).exp())

        theta = [10.6, 2.0, 1.0, 1.0, 0.1]
        theta = [torch.tensor(v) for v in theta]
        expected = torch.tensor([0.0008125254])
        actual = Loss_functions.nll_dghb(torch.tensor([20]), *theta)
        TestUtils.almost_equals(expected, (-actual).exp())

        times = 5000
        theta = [200.6, 2.0, 1.0, 1.1, 8]
        theta = [torch.tensor(v).repeat(times) for v in theta]

        Loss_functions.nll_dghb(torch.tensor([400]).repeat(times), *theta)

    def test_dghb6(self):
        theta = [0.10,
                 1.00,
                 3.00,
                 1.21,
                 1.60, ]
        theta = [torch.tensor(v) for v in theta]
        expected = torch.tensor([0.3178799])
        actual = Loss_functions.nll_dghb6(torch.tensor([1]), *theta)
        TestUtils.almost_equals(expected, (-actual).exp(), 1e-6)

        theta = [1.6,
                 1.00,
                 3.00,
                 1.0,
                 1.60, ]
        theta = [torch.tensor(v) for v in theta]
        expected = torch.tensor([0.09683605])
        actual = Loss_functions.nll_dghb6(torch.tensor([1]), *theta)
        TestUtils.almost_equals(expected, (-actual).exp(), 1e-6)


if __name__ == '__main__':
    unittest.main()
