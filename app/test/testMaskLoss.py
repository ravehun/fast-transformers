import unittest

import numpy as np
import torch
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
            print(f"expect {x}, actual {y}")
        self.assertTrue(res)

    # def test_MaskedAPE(self):
    #     eps = 1e-4
    #     from loss import MaskedAPE
    #
    #     metric = MaskedAPE()
    #     mock1 = {
    #         "outputs": {
    #             "pred": torch.tensor([1, 1, 1]).reshape(1, 3),
    #         },
    #         "target": torch.tensor([1, 1, 1]).reshape(1, 3),
    #         "mask": torch.tensor([1, 1, 1]).reshape(1, 3),
    #     }
    #
    #     expected = torch.tensor([0])
    #     self._almost_equals(expected, metric(**mock1))

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
        #
        # metric = APE()
        # mock1 = {
        #     "pred": torch.tensor([1, 1, 1, 2]).reshape(1, -1),
        #     "target": torch.tensor([1, 1, 1, 1]).reshape(1, -1),
        #     "group": torch.tensor([1, 2, 0, 1]).reshape((1, -1)),
        # }
        #
        # expected = {
        #     "train_loss": torch.tensor([50]),
        #     "valid_loss": torch.tensor([0]),
        # }
        # [self._almost_equals(e, a) for (e, a) in zip(expected.values(), metric(**mock1).values())]

    #     mock1 = [torch.tensor([1, 1, 1, 1, 0]).reshape(1, -1),
    #              torch.tensor([1, 1, 1, 2, 0]).reshape(1, -1),
    #              {
    #                  "group": torch.tensor([1, 2, 0, 1, 2]).reshape((1, -1))
    #              }
    #              ]
    #     expected = {
    #         "train_loss": torch.tensor([25]),
    #         "valid_loss": torch.tensor([0]),
    #     }
    #     # print(expected.values(), metric(*mock1).values())
    #     [self._almost_equals(e, a) for (e, a) in zip(expected.values(), metric(*mock1).values())]
    #
    # def testLABE(self):
    #     from loss import MaskedMLABE
    #     metric = MaskedMLABE()
    #     mock1 = [torch.tensor([1, 1, 1, 1, 0]).reshape(1, -1),
    #              torch.tensor([1, 1, 1, 2, 0]).reshape(1, -1),
    #              {
    #                  "group": torch.tensor([1, 2, 0, 1, 2]).reshape((1, -1))
    #              }
    #              ]
    #     expected = {
    #         "train_loss": torch.tensor([25]),
    #         "valid_loss": torch.tensor([0]),
    #     }
    #
    #     print(metric(*mock1).values())
    #     [self._almost_equals(e, a) for (e, a) in zip(expected.values(), metric(*mock1).values())]
    #
    #     print(metric)

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


if __name__ == '__main__':
    unittest.main()
