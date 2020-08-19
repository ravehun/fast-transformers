import unittest

import torch


class MaskLoss(unittest.TestCase):

    def _almost_equals(self, x, y, eps=1e-3, assertion=True):
        if y.dtype is torch.bool:
            res = torch.all(x == y)
        else:
            res = all((x - y).abs() < eps)
        if not res:
            print(f"expect {x}, actual {y}")
        if assertion:
            self.assertTrue(res)

    def torch_almost_equals(self, x, y, eps=1e-3, assertion=True):
        res = torch.all((x - y).abs() < eps)
        if not res:
            print(f"expect {x}, actual {y}")
        if assertion:
            self.assertTrue(res)

    def dict_eq(self, expect, actual):
        self.assertListEqual(list(expect.keys()), list(actual.keys()))
        self.assertTrue(all([self.torch_almost_equals(e, a) for (e, a) in zip(expect.values(), actual.values())]))

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
        # mock1 = {
        #     "outputs": {
        #         "pred": torch.tensor([1, 1, 1]).reshape(1, 3),
        #     },
        #     "target": torch.tensor([1, 1, 1]).reshape(1, 3),
        #     "group": torch.tensor([1, 2, 0]).reshape((1, 3)),
        #     "confidence": torch.tensor([1, 1, 1]).reshape((1, 3)),
        # }
        #
        # expected = {
        #     "train_loss": torch.tensor([0]),
        #     "valid_loss": torch.tensor([0]),
        # }
        #
        # [self._almost_equals(e, a) for (e, a) in zip(expected.values(), metric(**mock1).values())]

        mock1 = {
            "outputs": {
                "pred": torch.tensor([1, 1, 1, 0, 0]).reshape(1, -1),
                "confidence": torch.tensor([1, 1, 1, 1, -1e9]).reshape((1, -1)),
            },
            "target": torch.tensor([1, 1, 1, 1, 1]).reshape(1, -1),
            "group": torch.tensor([1, 2, 0, 1, 1]).reshape((1, -1)),
        }

        expected = {
            "train_loss": torch.tensor([50]),
            "valid_loss": torch.tensor([0]),
            "train_mask": torch.tensor([1, 0, 0, 1, 1]),
            "valid_mask": torch.tensor([0, 1, 0, 0, 0]),
        }
        print(metric(reduction=False, **mock1))
        for (e, a) in zip(expected.values(), metric(**mock1).values()):
            self._almost_equals(e, a)

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
