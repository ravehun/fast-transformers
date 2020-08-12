import unittest

import torch


class MaskLoss(unittest.TestCase):

    def _almost_equals(self, x, y, eps=1e-3):
        res = all((x - y).abs() < eps)
        if not res:
            print(f"expect {x}, actual {y}")
        self.assertTrue(res)

    def test_MaskedAPE(self):
        eps = 1e-4
        from loss import MaskedAPE
        metric = MaskedAPE()
        mock1 = [torch.tensor([1, 1, 1]).reshape(1, 3),
                 torch.tensor([1, 1, 1]).reshape(1, 3),
                 torch.tensor([1, 1, 1]).reshape(1, 3),
                 torch.tensor([1, 1, 1]).reshape(1, 3),
                 ]
        expected = torch.tensor([0])
        assert expected == metric(*mock1)

        mock1 = {
            "pred":       torch.tensor([1, 1, 1]).reshape(1, 3),
            "confidence": torch.tensor([1, 1, -100]).reshape(1, 3),
            "target":     torch.tensor([1, 1, 1]).reshape(1, 3),
            "mask":      torch.tensor([1, 1, 1]).reshape(1, 3),
        }

        expected = torch.tensor([0])
        self._almost_equals(expected, metric(**mock1))

        mock1 = [torch.tensor([1, 2, 1]).reshape(1, 3),

                 torch.tensor([1, 1, -100]).reshape(1, 3),
                 torch.tensor([1, 1, 0]).reshape(1, 3),
                 torch.tensor([1, 1, 1]).reshape(1, 3),
                 ]
        expected = torch.tensor([50])
        self._almost_equals(expected, metric(*mock1))

        mock1 = [torch.tensor([1, 2, 1]).reshape(1, 3),

                 torch.tensor([1, 1, 1]).reshape(1, 3),
                 torch.tensor([1, 1, 0]).reshape(1, 3),
                 torch.tensor([1, 1, 0]).reshape(1, 3),
                 ]
        expected = torch.tensor([50])
        self._almost_equals(expected, metric(*mock1))


    def test_APE(self):
        from loss import APE
        metric = APE()
        mock1 = {
            "pred":       torch.tensor([1, 1, 1]).reshape(1, 3),
            "confidence": torch.tensor([1, 1, 1]).reshape(1, 3),
            "target":     torch.tensor([1, 1, 1]).reshape(1, 3),
            "group":      torch.tensor([1, 2, 0]).reshape((1, 3)),
        }

        expected = {
            "train_loss": torch.tensor([0]),
            "valid_loss": torch.tensor([0]),
        }
        [self._almost_equals(e, a) for (e, a) in zip(expected.values(), metric(**mock1).values())]


        metric = APE()
        mock1 = {
            "pred":       torch.tensor([1, 1, 1,2]).reshape(1, -1),
            "confidence": torch.tensor([1, 1, 1,1]).reshape(1, -1),
            "target":     torch.tensor([1, 1, 1,1]).reshape(1, -1),
            "group":      torch.tensor([1, 2, 0,1]).reshape((1, -1)),
        }

        expected = {
            "train_loss": torch.tensor([50]),
            "valid_loss": torch.tensor([0]),
        }
        [self._almost_equals(e, a) for (e, a) in zip(expected.values(), metric(**mock1).values())]


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


if __name__ == '__main__':
    unittest.main()
