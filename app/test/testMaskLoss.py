import unittest

import torch


class MaskLoss(unittest.TestCase):
    # def test_MaskedAPE(self):
    #     from train_gpt2 import MaskedAPE
    #     metric = MaskedAPE()
    #     mock1 = [torch.tensor([1, 1, 1]).reshape(1, 3), torch.tensor([1, 1, 1]).reshape(1, 3)]
    #     expected = torch.tensor([0])
    #     assert expected == metric(*mock1)
    #
    #     mock1 = [torch.tensor([1, 1, 1]).reshape(1, 3), torch.tensor([1, 1, 0]).reshape(1, 3)]
    #     expected = torch.tensor([0])
    #     assert expected == metric(*mock1)
    #
    #     mock1 = [torch.tensor([1, 2, 1]).reshape(1, 3), torch.tensor([1, 1, 0]).reshape(1, 3)]
    #     expected = torch.tensor([50])
    #     assert expected == metric(*mock1)
    #
    #     mock1 = [torch.tensor([0, 2, 1, 3]).reshape(1, -1), torch.tensor([1, 1, 0, 0]).reshape(1, -1)]
    #     expected = torch.tensor([100])
    #     assert expected == metric(*mock1)

    def test_APE(self):
        from train_gpt2 import APE
        metric = APE()
        mock1 = [torch.tensor([1, 1, 1]).reshape(1, 3),
                 torch.tensor([1, 1, 1]).reshape(1, 3),
                 {
                     "group": torch.tensor([1, 2, 0]).reshape((1, 3))
                 }
                 ]
        expected = {
            "train_loss": torch.tensor([0]),
            "valid_loss": torch.tensor([0]),
        }
        [self.assertAlmostEqual(e, a) for (e, a) in zip(expected.values(), metric(*mock1).values())]

        mock1 = [torch.tensor([1, 1, 1, 1, 0]).reshape(1, -1),
                 torch.tensor([1, 1, 1, 2, 0]).reshape(1, -1),
                 {
                     "group": torch.tensor([1, 2, 0, 1, 2]).reshape((1, -1))
                 }
                 ]
        expected = {
            "train_loss": torch.tensor([50]),
            "valid_loss": torch.tensor([0]),
        }
        print(expected.values(), metric(*mock1).values())
        [self.assertAlmostEqual(e.tolist(), a.tolist()) for (e, a) in zip(expected.values(), metric(*mock1).values())]


if __name__ == '__main__':
    unittest.main()
