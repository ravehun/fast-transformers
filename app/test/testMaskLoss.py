import unittest

import torch


class MaskLoss(unittest.TestCase):
    def test_something(self):
        from train_gpt2 import MaskedAPE
        metric = MaskedAPE()
        mock1 = [torch.tensor([1, 1, 1]).reshape(1,3), torch.tensor([1, 1, 1]).reshape(1,3)]
        expected = torch.tensor([0])
        assert expected == metric(*mock1)

        mock1 = [torch.tensor([1, 1, 1]).reshape(1,3), torch.tensor([1, 1, 0]).reshape(1,3)]
        expected = torch.tensor([0])
        assert expected == metric(*mock1)

        mock1 = [torch.tensor([1, 2, 1]).reshape(1,3), torch.tensor([1, 1, 0]).reshape(1,3)]
        expected = torch.tensor([50])
        assert expected == metric(*mock1)


        mock1 = [torch.tensor([0, 2, 1, 3 ]).reshape(1,-1), torch.tensor([1, 1, 0,0]).reshape(1,-1)]
        expected = torch.tensor([100])
        assert expected == metric(*mock1)

if __name__ == '__main__':
    unittest.main()
