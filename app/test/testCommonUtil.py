import unittest

import numpy as np
import torch
from common_utils import Mapping
from common_utils import TestUtils


class TestCommonUtils(unittest.TestCase):
    def testMapping(self):
        m = Mapping()
        m.load('test/asset_list.txt')
        self.assertEqual(m.name2id(['a.us']), np.array([10]))
        self.assertEqual(m.id2name([10]), np.array(['a.us']))

    def test_embeding(self):
        # example with padding_idx
        embedding = torch.nn.Embedding(10, 3, padding_idx=0)
        input = torch.ones(1, 4).long()
        x = embedding(input)
        assert (x.shape == torch.Size([1, 4, 3]))

    def test_ModifiedBesselKv(self):
        from common_utils import ModifiedBesselKv
        modified_bessel = ModifiedBesselKv.apply
        x = torch.randn(5, requires_grad=True, dtype=torch.float64).exp()
        v = torch.zeros_like(x)
        modified_bessel(x, v)
        torch.autograd.gradcheck(modified_bessel, (x, v))

        # correctness
        x = torch.arange(1, 11)
        expected = [0.601907230, 0.253759755, 0.122170376, 0.062228812, 0.032706274, 0.017536731, 0.009534170,
                    0.005236422, 0.002898463, 0.001614255]
        expected = torch.tensor(expected)
        actual = modified_bessel(x, x)
        TestUtils.almost_equals(expected, actual, 1e-6)

        expected = torch.tensor([0.007874383])
        actual = modified_bessel(torch.tensor([200]), torch.tensor([300]))
        TestUtils.almost_equals(expected, actual, 1e-6)

    def test_ModifiedBesselKve(self):
        from common_utils import ModifiedBesselKve
        modified_bessel = ModifiedBesselKve.apply
        x = torch.randn(5, requires_grad=True, dtype=torch.float64).exp()
        v = torch.zeros_like(x)
        modified_bessel(x, v)
        torch.autograd.gradcheck(modified_bessel, (x, v))

        # correctness
        x = torch.arange(1, 11).float()
        expected = [0.601907230, 0.253759755, 0.122170376, 0.062228812, 0.032706274, 0.017536731, 0.009534170,
                    0.005236422, 0.002898463, 0.001614255]

        expected = torch.tensor(expected) * x.exp()
        actual = modified_bessel(x, x)
        TestUtils.almost_equals(expected, actual, 1)


if __name__ == '__main__':
    unittest.main()
