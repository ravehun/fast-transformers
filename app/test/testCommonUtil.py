import unittest

from common_utils import Mapping
import numpy as np
import torch

class CommonUtils(unittest.TestCase):
    def testMapping(self):
        m = Mapping()
        m.load('test/asset_list.txt')
        self.assertEqual(m.name2id(['a.us']),np.array([10]))
        self.assertEqual(m.id2name([10]), np.array(['a.us']))

    def test_embeding(self):
        # example with padding_idx
        embedding = torch.nn.Embedding(10, 3, padding_idx=0)
        input = torch.ones(1,4).long()
        x= embedding(input)
        assert(x.shape == torch.Size([1,4,3]))
if __name__ == '__main__':
    unittest.main()
