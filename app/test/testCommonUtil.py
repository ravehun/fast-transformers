import unittest

import numpy as np
import torch
from common_utils import Mapping


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

    def test(self):
        np.linspace(-2, 2)
        from scipy.stats import *

        import matplotlib.pyplot as plt
        norm.fit
        t.fit()
        t.pdf()
        import tqdm
        tqdm.tqdm_notebook()
        np.linspace

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(30, 10))
        plt.subplots(1, 1, figsize=(30, 10))
        np.concatenate


if __name__ == '__main__':
    unittest.main()
