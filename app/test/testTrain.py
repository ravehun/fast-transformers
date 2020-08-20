import unittest

import torch
from train_gpt2 import TimeSeriesTransformer


class MyTestCase(unittest.TestCase):
    def test_something(self):
        batch_size = 1
        input_dimensions = 5
        seq_len = 15
        feed_forward_dimensions = 8
        front_padding_num = 3
        model = TimeSeriesTransformer(input_dimensions=input_dimensions,
                                      file_re="",
                                      # project_dimension=128,
                                      n_layers=1,
                                      n_heads=2,
                                      query_dimensions=4,
                                      value_dimensions=4,
                                      feed_forward_dimensions=feed_forward_dimensions,
                                      num_workers=0,
                                      batch_size=batch_size,
                                      seq_len=seq_len,
                                      seed=101,
                                      lr=1e-4,
                                      asset_list='../data/asset_list.txt',
                                      front_padding_num=front_padding_num,
                                      train_start_date="2009-01-01",
                                      valid_start_date="2013-01-01",
                                      )
        self.assertIsNotNone(model)

        mock = {
            "x": torch.zeros(batch_size, seq_len, feed_forward_dimensions),
            "meta": {
                "stock_id": torch.tensor([[1, 2, 1]], dtype=torch.long),
                "days_off": torch.ones(batch_size, seq_len, dtype=torch.long)
            }
        }

        x, seq_mask, relative_days_off = \
            model.attach_head(**mock)
        self.assertTrue(
            torch.all(x[..., :front_padding_num, :].sum(-1) != 0)
        )
        self.assertTrue(
            torch.all(
                seq_mask[0, :front_padding_num] == torch.ones(front_padding_num)
            )
        )

        self.assertTrue(
            torch.all(
                relative_days_off[0, :front_padding_num] == torch.arange(1, front_padding_num + 1)
            )
        )
        self.assertTrue(
            torch.all(
                torch.tensor([1, 2]).repeat([2, 1]) == torch.tensor([[1, 2], [1, 2]])
            )
        )


if __name__ == '__main__':
    unittest.main()
