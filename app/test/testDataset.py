import unittest

import numpy as np


class MyTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.front_padding_num = 2
        self.train_start_date = '2012-01-01'
        self.valid_start_date = '2016-01-01'
        self.position_oov_size = 10
        self.stock_id_header_token = 1
        self.stock_price_start_token = 2
        self.asset_list = '../data/asset_list.txt'
        self.etf_list_fn = '../data/etf_list.txt'
        self.stock_oov_size = 10
        self.filenames = [
            "../data/sample_record_npz/window:10-agg:mean-date:2012-01-01:2017-01-011598597728.713553.npz"]
        self.market_fn = "../data/sample_record_npz/etf-date:2012-01-01:2017-01-01-1598523744.250263.npz"
        self.coor_fn = "../data/stock_most_relevent.parquet"

    def test_something(self):
        from dataset import MyIterableDataset
        from common_utils import Mapping

        dataset = MyIterableDataset(
            self.filenames[0]
            , self.market_fn
            , coor_fn=self.coor_fn
            , stock_mapping=Mapping().load(self.asset_list, self.stock_oov_size)
            , anchor_mapping=Mapping().load(self.etf_list_fn, 0)
            , front_padding_num=self.front_padding_num
            , train_start_date=self.train_start_date
            , valid_start_date=self.valid_start_date
            , position_oov_size=self.position_oov_size
            , stock_id_header_token=self.stock_id_header_token
            , stock_price_start_token=self.stock_price_start_token
        )

        for one in dataset:
            inputs, y, meta = one
            # print('\n'.join([(f'{k}:{v.shape}: {type(v)}') for (k, v) in inputs.items()]))

            stock_name = meta['name']
            group = inputs['group']
            header_tokens = inputs['header_tokens']
            days_off = inputs['days_off']
            anchor = inputs['anchor_feature']
            stock = inputs['stock_feature']
            anchor_names = meta['anchor_names']

            v = dataset.anchor_ds[stock_name].loc[days_off].values
            diff = v - stock.numpy()
            self.assertTrue(diff.sum() == 0)

            v = dataset.anchor_ds[list(anchor_names)].dropna('t').to_array()
            self.assertEqual(
                len(set(days_off.numpy().tolist()) - set(v.indexes['t']))
                , 0)
            diff = v.sel(t=days_off.numpy()).values - anchor.numpy()
            self.assertEqual(np.abs(diff).sum(), 0)

            break

    def test_dataframe(self):
        import pandas as pd
        import numpy as np
        data1 = [1, 2, 3]
        data2 = [1, 2, 4]
        data3 = [1, 3, 4]

        mock1 = pd.DataFrame(data=np.ones([1, 3]), columns=data1, index=[1])
        mock2 = pd.DataFrame(data=np.ones([1, 3]), columns=data2, index=[2])
        mock3 = pd.DataFrame(data=np.ones([1, 3]), columns=data3, index=[3])

        actual = pd.concat([mock1, mock2, mock3], axis=0, join="inner")
        expect = pd.DataFrame(data=np.ones([3, 1]), columns=[1], index=[1, 2, 3])
        print(actual)
        print(expect)
        # self.assertTrue(actual == expect)

    # def test_xa(self):
    #     l = 4
    #     d = 5
    #     ns = 3
    #     stock_features = np.ones((ns, l, d))
    #     l_indice = [
    #         [0, 5, 2, 3, ],
    #         [0, 5, 2, 4, ],
    #         [0, 2, 3, 5, ],
    #     ]
    #     t_indice = [
    #         [0, 1, 2, 3, ],
    #         [0, 1, 2, 4, ],
    #         [0, 2, 3, 7, ],
    #     ]
    #     from dataset import *
    #
    import pandas as pd
    pd.DataFrame().to_csv()


if __name__ == '__main__':
    unittest.main()
