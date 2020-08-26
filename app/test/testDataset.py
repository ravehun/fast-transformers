import unittest


class MyTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.front_padding_num = 2
        self.train_start_date = '2009-01-01'
        self.valid_start_date = '2014-01-01'
        self.position_oov_size = 10
        self.stock_id_header_token = 1
        self.stock_price_start_token = 2
        self.asset_list = '../data/asset_list.txt'
        self.etf_list_fn = '../data/etf_list.txt'
        self.stock_oov_size = 10
        self.filenames = ["../data/sample_record_npz/window:20-agg:max-1596971083.52575.npz"]
        self.market_fn = "../data/sample_record_npz/etf-date:2012-01-01:2017-01-011598345700.054786.npz"

    def test_something(self):
        from dataset import MyIterableDataset
        from common_utils import Mapping

        dataset = MyIterableDataset(
            self.filenames[0]
            , self.market_fn
            , stock_mapping=Mapping().load(self.asset_list, self.stock_oov_size)
            , etf_mapping=Mapping().load(self.etf_list_fn, 0)
            , front_padding_num=self.front_padding_num
            , train_start_date=self.train_start_date
            , valid_start_date=self.valid_start_date
            , position_oov_size=self.position_oov_size
            , stock_id_header_token=self.stock_id_header_token
            , stock_price_start_token=self.stock_price_start_token
        )

        for one in dataset:
            x, y, meta = one
            print(meta["etfs"].shape)


if __name__ == '__main__':
    unittest.main()
