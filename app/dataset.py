import warnings

warnings.filterwarnings("ignore")

from torch.utils.data import IterableDataset
import numpy as np
import torch

from common_utils import CommonUtils
from loss import *


class MyIterableDataset(IterableDataset):

    def __init__(self, fn, market_fn, train_start_date, valid_start_date, position_oov_size, stock_mapping=None,
                 etf_mapping=None,
                 front_padding_num=0,
                 stock_id_header_token=1, stock_price_start_token=2):
        self.fn = fn
        self.feature = None
        self.valid_start_date = valid_start_date
        self.stock_mapping = stock_mapping
        self.etf_mapping = etf_mapping
        self.front_padding_num = front_padding_num
        self.stock_id_token = stock_id_header_token
        self.price_start_token = stock_price_start_token
        self.train_start_date = train_start_date
        self.position_oov_size = position_oov_size
        self.market_fn = market_fn
        self.selected_etfs = [
            'DVY',
            'IWO',
            'IWN',
            'IVW',
            'IVE',
            'MTUM',
            'XLY',
            'XLP',
            'XLE',
            'XLF',
            'XLV',
            'XLI',
            'XLB',
            'XLK',
            'XLU',
            'XLRE',
            'GURU',
            'LRGF',
            'QYLD',
            'MOAT',
            'MINT',
            'SHY',
            'TLT',
            'LQD',
            'HYG',
            'MUB',
            'BKLN',
            'MBB',
            'CWB',
            'TIP',
            'USO',
            'USL',
            'BNO',
            'UWT',
            'DWT',
            'UGA',
            'UNG',
            'UGAZ',
            'DGAZ',
            'CORN',
            'SOYB',
            'WEAT',
            'NIB',
            'BJO',
            'BALB',
            'SGGB',
            'JJUB',
            'JJCB',
            'LD',
            'BJJN',
            'JJTB',
            'GLD',
            'SLV',
            'PALL',
            'PPLT',
            'UGLD',
            'DGLD',
            'USLV',
            'DSLV',
            'VXX',
            'SVXY',
            'TVIX',
            'SDS',
            'QQQ',
            'XIV',
            'IWM',

        ]

    @staticmethod
    def affine(x, inf, sup):
        x = (x.max(axis=1, keepdims=True) - x) / (x.max(axis=1, keepdims=True) - x.min(axis=1, keepdims=True) + 1e-6)
        x = x * (sup - inf) + inf
        return x

    @staticmethod
    def norm(x: np.ndarray, n_type="min_max", **kwargs):
        if len(x.shape) != 3:
            raise ValueError(f"expect shape is 3, actual {x.shape}")

        f = {
            "min_max": lambda: (x.max(axis=1, keepdims=True) - x) / (
                    x.max(axis=1, keepdims=True) - x.min(axis=1, keepdims=True)),
            "z_score": lambda: (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6),
            "affine": lambda: MyIterableDataset.affine(x, kwargs["inf"], kwargs["sup"]),
            "log": lambda: (x + 1e-6).log(),
        }

        return f[n_type]()

    def load(self):
        npz = np.load(self.fn)
        market_npz = np.load(self.market_fn)
        etf2id = dict([(x, i) for i, x in enumerate(market_npz["etf_name"])])
        # print(f"etf2id, {etf2id}")
        etf_indice = [etf2id[x] for x in self.etf_mapping.stock_names]
        # etf_indice, etf_name = zip(*etf_indice_name)
        # print(f"etf_name {self.etf_mapping.stock_names} {etf_indice}")

        etfs = market_npz["input"][etf_indice]

        def get_group_mask(xs, valid_split):
            origin_shape = xs.shape

            def get_group(x):
                if x == 0:
                    return SeqGroupMetric.PAD
                if x < valid_split:
                    return SeqGroupMetric.TRAIN
                else:
                    return SeqGroupMetric.VALID

            out = [get_group(x) for x in xs.flatten()]

            return np.array(out).reshape(origin_shape)

        self.stock_name = npz["stock_name"]

        ab_start_days_offset = CommonUtils.date_to_idx(self.train_start_date)
        relative_valid_start_offset = CommonUtils.date_to_idx(self.valid_start_date) - ab_start_days_offset

        feature = npz["input"].copy()
        feature[..., 4] = np.log(feature[..., 4] + 1e-6)

        label = npz["target"]
        relative_days_offset = (npz["days_offset"] - ab_start_days_offset + self.position_oov_size) * \
                               (npz["days_offset"] > 0).astype(npz["days_offset"].dtype)
        stock_id = self.stock_mapping.name2id(self.stock_name)

        etfs = etfs.copy()
        etfs[..., 4] = np.log(etfs[..., 4] + 1e-6)

        if self.front_padding_num > 0:
            feature = np.pad(feature
                             , [(0, 0), (self.front_padding_num, 0), (0, 0)]
                             , constant_values=0.0)
            label = np.pad(label
                           , [(0, 0), (self.front_padding_num, 0)]
                           , constant_values=0.0)
            relative_days_offset = np.pad(relative_days_offset
                                          , [(0, 0), (self.front_padding_num, 0)]
                                          , constant_values=0.0)
            etfs = np.pad(etfs
                          , [(0, 0), (self.front_padding_num, 0), (0, 0)]
                          , constant_values=0.0)

        group_mask = get_group_mask(relative_days_offset, relative_valid_start_offset)
        group_mask = group_mask * (label > 0).astype(group_mask.dtype)
        self.feature = torch.tensor(feature, dtype=torch.float32)
        self.label = torch.tensor(label, dtype=torch.float32)
        self.group = torch.tensor(group_mask, dtype=torch.float32)
        self.etfs = etfs

        self.stock_id = np.stack([
            np.ones_like(stock_id) * self.stock_id_token,
            stock_id,
            np.ones_like(stock_id) * self.price_start_token,
        ], axis=1)

        self.stock_id = torch.tensor(self.stock_id, dtype=torch.int64)
        self.days_offset = torch.tensor(relative_days_offset, dtype=torch.int64)

    def __iter__(self):
        if self.feature is None:
            self.load()
        indice = np.arange(self.feature.shape[0])
        np.random.shuffle(indice)
        for idx in indice:
            x = self.feature[idx]
            # print(f"x mask {(x[..., 0] > 0).sum(-1)}")
            y = self.label[idx]
            meta = {
                "stock_name": self.stock_name[idx],
                "group": self.group[idx],
                "stock_id": self.stock_id[idx],
                "days_off": self.days_offset[idx],
                "etfs": self.etfs
            }
            yield x, y, meta
