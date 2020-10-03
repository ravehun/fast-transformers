import warnings

warnings.filterwarnings("ignore")

from torch.utils.data import IterableDataset
import xarray as xr
from .common_utils import *
from .loss import *
import pandas as pd


class MyIterableDataset(IterableDataset):

    def __init__(self, fn, market_fn, coor_fn, train_start_date, valid_start_date, position_oov_size,
                 stock_mapping=None,
                 anchor_mapping=None,
                 front_padding_num=0,
                 stock_id_header_token=1, stock_price_start_token=2, limits=None
                 ):
        self.fn = fn
        self.feature = None
        self.valid_start_date = valid_start_date
        self.stock_mapping = stock_mapping
        self.anchor_mapping = anchor_mapping
        self.front_padding_num = front_padding_num
        self.stock_id_token = stock_id_header_token
        self.price_start_token = stock_price_start_token
        self.train_start_date = train_start_date
        self.position_oov_size = position_oov_size
        self.market_fn = market_fn
        self.coor_fn = coor_fn
        self.logger = LoggerUtil.get_logger("dataset")
        self.limits = limits
        self.eps = 1e-6

    @staticmethod
    def affine(x, inf, sup):
        x = (x.max(axis=1, keepdims=True) - x) / (
                    x.max(axis=1, keepdims=True) - x.min(axis=1, keepdims=True) + self.eps)
        x = x * (sup - inf) + inf
        return x

    @staticmethod
    def norm(x: np.ndarray, n_type="min_max", **kwargs):
        if len(x.shape) != 3:
            raise ValueError(f"expect shape is 3, actual {x.shape}")

        f = {
            "min_max": lambda: (x.max(axis=1, keepdims=True) - x) / (
                    x.max(axis=1, keepdims=True) - x.min(axis=1, keepdims=True)),
            "z_score": lambda: (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + self.eps),
            "affine": lambda: MyIterableDataset.affine(x, kwargs["inf"], kwargs["sup"]),
            "log": lambda: (x + self.eps).log(),
        }

        return f[n_type]()

    @staticmethod
    def pull_by_name(stock_name, coor, all_ds):
        keys = list(coor[stock_name]) + [stock_name]
        pulled = all_ds[keys].dropna('t')
        return pulled[stock_name], pulled[list(coor[stock_name])]

    @staticmethod
    def pad_or_trim_to(sd, seq_len, front=0):
        n = len(sd.shape)
        d = max(-n, -2)
        if n < seq_len:
            fill = [(front, 0)] * n
            fill[d] = (front, seq_len - sd.shape[d])
            psd = np.pad(sd, fill
                         , constant_values=0.0
                         )
            return psd
        else:
            if d == 2:
                return sd[:seq_len]
            elif d == 3:
                return sd[..., :seq_len, :]

    @staticmethod
    def refill_padding_marks(sd, mask_start, fill_value=0):
        idx = sd.indexes['t'].fillna(0).values
        idx[idx >= mask_start] = fill_value
        return idx

    @staticmethod
    def get_group_mask(days_offset, label, valid_split):
        origin_shape = days_offset.shape

        def get_group(x):
            if x == 0:
                return SeqGroupMetric.PAD
            if x < valid_split:
                return SeqGroupMetric.TRAIN
            else:
                return SeqGroupMetric.VALID

        out = [get_group(x) for x in days_offset.flatten()]
        group_mask = np.array(out).reshape(origin_shape)
        group_mask = group_mask * (label > 0).astype(group_mask.dtype)

        return group_mask

    def load(self):
        # read input
        stock_npz = np.load(self.fn)
        etf_npz = np.load(self.market_fn)
        self.coor = pd.read_parquet(self.coor_fn)

        # etf2id = dict([(x, i) for i, x in enumerate(etf_npz["etf_name"])])
        # print(f"etf2id, {etf2id}")
        # etf_indice = [etf2id[x] for x in self.etf_mapping.stock_names]
        # etf_indice, etf_name = zip(*etf_indice_name)
        # print(f"etf_name {self.etf_mapping.stock_names} {etf_indice}")
        # etfs = etf_npz["input"][etf_indice]

        self.stock_name = stock_npz["stock_name"]

        # manage offset
        ab_start_days_offset = CommonUtils.date_to_idx(self.train_start_date)
        self.relative_valid_start_offset = CommonUtils.date_to_idx(self.valid_start_date) - ab_start_days_offset

        # label = stock_npz["target"]

        def get_relative_offset(x):
            relative_days_offset = (x - ab_start_days_offset + self.position_oov_size) * \
                                   (x > 0).astype(x.dtype)
            return relative_days_offset

        # relative_days_offset = (stock_npz["days_offset"] - ab_start_days_offset + self.position_oov_size) * \
        #                        (stock_npz["days_offset"] > 0).astype(stock_npz["days_offset"].dtype)
        k2 = stock_npz["days_offset"].flatten()
        k2 = np.unique(k2)
        k2.sort()
        self.logger.debug(f'min max date {CommonUtils.idx_to_date(k2[1])}'
                          f' {CommonUtils.idx_to_date(stock_npz["days_offset"].max())}')

        stock_relative_do = get_relative_offset(stock_npz["days_offset"])
        etf_relative_do = get_relative_offset(etf_npz["days_offset"])
        self.logger.debug(f"stock_relative_do {CommonUtils.idx_to_date(stock_relative_do.max())}")
        # stock_id
        stock_id = self.stock_mapping.name2id(self.stock_name)

        # preprocess data
        etfs = etf_npz["input"].copy()
        # etfs[..., 4] = np.log(etfs[..., 4] + 1)
        etfs = np.log(etfs + 1)
        feature = stock_npz["input"].copy()
        # feature[..., 4] = np.log(feature[..., 4] + 1)
        feature = np.log(feature + 1)

        self.seq_len = feature.shape[1]
        self.xrds_autojoin_padding_start = self.seq_len * 10
        # create dataset
        stock_repad = self.repadding(stock_relative_do, self.xrds_autojoin_padding_start)
        stock_das = self.create_xrds3(stock_features=feature
                                      , stock_names=stock_npz["stock_name"]
                                      , l_indice=stock_repad,
                                      merge_out=False)

        etf_das = self.create_xrds3(stock_features=etfs
                                    , stock_names=etf_npz["etf_name"],
                                    l_indice=self.repadding(etf_relative_do, self.xrds_autojoin_padding_start),
                                    merge_out=False)

        self.anchor_ds = xr.merge(stock_das + etf_das)
        x = stock_npz["input"][..., 3]
        y = stock_npz['target']
        y = np.log((y + self.eps) / (x + self.eps))

        label = self.create_xrds2(y, stock_npz['stock_name'], stock_repad)
        self.label = label

    def pad_stock_id(self, stock_id):
        return np.array([self.stock_id_token, stock_id, self.price_start_token])

    def pad_label(self, y):
        return np.concatenate([np.zeros(self.front_padding_num), y])

    def pull_data_by_stock_name(self, stock_name, coor, anchor, label):
        # inner join on date
        pad = lambda x: self.pad_or_trim_to(x, self.seq_len)
        stock_da, anchor_ds = self.pull_by_name(stock_name, coor, anchor)

        stock_np = pad(stock_da.values)
        anchor_np = pad(anchor_ds.to_array().values)
        # joined date indice
        idx = pad(self.refill_padding_marks(stock_da, self.xrds_autojoin_padding_start))
        # get label indice
        label_da = label[stock_name].loc[idx]
        return label_da.values, stock_np, anchor_np, idx

    @staticmethod
    def create_xrds3(stock_features, stock_names, l_indice, merge_out=True):
        N, L, D = stock_features.shape
        da_list = [
            xr.DataArray(stock_features[i],
                         [
                             ('t', l_indice[i]),
                             ('d', range(D)),
                         ],
                         name=stock_names[i]) for i in range(N)
        ]
        return xr.merge(
            da_list
        ) if merge_out else da_list

    @staticmethod
    def create_xrds2(stock_features, stock_names, l_indice, merge_out=True):
        N, L = stock_features.shape
        da_list = [
            xr.DataArray(stock_features[i],
                         [
                             ('t', l_indice[i]),
                         ],
                         name=stock_names[i]) for i in range(N)
        ]
        return xr.merge(
            da_list
        ) if merge_out else da_list

    @staticmethod
    def repadding(x, start):
        def repadding_one(x, padding_start):
            x[x == 0] = np.arange(x[x == 0].shape[0]) + padding_start
            x[0] = 0
            return x

        repad = [
            repadding_one(x[i], start) for i in range(x.shape[0])
        ]
        repad = np.stack(repad)
        return repad

    def get_by_name(self, name):
        y, x, anchor, rdf = \
            self.pull_data_by_stock_name(name, self.coor, self.anchor_ds, self.label)
        stock_id = self.stock_mapping.name2id([name])[0]
        anchor_id = self.anchor_mapping.name2id(self.coor[name])

        group = self.get_group_mask(rdf, y, self.relative_valid_start_offset)

        group = self.pad_label(group)
        y = torch.tensor(self.pad_label(y))
        meta = {
            "name": name,
            "anchor_names": self.coor[name].values,
        }
        inputs = {
            "group": torch.tensor(group, dtype=torch.long),
            "header_tokens": torch.tensor(self.pad_stock_id(stock_id), dtype=torch.long),
            "days_off": torch.tensor(rdf, dtype=torch.long),
            "anchor_feature": torch.tensor(anchor),
            "stock_feature": torch.tensor(x),
            "anchor_ids": torch.tensor(anchor_id),
        }
        return inputs, y, meta

    def __iter__(self):
        if not hasattr(self, 'anchor_ds'):
            self.load()
        stock_name_list = self.stock_name.copy()
        np.random.shuffle(stock_name_list)
        for name in stock_name_list:
            inputs, y, meta = self.get_by_name(name)
            yield inputs, y
            # yield inputs, y, meta

    def __getitem__(self, idx):
        if not hasattr(self, 'anchor_ds'):
            self.load()

        name = self.stock_name[idx]
        inputs, y, meta = self.get_by_name(name)
        return inputs, y

    def __len__(self):
        if not hasattr(self, 'anchor_ds'):
            self.load()
        if self.limits is None:
            return len(self.stock_name)
        return self.limits
