from datetime import timedelta, datetime

import numpy as np
import pandas as pd


class CommonUtils():
    @staticmethod
    def date_to_idx(cur_date, start="1950-01-01"):
        date_format = "%Y-%m-%d"
        a = datetime.strptime(cur_date, date_format)
        b = datetime.strptime(start, date_format)
        return (a - b).days

    @staticmethod
    def idx_to_date(days, start="1950-01-01"):
        date_format = "%Y-%m-%d"
        cur = datetime.strptime(start, date_format) + timedelta(days=days)
        return cur.isoformat()


class Mapping():
    def __init__(self, ):
        pass
    def __len__(self):
        return self.n2i.max()

    def load(self, fn, offset=10):
        data = pd.read_csv(fn, header=None, names=["stock_name"])
        data["stock_id"] = data.index + offset

        self.n2i = data.set_index("stock_name").stock_id
        self.i2n = data.set_index("stock_id").stock_name
        self.stock_names = self.i2n.values

        return self

    def name2id(self, names):
        return np.array(list(self.n2i[name] for name in names))

    def id2name(self, ids):
        return np.array(list(self.i2n[id] for id in ids))
