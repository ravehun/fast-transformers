import logging
import sys
from datetime import timedelta, datetime

import numpy as np
import pandas as pd
from scipy.stats import t, norm


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
        cur = datetime.strptime(start, date_format) + timedelta(days=int(days))
        return cur.isoformat()

    @staticmethod
    def unstack(x):
        return [x[i] for i in range(x.shape[0])]


class DistUtil():
    @staticmethod
    def t_to_norm(x, dropna=True):
        if dropna:
            z = x[~np.isnan(x)]
        else:
            z = x
        args = t.fit(z)
        t_df = t(*args)
        t_cdf_v = t_df.cdf(x)
        n_df = norm(0, 1)
        return n_df.ppf(t_cdf_v)


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


class LoggerUtil():
    @staticmethod
    def setup_all(fn):
        handlers = []
        handlers.append(LoggerUtil.get_handler(None, logging.INFO))
        if fn is not None:
            handlers.append(LoggerUtil.get_handler(fn, logging.INFO))

        loggers_profiles = [
            ('dataset', *handlers),
            ("train", *handlers,),
        ]

        for args in loggers_profiles:
            LoggerUtil.setup(*args)

    @staticmethod
    def get_handler(fn, level):
        if fn is None:
            hdl = logging.StreamHandler(sys.stdout)
        else:
            hdl = logging.FileHandler(fn)

        hdl.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        hdl.setFormatter(formatter)
        return hdl

    @staticmethod
    def setup(name, *handler):

        root = logging.getLogger(name)
        root.setLevel(logging.DEBUG)

        for x in handler:
            root.addHandler(x)
        return root

    @staticmethod
    def get_logger(name):
        root = logging.getLogger(name)
        return root


class CudaMemUtil():
    @staticmethod
    def show_mem_usage(divice_id=0):
        import nvidia_smi

        nvidia_smi.nvmlInit()

        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(divice_id)
        # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

        print("Total memory:", info.total)
        print("Free memory:", info.free)
        print("Used memory:", info.used)

        nvidia_smi.nvmlShutdown()
