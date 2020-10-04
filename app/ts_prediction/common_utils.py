import logging
import sys
from datetime import timedelta, datetime

import numpy as np
import pandas as pd
import scipy.special
import torch
from scipy.misc import derivative
from scipy.special import kv, kvp
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

    @staticmethod
    def shape_check(x, dim):
        if len(x.shape) != dim:
            raise ValueError(f"expect {dim}, actual {len(x.shape)})")


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
            import os
            os.makedirs(os.path.dirname(fn), exist_ok=True)
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


class TestUtils:
    @staticmethod
    def almost_equals(expected, actual, eps=1e-3):
        r = abs(actual / expected - 1)
        res = all(r < eps) and all(r >= 0)
        if not res:
            print(f"expect {expected}\n, actual {actual}")

        assert res

    @staticmethod
    def less_than_expected(expected, actual, eps=1e-3):
        r = expected - actual
        res = all(r < eps) and all(r >= 0)
        if not res:
            print(f"expect {expected}\n, actual {actual}")

        assert res


# note, we do not differentiate w.r.t. nu
class ModifiedBesselKv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, nu):
        is_cuda = inp.is_cuda
        ctx.save_for_backward(inp, nu)

        nu = nu.detach().cpu()
        inp = inp.detach().cpu()
        # ctx.save_for_backward(inp, nu)
        kv = scipy.special.kv(nu.detach().cpu().numpy(), inp.detach().cpu().numpy())
        ret = torch.from_numpy(np.array(kv))
        if is_cuda:
            ret = ret.cuda()
        return ret

    @staticmethod
    def backward(ctx, grad_out):
        inp, nu = ctx.saved_tensors

        def to_1dnp(x):
            return x.squeeze().detach().cpu().numpy()

        def to_tensor(x: np.ndarray, shape=grad_out.shape):
            x = x.reshape(shape)
            x = torch.tensor(x)
            if grad_out.is_cuda:
                x = x.cuda()
            return x

        def numeric_derivative(v, z):
            return derivative(lambda _v: kv(_v, z), v, dx=1e-10)

        z, v = to_1dnp(inp), to_1dnp(nu)
        df_2_order = numeric_derivative(v, z)
        to_tensor(df_2_order)
        # df_2_order = df_2_order.reshape(grad_out.shape)
        # df_2_order = torch.from_numpy(df_2_order)
        # if grad_out.is_cuda:
        #     df_2_order = df_2_order.cuda()

        df_2_z = kvp(v, z)
        return grad_out * to_tensor(df_2_z) \
            , grad_out * to_tensor(df_2_order)
        # df_2_order


class ModifiedBesselKve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, nu):
        ctx._nu = nu
        ctx.save_for_backward(x)
        kv = scipy.special.kve(nu.detach().cpu().numpy(), x.detach().cpu().numpy())
        ctx._kve = torch.from_numpy(np.array(kv))
        return ctx._kve

    @staticmethod
    def backward(ctx, grad_out):
        inp, = ctx.saved_tensors
        nu = ctx._nu
        kve = ctx._kve
        r = (kve +
             inp.exp() * torch.from_numpy(scipy.special.kvp(nu.detach().numpy(), inp.detach().numpy()))
             )
        return grad_out * r, None
