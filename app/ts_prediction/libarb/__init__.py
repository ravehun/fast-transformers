import sys

import cppimport
import numpy as np

sys.path.append('ts_prediction/libarb')
funcs = cppimport.imp('arb_wrapper')
from ..common_utils import CommonUtils


def hypergeometric_pfq(a: np.ndarray, b: np.ndarray, z: np.ndarray):
    CommonUtils.shape_check(a, 2)
    CommonUtils.shape_check(b, 2)
    CommonUtils.shape_check(z, 1)
    if not (a.shape[0] == b.shape[0] and z.shape[0] == b.shape[0]):
        raise ValueError("shape[0] not same")

    return funcs.hypergeometric_pfq_cpu(a, b, z)
