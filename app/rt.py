import cppimport
import numpy as np

funcs = cppimport.imp('ts_prediction.libarb.arb_wrapper')
from datetime import datetime

t = 1500
a = np.array([[2, 2, 1]]).repeat(t, 0)
b = np.array([[2, 2, 3, 3]]).repeat(t, 0)
z = np.array(4).repeat(t)
now = datetime.now()
res = funcs.hypergeometric_pfq_cpu(a, b, z)
print(datetime.now() - now)
# print(res)
