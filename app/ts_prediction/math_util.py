import numpy as np
import torch
from numpy import pi
from scipy.special import iv

from app.ts_prediction.common_utils import ModifiedBesselKv, ModifiedBesselKve

besselK = ModifiedBesselKv.apply
besselKe = ModifiedBesselKve.apply

from ts_prediction.libarb import hypergeometric_pfq

from scipy.special import digamma, gamma


class Loss_functions():

    @staticmethod
    def ape(pred, target, eps=1e-6):
        return 100 * abs((pred - target) / (target + eps))

    @staticmethod
    def nll_lognorm(mu, ln_sigma, target, eps=1e-6):
        return ln_sigma + (((target + eps).log() - mu) / ln_sigma.exp()) ** 2 / 2

    @staticmethod
    def nll_norm(mu, ln_sigma, target, eps=1e-6):
        return ln_sigma + (((target + eps) - mu) / ln_sigma.exp()) ** 2 / 2

    @staticmethod
    def mlabe(pred, target, eps=1e-6):
        return torch.log(((pred - target) / (target + eps)).abs() + 1)

    @staticmethod
    def nll_logt(target, df, loc, scale):
        return -(
                torch.lgamma((df + 1) / 2) - torch.lgamma(df / 2)
                + 0.5 * torch.log(scale / np.pi / df)
                - (df + 1) / 2 * torch.log(1 + scale / df * ((target - loc) ** 2))
        )

    @staticmethod
    def dist2(target, pred):
        return (target - pred) ** 2

    @staticmethod
    def dist1(target, pred):
        return (target - pred).abs()

    @staticmethod
    def nll_dghb6(
            target, mu, gamma, pc, chi, ld,
    ):
        '''

        Parameters
        ----------
        target
        mu  R
        gamma R+
        pc R+
        chi R+
        ld R

        Returns
        -------

        '''
        sigma = 1
        gmc = gamma * sigma
        q = (target - mu) / sigma
        alpha = pc + gmc ** 2
        chi_q = chi + q ** 2
        xp_z = (chi_q * alpha).sqrt()

        # print(f"bessel ratio {besselK(xp_z, ld - 0.5).log() - besselK((chi * pc) ** 0.5, ld).log()} {besselK(xp_z, ld - 0.5).log()} {besselK((chi * pc) ** 0.5, ld).log()} "
        #       f"\n xp, z  {chi_q}, {alpha}"
        #       f"\n front {-(0.5 * ld * (pc / chi).log() + (0.5 - ld) * alpha.log() - 0.5 * np.log(2 * np.pi) + (ld - 0.5) * xp_z.log() + q * gmc)}"
        #       )
        return -(0.5 * ld * (pc / chi).log() + (0.5 - ld) * alpha.log() - 0.5 * np.log(2 * np.pi) \
                 + (ld - 0.5) * xp_z.log() + q * gmc \
                 + besselK(xp_z, ld - 0.5).log() - besselK((chi * pc) ** 0.5, ld).log()
                 )

    @staticmethod
    def nll_dghb(target,
                 ld,
                 alpha,
                 beta,
                 delta,
                 mu, ):
        gamma = (alpha ** 2 - beta ** 2).sqrt()
        y = (delta ** 2 + (target - mu) ** 2).sqrt()
        bx = beta * (target - mu)

        besselRatio = (besselK(y * alpha, ld - 1 / 2) / besselK(delta * gamma, ld)).log()
        st = ld * (gamma / delta).log()
        ft = (ld - 0.5) * (y / alpha).log() - 0.5 * np.log((2 * np.pi)) + bx

        return - (besselRatio + ft + st)

    @staticmethod
    def gh_ex(mu, gamma, pc, chi, ld):
        chi_s = chi.sqrt()
        return mu + (chi_s * gamma * besselK(chi_s * pc, ld + 1)) / (pc * besselK(chi_s * pc, ld))

    @staticmethod
    def d_kv_2_order(z: np.ndarray, v: np.ndarray):
        N = z.shape[0]

        ivz = iv(v, z)
        imvz = iv(-v, z)
        z_square = np.square(z)
        t1 = 0.5 * pi / np.sin(pi * v)
        t2 = pi / np.tan(pi * v) * ivz
        t3 = ivz + imvz
        t14 = z_square / 4 / (1 - np.square(v))

        # vfunc = np.vectorize(lambda v, z_square: hyper([1, 1, 1.5], [2, 2, 2 - v, 2 + v], z_square))
        # t4 = vfunc(v, z_square)
        t4 = hypergeometric_pfq(np.array([[1, 1, 1.5]]).repeat(N, 0),
                                np.stack([2 * np.ones_like(v), 2 * np.ones_like(v), 2 - v, 2 + v], 1),
                                z_square
                                )
        # t4 = hyper([1, 1, 1.5], [2, 2, 2 - v, 2 + v], z_square)
        t5 = np.log(0.5 * z) - digamma(v) - 0.5 * v
        t6 = imvz
        t7 = np.square(gamma(-v))
        t8 = (0.5 * z) ** (2 * v)
        # vfunc = np.vectorize(lambda v, z_square: hyp2f3(v, 0.5 + v, 1 + v, 1 + v, 1 + 2 * v, z_square))
        # t9 = vfunc(v, z_square)
        # # t9 = hyp2f3(v, 0.5 + v, 1 + v, 1 + v, 1 + 2 * v, z_square)
        t9 = hypergeometric_pfq(
            np.stack([v, 0.5 + v], 1),
            np.stack([1 + v, 1 + v, 1 + 2 * v], 1),
            z_square
        )
        t10 = ivz
        t11 = np.square(gamma(v))
        t12 = (z / 2) ** (-2 * v)
        # vfunc = np.vectorize(lambda v, z_square: hyp2f3(-v, 0.5 - v, 1 - v, 1 - v, 1 - 2 * v, z_square))
        # t13 = vfunc(v, z_square)
        # t13 = hyp2f3(-v, 0.5 - v, 1 - v, 1 - v, 1 - 2 * v, z_square)
        t13 = hypergeometric_pfq(
            np.stack([-v, 0.5 - v, ], 1),
            np.stack([1 - v, 1 - v, 1 - 2 * v], 1),
            z_square
        )
        return t1 * (t2 - t3 * (t14 * t4) + t5) \
               + 0.25 * (t6 * t7 * t8 * t9 - t10 * t11 * t12 * t13)
