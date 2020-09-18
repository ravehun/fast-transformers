import numpy as np
import torch
from pytorch_lightning.metrics import Metric


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


class MaskedMetric(Metric):
    def __init__(self, metric_func, name):
        self.func = metric_func
        super(MaskedMetric, self).__init__(name)

    def forward(self,
                outputs: dict,
                target: torch.Tensor,
                mask: torch.Tensor,
                eps=1e-6,
                reduction=True,
                *args,
                **kwargs
                ) -> torch.Tensor:
        def _shape_check(x, dim):
            if len(x.shape) != dim:
                raise ValueError(f"expect {dim}, actual {len(x.shape)})")

        # _shape_check(outputs, 2)
        _shape_check(target, 2)
        _shape_check(mask, 2)

        mask = mask.float()
        metric = self.func(target=target, **outputs)
        if reduction:
            metric = (mask * metric).sum(dim=1) / (mask.sum(dim=1) + eps)
            metric = metric.mean()
        else:
            metric = (mask * metric) / (mask.sum(dim=1, keepdim=True) + eps)
        return metric


class MaskedReweightedDiff(MaskedMetric):
    def forward(self,
                outputs: dict,
                target: torch.Tensor,
                mask: torch.Tensor,
                eps=1e-6,
                reduction=True,
                reweight_by=None,
                *args,
                **kwargs
                ):
        original = super().forward(outputs=outputs, target=target, mask=mask, eps=eps, reduction=reduction)
        mask = mask.float() * reweight_by
        reweighted = super().forward(outputs=outputs, target=target, mask=mask, eps=eps, reduction=reduction)
        return {
            "diff": reweighted - original,
            "original": original
        }


class SeqGroupMetric(MaskedMetric):
    TRAIN = 1
    VALID = 2
    PAD = 0

    def forward(self,
                outputs,
                target,
                group,
                meta=None,
                eps=1e-6,
                reduction=True,
                *args, **kwargs):
        train_mask = (group == SeqGroupMetric.TRAIN)
        valid_mask = (group == SeqGroupMetric.VALID)
        train_loss = super(SeqGroupMetric, self).forward(outputs=outputs, target=target,
                                                         mask=train_mask, reduction=reduction)
        with torch.no_grad():
            valid_loss = super(SeqGroupMetric, self).forward(outputs=outputs, target=target,
                                                             mask=valid_mask, reduction=reduction)

        return {
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "train_mask": train_mask,
            "valid_mask": valid_mask,
        }


class MaskedMLABE(SeqGroupMetric):
    def __init__(self):
        super(MaskedMLABE, self).__init__(Loss_functions.mlabe, "MaskMeanLogAbsoluteBoostError")


class MaskedMetricMLABE(MaskedMetric):
    def __init__(self):
        super().__init__(Loss_functions.mlabe, "MaskedMetricMLABE")


class SeqMLABE(SeqGroupMetric):
    def __init__(self):
        super().__init__(Loss_functions.mlabe, "SeqMLABE")


class MaskedReweightedDiffMLABE(MaskedReweightedDiff):
    def __init__(self):
        super(MaskedReweightedDiffMLABE, self).__init__(Loss_functions.mlabe, "MaskedReweightedDiffMLABE")


class MaskedAPE(MaskedMetric):
    def __init__(self):
        super(MaskedAPE, self).__init__(Loss_functions.ape, "MaskAPE")


class APE(SeqGroupMetric):
    def __init__(self):
        super(APE, self).__init__(Loss_functions.ape, "MaskAPE")


class NegtiveLogNormLoss(SeqGroupMetric):
    def __init__(self):
        super(NegtiveLogNormLoss, self).__init__(Loss_functions.nll_lognorm, "LogNormLoss")


class NLLtPdf(SeqGroupMetric):
    def __init__(self):
        super(NLLtPdf, self).__init__(Loss_functions.nll_logt, "NLLtPdf")
