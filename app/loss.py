import torch
from pytorch_lightning.metrics import Metric


class MaskedMetric(Metric):
    def __init__(self, metric_func, name):
        self.func = metric_func
        super(MaskedMetric, self).__init__(name)

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                mask: torch.Tensor = None,
                eps=1e-6,
                reduction=True,
                *args,
                **kwargs
                ) -> torch.Tensor:
        if len(pred.shape) != 2:
            raise ValueError("pred dim needs to be 2")
        if mask is None:
            mask = (target > 0)
        mask = mask.float()
        metric = self.func(pred, target)
        metric = (mask * metric).sum(dim=1) / (mask.sum(dim=1) + eps)
        if reduction:
            metric = metric.mean()

        return metric


class MaskedRMSLE(MaskedMetric):
    def __init__(self):
        super(MaskedRMSLE, self).__init__(self.rsmle, "MaskRmsle")

    @staticmethod
    def rsmle(pred, target):
        return torch.log((pred - target).abs() + 1) - torch.log(target + 1)


class SeqGroupMetric(MaskedMetric):
    TRAIN = 1
    VALID = 2
    PAD = 0

    def forward(self,
                pred,
                target,
                meta=None,
                eps=1e-6,
                reduction=True,
                *args, **kwargs):
        group = meta["group"]
        train_mask = (group == SeqGroupMetric.TRAIN)
        valid_mask = (group == SeqGroupMetric.VALID)
        train_loss = super(SeqGroupMetric, self).forward(pred, target, train_mask, reduction=reduction)
        with torch.no_grad():
            valid_loss = super(SeqGroupMetric, self).forward(pred, target, valid_mask, reduction=reduction)

        return {
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "train_mask": train_mask,
            "valid_mask": valid_mask,
        }


class MaskedMLABE(SeqGroupMetric):
    def __init__(self):
        super(MaskedMLABE, self).__init__(self.mlabe, "MaskMeanLogAbsoluteBoostError")

    @staticmethod
    def mlabe(pred, target, eps=1e-6):
        return torch.log(((pred - target) / (target + eps)).abs() + 1)


class MaskedAPE(MaskedMetric):
    def __init__(self):
        super(MaskedAPE, self).__init__(self.ape, "MaskAPE")

    @staticmethod
    def ape(pred, target, eps=1e-6):
        return 100 * abs((pred - target) / (target + eps))


class APE(SeqGroupMetric):
    def __init__(self):
        super(APE, self).__init__(self.ape, "MaskAPE")

    @staticmethod
    def ape(pred, target, eps=1e-6):
        return 100 * abs((pred - target) /   (target + eps))
