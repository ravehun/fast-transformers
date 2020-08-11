import torch
from pytorch_lightning.metrics import Metric


class MaskedMetric(Metric):
    def __init__(self, metric_func, name):
        self.func = metric_func
        super(MaskedMetric, self).__init__(name)
    @staticmethod
    def tril_softmax(mask):

        l = mask.shape[1]
        tril = torch.tril(torch.ones(l, l))
        mask = mask.exp()
        tril_normed = torch.einsum('sl,nl->ns', tril, mask)
        return mask / tril_normed

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                confidence: torch.Tensor,
                mask: torch.Tensor,
                eps=1e-6,
                reduction=True,
                *args,
                **kwargs
                ) -> torch.Tensor:
        if len(pred.shape) != 2:
            raise ValueError("pred dim needs to be 2")

        mask = (1 - mask.float()) * -1e8
        mask = mask + confidence

        mask = self.tril_softmax(mask)
        # mask = torch.nn.Softmax(dim=1)(mask)

        metric = self.func(pred, target)
        metric = (mask * metric).sum(dim=1)
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
                confidence,
                target,
                group,
                meta=None,
                eps=1e-6,
                reduction=True,
                *args, **kwargs):
        # group = meta["group"]
        train_mask = (group == SeqGroupMetric.TRAIN)
        valid_mask = (group == SeqGroupMetric.VALID)
        train_loss = super(SeqGroupMetric, self).forward(pred, confidence, target, train_mask, reduction=reduction)
        with torch.no_grad():
            valid_loss = super(SeqGroupMetric, self).forward(pred, confidence, target, valid_mask, reduction=reduction)

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
        return 100 * abs((pred - target) / (target + eps))
