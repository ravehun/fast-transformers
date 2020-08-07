import warnings

import click
import tqdm
from pytorch_lightning.metrics import RMSLE, Metric
from pytorch_lightning.metrics.functional import mse
from torch.optim import SGD
from transformers import GPT2Config, GPT2Model

warnings.filterwarnings("ignore")

import math
import random
import pandas as pd
from torch import functional as F
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
import glob
import torch
import pytorch_lightning as pl
import torch.nn as nn

from common_utils import CommonUtils


class MaskedMetric(Metric):
    def __init__(self, metric_func, name):
        self.func = metric_func
        super(MaskedMetric, self).__init__(name)

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                eps=1e-6,
                *args,
                **kwargs
                ) -> torch.Tensor:
        if len(pred.shape) != 2:
            raise ValueError("pred dim needs to be 2")
        mask = (target > 0).long()

        metric = self.func(pred, target)
        metric = (mask * metric).sum(dim=1) / (mask.sum(dim=1) + eps)
        metric = metric.mean()

        return metric


class MaskedRMSLE(MaskedMetric):
    def __init__(self):
        super(MaskedRMSLE, self).__init__(self.rsmle, "MaskRmsle")

    @staticmethod
    def rsmle(pred, target):
        return mse(torch.log(pred + 1), torch.log(target + 1))


class SeqGroupMetric(MaskedMetric):
    TRAIN = 1
    VALID = 2
    PAD = 0

    def forward(self,
                pred,
                target,
                meta=None,
                eps=1e-6,
                *args, **kwargs):
        group = meta["group"]
        train_mask = (group == SeqGroupMetric.TRAIN)
        valid_mask = (group == SeqGroupMetric.VALID)

        train_loss = super(SeqGroupMetric, self).forward(pred, target, train_mask)

        with torch.no_grad():
            valid_loss = super(SeqGroupMetric, self).forward(pred, target, valid_mask)

        return {
            "train_loss": train_loss,
            "valid_loss": valid_loss,
        }


class MyIterableDataset(IterableDataset):

    def __init__(self, fn, valid_start_date="2013-01-01"):
        self.fn = fn
        self.feature = None
        self.valid_start_date = valid_start_date

    @staticmethod
    def affine(x, inf, sup):
        x = (x.max(axis=1, keepdims=True) - x) / (x.max(axis=1, keepdims=True) - x.min(axis=1, keepdims=True))
        x = x * (sup - inf) + inf
        return x

    @staticmethod
    def norm(x: np.ndarray, n_type="min_max", **kwargs):
        if len(x.shape) != 3:
            raise ValueError(f"expect shape is 3, actual {x.shape}")

        f = {
            "min_max": lambda: (x.max(axis=1, keepdims=True) - x) / (
                    x.max(axis=1, keepdims=True) - x.min(axis=1, keepdims=True)),
            "z_score": lambda: (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6),
            "affine": lambda: MyIterableDataset.affine(x, kwargs["inf"], kwargs["sup"])
        }

        return f[n_type]()

    def load(self):
        npz = np.load(self.fn)

        def group_mask(xs, valid_split):
            origin_shape = xs.shape
            def get_group(x):
                if x == 0:
                    return SeqGroupMetric.PAD
                if x < valid_split:
                    return SeqGroupMetric.TRAIN
                else:
                    return SeqGroupMetric.VALID

            out = [get_group(x) for x in xs.flatten()]

            return np.array(out).reshape(origin_shape)

        volume = npz["input"][..., 4:5]
        volume = self.norm(volume)
        feature = np.concatenate([npz["input"][..., :4], volume], axis=2)
        self.feature = torch.tensor(feature, dtype=torch.float32)
        self.label = torch.tensor(npz["target"], dtype=torch.float32)
        days_offset = npz["days_offset"]
        valid_start_offset = CommonUtils.date_to_idx(self.valid_start_date)
        self.group = torch.tensor(group_mask(days_offset, valid_start_offset), dtype=torch.float32)
        self.stock_name = npz["stock_name"]

    def __iter__(self):
        if self.feature is None:
            self.load()
        indice = np.arange(self.feature.shape[0])
        np.random.shuffle(indice)
        for idx in indice:
            x = self.feature[idx]
            y = self.label[idx]
            meta = {
                "stock_id": self.stock_name[idx],
                "group": self.group[idx],
            }
            yield x, y, meta


class ToTensor():
    def __call__(self, xs):
        return tuple(torch.tensor(x, dtype=torch.float32) for x in xs)

    def __repr__(self):
        return self.__class__.__name__ + '()'


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


class PositionalEncoding(nn.Module):
    "Encode the position with a sinusoid."

    def __init__(self, d: int):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('freq', 1 / (10000 ** (torch.arange(0., d, 2.) / d)))

    def forward(self, pos: torch.Tensor):
        inp = torch.ger(pos, self.freq)
        enc = torch.cat([inp.sin(), inp.cos()], dim=-1)
        return enc


class TimeSeriesTransformer(pl.LightningModule):
    def __init__(self,
                 file_re,
                 input_dimensions=5,
                 # project_dimension=128,
                 n_layers=8,
                 n_heads=8,
                 query_dimensions=64,
                 value_dimensions=64,
                 feed_forward_dimensions=1024,
                 attention_type='full',
                 num_workers=1,
                 batch_size=2,
                 lr=1e-7,
                 seq_len=1000,
                 seed=100,
                 **kwargs):
        super(TimeSeriesTransformer, self).__init__()

        project_dimension = query_dimensions * n_heads
        self.project_dimension = project_dimension
        # self.stock_num = stock_num
        # self.stock_mapping_fn = stock_mapping_fn
        # self.model_projection = nn.Conv1d(input_dimensions, project_dimension, 1)
        self.model_projection = nn.Linear(input_dimensions, project_dimension)
        self.positional_encoder = PositionalEncoding(project_dimension)
        self.seq_len = seq_len
        self.loss = APE()
        self.file_re = file_re
        self.batch_size = batch_size
        self.metric_object = APE()
        self.output_projection = nn.Linear(n_heads * value_dimensions, 1)
        # self.output_projection = nn.Conv1d(n_heads * value_dimensions, 1, 1)
        self.filenames = glob.glob(self.file_re)
        self.lr = lr
        self.seed = seed
        self.pre_normalize = nn.LayerNorm(normalized_shape=[self.seq_len, input_dimensions])
        np.random.seed(seed)
        # self.split_date = split_date
        # self.end_date = end_date
        # random.shuffle(self.filenames)
        # self.training_files = self.filenames
        # self.valid_files = glob.glob(valid_file_re)
        # if not self.training_files:
        #     # if not self.training_files or (not self.valid_files):
        #     raise ValueError(f"input file train {self.training_files} is empty")
        self.num_workers = num_workers
        # self.metric = MaskedABE()

        config = GPT2Config(
            batch_size=self.batch_size,
            vocab_size=2,
            n_embd=project_dimension,
            n_layer=n_layers,
            n_head=n_heads,
            # intermediate_size=intermediate_size,
            # hidden_act=hidden_act,
            # hidden_dropout_prob=hidden_dropout_prob,
            # attention_probs_dropout_prob=attention_probs_dropout_prob,
            n_positions=self.seq_len,
            n_ctx=self.seq_len,
            # type_vocab_size=type_vocab_size,
            # initializer_range=initializer_range,
            # bos_token_id=,
            # eos_token_id=eos_token_id,
            return_dict=True,
        )

        self.transformer = GPT2Model(config=config)

    @property
    def get_device(self):
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'

    def forward(self, x, meta):
        # ar = torch.arange(self.seq_len).float().type_as(x)
        # relative_pos = self.positional_encoder(ar).unsqueeze(0).repeat(
        #     [x.shape[0], 1, 1])

        seq_mask = (x[..., 0] > 0)
        # x = self.pre_normalize(x)
        x = self.model_projection(x) * math.sqrt(self.project_dimension)

        # x = x + relative_pos
        regress_embeddings, _ = self.transformer(inputs_embeds=x, attention_mask=seq_mask)
        pred = self.output_projection(regress_embeddings)

        return pred

    def reset(self):
        pass

    def training_step(self, batch, bn):
        x, y, meta = batch
        pred = self(x, meta)
        pred = pred.squeeze(-1)
        output = self.loss.forward(pred, y, meta)
        log = {"train_loss": output["train_loss"], "valid_loss": output["valid_loss"]}

        return {
            'loss': output["train_loss"],
            "valid_loss": output["valid_loss"],
            "log": log,
        }

    def training_epoch_end(
            self,
            outputs
    ):
        train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        valid_loss = torch.stack([x["valid_loss"] for x in outputs]).mean()
        print(f"train loss: {train_loss}, valid_loss: {valid_loss}")
        return {"train_loss": train_loss, "val_loss": valid_loss}

    #
    # def validation_step(self, batch, batch_nb):
    #     x, y, meta = batch
    #     y_hat = self(x, meta)
    #     loss = self.loss.forward(y_hat, y)
    #     mae = self.metric_object(y_hat, y)
    #     log = {"val_loss": loss, 'mae': mae}
    #     return {
    #         'val_loss': loss,
    #         'mae': mae,
    #         'log': log
    #     }

    # def validation_epoch_end(self, outputs):
    #
    #     if len(outputs) > 0:
    #
    #         val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    #         mae = torch.stack([x["mae"] for x in outputs]).mean()
    #
    #
    #     else:
    #         val_loss = torch.Tensor(0)
    #         mae = torch.Tensor(0)
    #     print(f"epoch {self.current_epoch}, val_loss {val_loss}, mae {mae}")
    #     return {"val_loss": val_loss}

    def configure_optimizers(self):
        # from radam import RAdam
        # optimizer = RAdam(self.parameters())
        from torch.optim import Adam
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer

    def train_dataloader(self) -> DataLoader:
        dataset = MyIterableDataset(
            self.filenames[0]
        )
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)


@click.command()
@click.option('--file-re', type=str, default='../data/sample_record_npz/*.npz', show_default=True, help="file regex")
@click.option('--batch-size', type=int, default=1, show_default=True, help="batch size")
@click.option('--attention-type', type=str, default='full', show_default=True, help="file regex")
@click.option('--gpus', type=int, default=None, show_default=True, help="gpu num")
@click.option('--accumulate_grad_batches', type=int, default=1, show_default=True, help="update with N batches")
@click.option('--auto_lr_find', type=bool, default=False, show_default=True, help="auto_lr_find")
def train(file_re, batch_size, attention_type, gpus, accumulate_grad_batches, auto_lr_find):
    torch.cuda.empty_cache()
    model = TimeSeriesTransformer(input_dimensions=5,
                                  file_re=file_re,
                                  # project_dimension=128,
                                  n_layers=2,
                                  n_heads=8,
                                  query_dimensions=64,
                                  value_dimensions=64,
                                  feed_forward_dimensions=1024,
                                  attention_type=attention_type,
                                  num_workers=0,
                                  batch_size=batch_size,
                                  seq_len=1500,
                                  seed=101,
                                  lr=1e-4,
                                  )

    trainer = pl.Trainer(
        max_epochs=100,
        # limit_val_batches=1.0,
        # reload_dataloaders_every_epoch=True,
        # fast_dev_run=True,
        terminate_on_nan=True,
        gpus=gpus,
        auto_lr_find=auto_lr_find,
        # use_amp=False,
        gradient_clip_val=2.5,
        accumulate_grad_batches=accumulate_grad_batches,
        # precision=16,
    )

    trainer.fit(model)


if __name__ == "__main__":
    train()
