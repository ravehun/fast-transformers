import warnings

import click
import tqdm
from pytorch_lightning.metrics import RMSLE, Metric
from pytorch_lightning.metrics.functional import mse

from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.masking import TriangularCausalMask, LengthMask

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


class MaskedMetric(Metric):
    def __init__(self, metric_func, name):
        self.func = metric_func
        super(MaskedMetric, self).__init__(name)

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                eps= 1e-9
                ) -> torch.Tensor:
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


class MyIterableDataset(IterableDataset):

    def __init__(self, fn):
        self.fn = fn
        self.feature = None
    def load(self):
        npz = np.load(self.fn)

        self.feature = torch.tensor(npz["input"], dtype=torch.float32)
        self.label = torch.tensor(npz["target"], dtype=torch.float32)
        self.stock_name = npz["stock_name"]

    def __iter__(self):
        if self.feature is None:
            self.load()
        indice = np.arange(self.feature.shape[0])
        np.random.shuffle(indice)
        for idx in indice:
            x = self.feature[idx]
            y = self.label[idx]
            meta = self.stock_name[idx]
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
    def ape(pred, target, eps=1e-9):
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

        builder = TransformerEncoderBuilder.from_kwargs(
            n_layers=n_layers,
            n_heads=n_heads,
            query_dimensions=query_dimensions,
            value_dimensions=value_dimensions,
            feed_forward_dimensions=feed_forward_dimensions
        )
        # Build a transformer with softmax attention
        builder.attention_type = attention_type

        self.transformer = builder.get()
        project_dimension = query_dimensions * n_heads
        self.project_dimension = project_dimension
        # self.stock_num = stock_num
        # self.stock_mapping_fn = stock_mapping_fn
        # self.model_projection = nn.Conv1d(input_dimensions, project_dimension, 1)
        self.model_projection = nn.Linear(input_dimensions, project_dimension)
        self.positional_encoder = PositionalEncoding(project_dimension)
        self.seq_len = seq_len
        self.loss = MaskedAPE()
        self.file_re = file_re
        self.batch_size = batch_size
        self.metric_object = MaskedAPE()
        self.output_projection = nn.Linear(n_heads * value_dimensions, 1)
        # self.output_projection = nn.Conv1d(n_heads * value_dimensions, 1, 1)
        self.filenames = glob.glob(self.file_re)
        self.lr = lr
        self.seed = seed
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

    def forward(self, x, meta):
        ar = torch.arange(self.seq_len).float().type_as(x)
        relative_pos = self.positional_encoder(ar).unsqueeze(0).repeat(
            [x.shape[0], 1, 1])

        att_mask = TriangularCausalMask(self.seq_len)
        seq_mask = (x[..., 0] > 0).sum(dim=1).long()
        seq_mask = LengthMask(seq_mask, max_len=self.seq_len)

        x = self.model_projection(x) * math.sqrt(self.project_dimension)

        x = x + relative_pos
        regress_embeddings = self.transformer(x, att_mask, seq_mask)
        pred = self.output_projection(regress_embeddings)

        return pred

    def reset(self):
        pass

    def training_step(self, batch, bn):
        x, y, meta = batch
        pred = self(x, meta)
        loss = self.loss.forward(pred, y.unsqueeze(-1))
        log = {"train_loss": loss, }
        return {'loss': loss, "log": log}

    def training_epoch_end(
            self,
            outputs
    ):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        print(f"train loss: {loss}")
        return {"train_loss": loss}
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
        from radam import RAdam
        optimizer = RAdam(self.parameters())
        return optimizer


    def train_dataloader(self) -> DataLoader:
        dataset = MyIterableDataset(
            self.filenames[0]
        )
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)

@click.command()
@click.option('--file-re', type=str, default='../data/sample_record_npz/*.npz', show_default=True, help="file regex")
@click.option('--batch-size', type=int, default=2, show_default=True, help="batch size")
@click.option('--attention-type', type=str, default='full', show_default=True, help="file regex")
def train(file_re,batch_size,attention_type):
    torch.cuda.empty_cache()
    model = TimeSeriesTransformer(input_dimensions=5,
                                  file_re=file_re,
                                  # project_dimension=128,
                                  n_layers=8,
                                  n_heads=8,
                                  query_dimensions=64,
                                  value_dimensions=64,
                                  feed_forward_dimensions=1024,
                                  attention_type=attention_type,
                                  num_workers=1,
                                  batch_size=batch_size,
                                  seq_len=1500,
                                  seed=101,
                                  lr=1e-7)

    trainer = pl.Trainer(
        max_epochs=100,
        # limit_val_batches=1.0,
        # reload_dataloaders_every_epoch=True,
        # fast_dev_run=True,
        terminate_on_nan=True,
        # auto_lr_find=True,
        # use_amp=False,  # todo remove for gpu
    )

    trainer.fit(model)


if __name__ == "__main__":
    train()