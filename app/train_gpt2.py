import warnings

import click
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import GPT2Config, GPT2Model

warnings.filterwarnings("ignore")

import math
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
import glob
import torch
import pytorch_lightning as pl
import torch.nn as nn

from common_utils import CommonUtils, Mapping

from loss import *


class MyIterableDataset(IterableDataset):

    def __init__(self, fn, train_start_date, valid_start_date, position_oov_size, mapping=None, front_padding_num=0,
                 stock_id_header_token=1, stock_price_start_token=2):
        self.fn = fn
        self.feature = None
        self.valid_start_date = valid_start_date
        self.mapping = mapping
        self.front_padding_num = front_padding_num
        self.stock_id_token = stock_id_header_token
        self.price_start_token = stock_price_start_token
        self.train_start_date = train_start_date
        self.position_oov_size = position_oov_size

    @staticmethod
    def affine(x, inf, sup):
        x = (x.max(axis=1, keepdims=True) - x) / (x.max(axis=1, keepdims=True) - x.min(axis=1, keepdims=True) + 1e-6)
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
            "affine": lambda: MyIterableDataset.affine(x, kwargs["inf"], kwargs["sup"]),
            "log": lambda: (x + 1e-6).log(),
        }

        return f[n_type]()

    def load(self):
        npz = np.load(self.fn)

        def get_group_mask(xs, valid_split):
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

        self.stock_name = npz["stock_name"]

        ab_start_days_offset = CommonUtils.date_to_idx(self.train_start_date)
        relative_valid_start_offset = CommonUtils.date_to_idx(self.valid_start_date) - ab_start_days_offset
        volume = npz["input"][..., 4:5]
        volume = self.norm(volume)
        feature = np.concatenate([npz["input"][..., :4], volume], axis=2)
        label = npz["target"]
        relative_days_offset = (npz["days_offset"] - ab_start_days_offset + self.position_oov_size) * \
                               (npz["days_offset"] > 0).astype(npz["days_offset"].dtype)
        stock_id = self.mapping.name2id(self.stock_name)

        if self.front_padding_num > 0:
            feature = np.pad(feature
                             , [(0, 0), (self.front_padding_num, 0), (0, 0)]
                             , constant_values=0.0)
            label = np.pad(label
                           , [(0, 0), (self.front_padding_num, 0)]
                           , constant_values=0.0)
            relative_days_offset = np.pad(relative_days_offset
                                          , [(0, 0), (self.front_padding_num, 0)]
                                          , constant_values=0.0)
        group_mask = get_group_mask(relative_days_offset, relative_valid_start_offset)
        group_mask = group_mask * (label > 0).astype(group_mask.dtype)
        self.feature = torch.tensor(feature, dtype=torch.float32)
        self.label = torch.tensor(label, dtype=torch.float32)
        self.group = torch.tensor(group_mask, dtype=torch.float32)

        self.stock_id = np.stack([
            np.ones_like(stock_id) * self.stock_id_token,
            stock_id,
            np.ones_like(stock_id) * self.price_start_token,
        ], axis=1)

        self.stock_id = torch.tensor(self.stock_id, dtype=torch.int64)
        self.days_offset = torch.tensor(relative_days_offset, dtype=torch.int64)

    def __iter__(self):
        if self.feature is None:
            self.load()
        indice = np.arange(self.feature.shape[0])
        np.random.shuffle(indice)
        for idx in indice:
            x = self.feature[idx]
            # print(f"x mask {(x[..., 0] > 0).sum(-1)}")
            y = self.label[idx]
            meta = {
                "stock_name": self.stock_name[idx],
                "group": self.group[idx],
                "stock_id": self.stock_id[idx],
                "days_off": self.days_offset[idx],
            }
            yield x, y, meta


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
                 asset_list=None,
                 front_padding_num=2,
                 train_start_date=None,
                 valid_start_date=None,
                 **kwargs):
        super(TimeSeriesTransformer, self).__init__()

        self.valid_start_date = valid_start_date
        project_dimension = query_dimensions * n_heads
        self.project_dimension = project_dimension
        # self.stock_num = stock_num
        # self.stock_mapping_fn = stock_mapping_fn
        # self.model_projection = nn.Conv1d(input_dimensions, project_dimension, 1)
        self.model_projection = nn.Linear(input_dimensions, project_dimension)
        self.seq_len = seq_len + front_padding_num
        # self.loss = MaskedMLABE()
        self.loss = NegtiveLogNormLoss()
        # self.loss = APE()
        self.file_re = file_re
        self.batch_size = batch_size
        self.metric_object = MaskedReweightedDiffMLABE()
        self.output_projection = nn.Linear(n_heads * value_dimensions, 2)
        # self.output_projection = nn.Conv1d(n_heads * value_dimensions, 1, 1)
        if type(file_re) != dict:
            self.filenames = glob.glob(file_re)
        self.lr = lr
        self.seed = seed
        np.random.seed(seed)
        self.asset_list = asset_list
        self.stock_oov_size = 10
        self.position_oov_size = 10
        self.mapping = Mapping().load(self.asset_list, self.stock_oov_size)
        self.num_workers = num_workers
        self.stock_embedings_length = len(self.mapping) + self.stock_oov_size
        self.stock_embeddings = torch.nn.Embedding(self.stock_embedings_length, project_dimension)
        self.front_padding_num = front_padding_num
        self.stock_id_header_token = 1
        self.stock_price_start_token = 2
        self.train_start_date = train_start_date
        self.n_positions = int(self.seq_len * 366 / 250) + self.position_oov_size
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
            n_positions=self.n_positions,
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

    def attach_head(self, x, meta, seq_mask):
        # print(f"seq_mask {seq_mask.sum(-1)}")
        relative_days_off = meta["days_off"]
        b, s = relative_days_off.shape[:2]

        if self.front_padding_num > 0:
            stock_id = meta["stock_id"]
            stock_embedding = self.stock_embeddings(stock_id)
            x[:, :self.front_padding_num + 1, :] = stock_embedding
            # open price start token
            seq_mask[..., :self.front_padding_num + 1] = torch.ones_like(seq_mask[..., :self.front_padding_num + 1])

            relative_days_off[..., :self.front_padding_num] = torch.arange(1, self.front_padding_num + 1).repeat(
                [b, 1])
        return x, seq_mask, relative_days_off

    def forward(self, x, meta, **transformer_kwargs):
        seq_mask = (x[..., 0] > 0.1)

        x = self.model_projection(x) * math.sqrt(self.project_dimension)
        x, seq_mask, relative_days_off = self.attach_head(x, meta, seq_mask)

        regress_embeddings, _ = self.transformer(inputs_embeds=x,
                                                 attention_mask=seq_mask,
                                                 position_ids=relative_days_off,
                                                 **transformer_kwargs)
        mu, sigma = self.output_projection(regress_embeddings).split(1, -1)
        return {
            "mu": mu.squeeze(-1),
            "ln_sigma": sigma.squeeze(-1),
        }

    def pred_with_attention(self, x, meta=None, **transformer_kwargs):
        with torch.no_grad():
            x = self.model_projection(x) * math.sqrt(self.project_dimension)
            x, seq_mask, relative_days_off = self.attach_head(x, meta)

            regress_embeddings, cache, attention = self.transformer(inputs_embeds=x,
                                                                    attention_mask=seq_mask,
                                                                    position_ids=relative_days_off,
                                                                    output_attentions=True,
                                                                    **transformer_kwargs)

            mu, sigma = self.output_projection(regress_embeddings).split(1, -1)
        return {
            "mu": mu.squeeze(-1),
            "ln_sigma": sigma.squeeze(-1),
            "cache": cache,
            "attention": attention
        }

    def reset(self):
        pass

    def training_step(self, batch, bn):
        x, target, meta = batch
        model_output = self(x, meta)
        loss_output = self.loss(model_output, target, group=meta["group"])
        # with torch.no_grad():
        #     metric_output = self.metric_object(
        #         outputs={
        #             "pred": model_output["mu"].exp()
        #         },
        #         target=target,
        #         mask=loss_output["valid_mask"],
        #         reweight_by=((-model_output["ln_sigma"] + (1 - loss_output["valid_mask"].float()) * -1e6)).softmax(
        #             dim=1),
        #     )
        log = {
            "train_loss": loss_output["train_loss"]
            , "valid_loss": loss_output["valid_loss"]

        }

        ret = {
            'loss': loss_output["train_loss"],
            "valid_loss": loss_output["valid_loss"],
            # "diff": metric_output["diff"],
            # "original": metric_output["original"],
            "log": log,
        }

        return ret

    def training_epoch_end(
            self,
            outputs
    ):
        def _agg_by_key(key):
            return torch.stack([x[key] for x in outputs]).mean()

        keys = ["loss", "valid_loss",
                # "diff", 'original'
                ]
        train_loss, valid_loss = [_agg_by_key(key) for key in keys]
        print(f"tr: {train_loss:.6f}, va: {valid_loss:.6f}")
        # print(f"train loss: {train_loss:.6f}, valid_loss: {valid_loss:.6f}, diff: {diff:.6f} original: {original:.6f}")
        return {"train_loss": train_loss, "val_loss": valid_loss}

    def configure_optimizers(self):
        # from radam import RAdam
        # optimizer = RAdam(self.parameters())
        from torch.optim import Adam
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer

    def train_dataloader(self) -> DataLoader:
        dataset = MyIterableDataset(
            self.filenames[0]
            , mapping=self.mapping
            , front_padding_num=self.front_padding_num
            , train_start_date=self.train_start_date
            , valid_start_date=self.valid_start_date
            , position_oov_size=self.position_oov_size
            , stock_id_header_token=self.stock_id_header_token
            , stock_price_start_token=self.stock_price_start_token
        )
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)


@click.command()
@click.option('--file-re', type=str, default='../data/sample_record_npz/*.npz', show_default=True, help="file regex")
@click.option('--batch-size', type=int, default=1, show_default=True, help="batch size")
@click.option('--attention-type', type=str, default='full', show_default=True, help="file regex")
@click.option('--gpus', type=int, default=None, show_default=True, help="gpu num")
@click.option('--accumulate_grad_batches', type=int, default=11, show_default=True, help="update with N batches")
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
                                  asset_list='../data/asset_list.txt',
                                  front_padding_num=2,
                                  train_start_date="2009-01-01",
                                  valid_start_date="2013-01-01",
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
        checkpoint_callback=ModelCheckpoint(
            monitor='valid_loss',
        ),
        # early_stop_callback=TrainEarlyStopping(patience=5),
        # weights_save_path='lightning_logs1',
        # resume_from_checkpoint='lightning_logs/version_8/checkpoints/epoch=2.ckpt'
        # precision=16,
        # tpu_cores=8,

    )

    trainer.fit(model)
    # TimeSeriesTransformer.load_from_checkpoint()


if __name__ == "__main__":
    train()
