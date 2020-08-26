import os
import warnings

import click
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import Parameter
from transformers import GPT2Config, GPT2Model

warnings.filterwarnings("ignore")

import math
from torch.utils.data import DataLoader
import numpy as np
import glob
import torch
import pytorch_lightning as pl
import torch.nn as nn

from common_utils import Mapping

from loss import *
from dataset import *


class TimeSeriesTransformer(pl.LightningModule):
    def __init__(self,
                 stock_fn,
                 market_fn,
                 stock_input_dimensions=5,
                 # project_dimension=128,
                 temporal_n_layers=8,
                 temporal_n_heads=8,
                 temporal_query_dimensions=64,
                 temporal_value_dimensions=64,
                 spatial_n_layers=4,
                 spatial_n_heads=8,
                 spatial_query_dimensions=64,
                 spatial_value_dimensions=64,
                 num_workers=1,
                 batch_size=2,
                 lr=1e-7,
                 seq_len=1000,
                 seed=100,
                 asset_list=None,
                 etf_list=None,
                 front_padding_num=2,
                 train_start_date=None,
                 valid_start_date=None,
                 **kwargs):
        super(TimeSeriesTransformer, self).__init__()
        self.market_fn = market_fn
        self.valid_start_date = valid_start_date
        temporal_project_dimension = temporal_query_dimensions * temporal_n_heads

        spatial_project_dimension = spatial_n_heads * spatial_query_dimensions
        self.project_dimension = temporal_project_dimension
        self.temporal_input_projection = nn.Linear(stock_input_dimensions, temporal_project_dimension)
        # self.spatial_input_projection = nn.Linear(stock_input_dimensions, spatial_project_dimension)
        self.seq_len = seq_len + front_padding_num
        self.loss = NegtiveLogNormLoss()
        self.file_re = stock_fn
        self.batch_size = batch_size
        self.metric_object = MaskedMetricMLABE()
        self.output_projection = nn.Linear(temporal_n_heads * temporal_value_dimensions, 2)
        if type(stock_fn) != dict:
            self.filenames = glob.glob(stock_fn)
        self.lr = lr
        self.seed = seed
        np.random.seed(seed)
        self.asset_list = asset_list
        self.stock_oov_size = 10
        self.position_oov_size = 10
        self.stock_mapping = Mapping().load(self.asset_list, self.stock_oov_size)
        self.etf_mapping = Mapping().load(etf_list, 0)
        self.etf_nums = len(self.etf_mapping.n2i)
        self.num_workers = num_workers
        self.stock_num = len(self.stock_mapping)
        self.stock_embeddings_length = self.stock_num + self.stock_oov_size
        # represents bias of each stock in temporal dimension
        self.temporal_stock_embeddings = torch.nn.Embedding(self.stock_embeddings_length
                                                            , temporal_project_dimension)
        self.register_parameter("weight", Parameter(
            torch.zeros(self.stock_num, stock_input_dimensions, spatial_project_dimension),
        ))
        self.register_parameter("bias", Parameter(
            torch.zeros(self.stock_num, spatial_project_dimension),
        ))
        self.spatial2temporal_projection = nn.Linear(spatial_project_dimension, temporal_project_dimension)
        self.front_padding_num = front_padding_num
        self.stock_id_header_token = 1
        self.stock_price_start_token = 2
        self.train_start_date = train_start_date
        self.n_positions = int(self.seq_len * 366 / 250) + self.position_oov_size

        self.spatial_etf_ids = torch.arange(self.etf_nums)[None, :].repeat(self.seq_len, 1)

        temporal_gpt2_config = GPT2Config(
            batch_size=self.batch_size,
            vocab_size=2,
            n_embd=temporal_project_dimension,
            n_layer=temporal_n_layers,
            n_head=temporal_n_heads,
            # intermediate_size=intermediate_size,
            # hidden_act=hidden_act,
            # hidden_dropout_prob=hidden_dropout_prob,
            # attention_probs_dropout_prob=attention_probs_dropout_prob,
            n_positions=self.n_positions,
            n_ctx=self.seq_len,
            return_dict=True,
        )
        spatial_gpt2_config = GPT2Config(
            batch_size=self.seq_len,
            vocab_size=2,
            n_embd=spatial_project_dimension,
            n_layer=spatial_n_layers,
            n_head=spatial_n_heads,
            # intermediate_size=intermediate_size,
            # hidden_act=hidden_act,
            # hidden_dropout_prob=hidden_dropout_prob,
            # attention_probs_dropout_prob=attention_probs_dropout_prob,
            n_positions=self.etf_nums,
            n_ctx=seq_len,
            # type_vocab_size=type_vocab_size,
            # initializer_range=initializer_range,
            # bos_token_id=bos_token_id,
            # eos_token_id=eos_token_id,
            attention_type="full",
            return_dict=True,

        )

        self.temporal_transformer = GPT2Model(config=temporal_gpt2_config)
        self.spatial_transformer = GPT2Model(config=spatial_gpt2_config)

    @property
    def get_device(self):
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'

    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

            init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def attach_temporal_head(self, x, meta, seq_mask):
        # print(f"seq_mask {seq_mask.sum(-1)}")
        relative_days_off = meta["days_off"]
        b, s = relative_days_off.shape[:2]

        if self.front_padding_num > 0:
            stock_id = meta["stock_id"]
            stock_embedding = self.temporal_stock_embeddings(stock_id)
            x[:, :self.front_padding_num + 1, :] = stock_embedding
            # open price start token
            seq_mask[..., :self.front_padding_num + 1] = torch.ones_like(seq_mask[..., :self.front_padding_num + 1])

            relative_days_off[..., :self.front_padding_num] = torch.arange(1, self.front_padding_num + 1).repeat(
                [b, 1])
        return x, seq_mask, relative_days_off

    def forward(self, x, meta, **transformer_kwargs):
        seq_mask = (x[..., 0] > 0.1)
        stock_id = meta["stock_id"][0, 1]
        # print(f"""self.weight[stock_id] {self.weight[stock_id].shape}
        #       self.bias[stock_id] {self.bias[stock_id].shape}
        #       meta[etfs] {meta["etfs"].shape}
        #         """)
        spatial_embeddings = torch.einsum("nli,id->lnd", meta["etfs"][0], self.weight[stock_id]) \
                             + self.bias[stock_id][None, None, :]
        # spatial_embeddings = spatial_embeddings.permute([1, 0, 2]).contiguous()
        # spatial_embeddings = self.spatial_input_projection(meta["etfs"])[0]  # n_etf,l,md
        # print(f"spatial_embeddings {spatial_embeddings.shape}")
        # spatial_embeddings = spatial_embeddings.permute([1, 0, 2]).contiguous()  # l,n,md
        # spatial_stock_embedding = self.spatial_stock_embeddings(meta["stock_id"][:, 1])[0]  # 1,md
        # spatial_stock_embedding = spatial_stock_embedding[None, None, :]  # 1,1,md
        # print(f"spatial_stock_embedding {spatial_stock_embedding.shape}")
        etfs, _ = self.spatial_transformer(
            inputs_embeds=spatial_embeddings,
            position_ids=self.spatial_etf_ids

        )
        etfs = etfs.sum(1)[None, :, :]
        etfs = self.spatial2temporal_projection(etfs)

        x = self.temporal_input_projection(x) * math.sqrt(self.project_dimension)
        x, seq_mask, relative_days_off = self.attach_temporal_head(x, meta, seq_mask)

        regress_embeddings, _ = \
            self.temporal_transformer(inputs_embeds=x + etfs,
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
            seq_mask = (x[..., 0] > 0.1)
            x = self.temporal_input_projection(x) * math.sqrt(self.project_dimension)
            x, seq_mask, relative_days_off = self.attach_temporal_head(x, meta, seq_mask)

            regress_embeddings, cache, attention = \
                self.temporal_transformer(inputs_embeds=x,
                                          attention_mask=seq_mask,
                                          position_ids=relative_days_off,
                                          output_attentions=True,
                                          **transformer_kwargs)

            mu, sigma = self.output_projection(regress_embeddings).split(1, -1)
        return {
            "mu": mu.squeeze(-1),
            "ln_sigma": sigma.squeeze(-1),
            "cache": cache,
            "attention": attention,
            "regress_embeddings": regress_embeddings,
        }

    def reset(self):
        pass

    def training_step(self, batch, bn):
        x, target, meta = batch
        model_output = self(x, meta)
        loss_output = self.loss(model_output, target, group=meta["group"])
        with torch.no_grad():
            metric_output = self.metric_object(
                outputs={
                    "pred": model_output["mu"].exp()
                },
                target=target,
                mask=loss_output["valid_mask"],
            )
        log = {
            "train_loss": loss_output["train_loss"]
            , "valid_loss": loss_output["valid_loss"]
            , "metric_output": metric_output
        }

        ret = {
            'loss': loss_output["train_loss"],
            "valid_loss": loss_output["valid_loss"],
            'metric_output': metric_output,
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
                "metric_output",
                # "diff", 'original'
                ]
        res = [_agg_by_key(key) for key in keys]
        train_loss, valid_loss, metric_output = res
        print(f"tr: {train_loss:.6f}, va: {valid_loss:.6f}, mo:,{metric_output:.6f}")
        # print(f"train loss: {train_loss:.6f}, valid_loss: {valid_loss:.6f}, diff: {diff:.6f} original: {original:.6f}")
        return {"loss": train_loss, "val_loss": valid_loss}

    def configure_optimizers(self):
        # from radam import RAdam
        # optimizer = RAdam(self.parameters())
        from torch.optim import Adam
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer

    def train_dataloader(self) -> DataLoader:
        dataset = MyIterableDataset(
            self.filenames[0]
            , market_fn=self.market_fn
            , stock_mapping=self.stock_mapping
            , etf_mapping=self.etf_mapping
            , front_padding_num=self.front_padding_num
            , train_start_date=self.train_start_date
            , valid_start_date=self.valid_start_date
            , position_oov_size=self.position_oov_size
            , stock_id_header_token=self.stock_id_header_token
            , stock_price_start_token=self.stock_price_start_token
        )
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)


def init_weights(m):
    if type(m) == TimeSeriesTransformer:
        torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(m.bias, -bound, bound)


@click.command()
@click.option('--file-re', type=str,
              default='../data/sample_record_npz/window:10-agg:mean-date:2012-01-01:2017-01-011598383917.111129.npz',
              show_default=True,
              help="stock file regex")
@click.option('--market-fn', type=str,
              default='../data/sample_record_npz/etf-date:2012-01-01:2017-01-011598345700.054786.npz',
              show_default=True,
              help="etf file regex")
@click.option('--batch-size', type=int, default=1, show_default=True, help="batch size")
@click.option('--attention-type', type=str, default='full', show_default=True, help="file regex")
@click.option('--gpus', type=int, default=None, show_default=True, help="gpu num")
@click.option('--accumulate_grad_batches', type=int, default=11, show_default=True, help="update with N batches")
@click.option('--auto_lr_find', type=bool, default=False, show_default=True, help="auto_lr_find")
def train(file_re, batch_size, attention_type, gpus, accumulate_grad_batches, auto_lr_find, market_fn):
    torch.cuda.empty_cache()
    model = TimeSeriesTransformer(stock_input_dimensions=5,
                                  stock_fn=file_re,
                                  market_fn=market_fn,
                                  # project_dimension=128,
                                  temporal_n_layers=2,
                                  temporal_n_heads=10,
                                  temporal_query_dimensions=10,
                                  temporal_value_dimensions=10,
                                  spatial_n_heads=8,
                                  spatial_n_layers=2,
                                  spatial_query_dimensions=8,
                                  spatial_value_dimensions=8,
                                  feed_forward_dimensions=100,
                                  attention_type=attention_type,
                                  num_workers=0,
                                  batch_size=batch_size,
                                  seq_len=1300,
                                  seed=101,
                                  lr=1e-4,
                                  asset_list='../data/asset_list.txt',
                                  etf_list='../data/etf_list.txt',
                                  front_padding_num=2,
                                  train_start_date="2012-01-01",
                                  valid_start_date="2016-01-01",
                                  )
    init_weights(model)
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
            filepath=os.path.join("lightning_logs", '{epoch}-{val_loss:.3f}'),
            monitor='val_loss',
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
