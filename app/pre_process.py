import csv
import datetime
import glob
import os
from collections import Counter

import click
import numpy as np
import tqdm
import traceback
import pandas as pd
from common_utils import CommonUtils

_ROOT = os.path.abspath(os.path.dirname(__file__))
TF_RECORDS = _ROOT + "/../data/stocks_seq"


def get_stock_name(x):
    return x.split("/")[-1][:-4]


def future(z, window, agg):
    z = z.shift(-window)
    return agg(z.rolling(window))


def create_tf_records(text_files, min_seq_len, max_seq_len, per_file_limit=50000
                      , train_date='2008-01-01'
                      , valid_date='2014-01-01'
                      , output_fn=None
                      , window=20
                      ):
    # if not os.path.exists(TF_RECORDS):
    #     os.makedirs(TF_RECORDS)
    output_fn = output_fn + str(datetime.datetime.now().timestamp()) + ".npz"

    doc_counts = 0
    r_input = []
    r_target = []
    r_stock_name = []
    r_days_offset = []
    for filename in tqdm.tqdm(text_files):
        try:
            df = pd.read_csv(filename)
            df["days_offset"] = df["Date"].apply(CommonUtils.date_to_idx)
            stock_name = get_stock_name(filename)
            feature = [
                # 'Date',
                'Open',
                'High',
                'Low',
                'Close',
                'Volume',
                # 'OpenInt'
            ]
            df = df[(df.Date >= train_date) & (df.Date < valid_date)]
            x = df[feature].values
            if max_seq_len > x.shape[0] > min_seq_len:
                inputs = np.concatenate([np.zeros([1, len(feature)]), x], axis=0).astype(np.float32)
                max_within_window = future(df.Close, window, agg=lambda x: x.max())
                # min_within_window = future(df.Close, window, agg=lambda x: x.min())

                max_within_window.fillna(0, inplace=True)
                # min_within_window.fillna(0, inplace=True)
                targets = np.concatenate([max_within_window, np.zeros(1), ], axis=0).astype(np.float32).tolist()
                days_offset = np.concatenate([np.zeros(1), df.days_offset], axis=0).astype(np.float32)

                # TODO padding front
                def pad(x, sd=[]):
                    return np.pad(x, [(0, max_seq_len - df.shape[0] - 1)] + sd, constant_values=0.0)

                inputs = pad(inputs, sd=[(0, 0)])
                targets = pad(targets)
                days_offset = pad(days_offset).astype(np.int32)

                r_days_offset.append(days_offset)
                r_input.append(inputs)
                r_target.append(targets)
                r_stock_name.append(stock_name)
        except:
            traceback.print_exc()
            pass

    if len(r_input) == 0:
        raise ValueError("no data input")

    data = {
        "input": np.stack(r_input, 0),
        "target": np.stack(r_target, 0),
        "stock_name": np.stack(r_stock_name, 0),
        "days_offset": np.stack(r_days_offset, 0),
    }
    print([f"{k}: {v.shape}" for (k, v) in data.items()])

    np.savez(output_fn, **data)


@click.command()
@click.option('--data-dir', type=str, default="../data/Stocks/a*.txt", show_default=True, help="training data path")
@click.option('--output-fn', type=str,
              default="/Users/hehehe/PycharmProjects/fast-transformers/data/sample_record_npz/", show_default=True,
              help="training data path")
@click.option('--min-seq-len', type=int, default=800, show_default=True, help="minimum sequence length")
@click.option('--max-seq-len', type=int, default=1500, show_default=True, help="minimum sequence length")
@click.option('--train-date', type=str, default='2009-01-01', show_default=True, help="example start")
@click.option('--valid-date', type=str, default='2014-01-01', show_default=True, help="example end")
def train(data_dir, min_seq_len, max_seq_len, train_date, valid_date, output_fn):
    text_files = glob.glob(data_dir)
    create_tf_records(text_files, min_seq_len, max_seq_len, train_date=train_date, valid_date=valid_date,
                      output_fn=output_fn)
    print("Pre-processing is done............")


if __name__ == "__main__":
    train()
