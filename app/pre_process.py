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

_ROOT = os.path.abspath(os.path.dirname(__file__))
TF_RECORDS = _ROOT + "/../data/stocks_seq"


def get_stock_name(x):
    return x.split("/")[-1][:-4]


def create_tf_records(text_files, min_seq_len, max_seq_len, per_file_limit=50000
                      , train_date='2008-01-01'
                      , valid_date='2014-01-01'
                      ,output_fn=None
                      ):
    # if not os.path.exists(TF_RECORDS):
    #     os.makedirs(TF_RECORDS)
    output_fn = output_fn + str(datetime.datetime.now().timestamp()) + ".npz"

    doc_counts = 0
    r_input = []
    r_target = []
    r_stock_name = []

    for filename in tqdm.tqdm(text_files):
        try:
            df = pd.read_csv(filename)
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
                inputs = np.concatenate([x, np.zeros([1, len(feature)])], axis=0).astype(np.float32)
                targets = np.concatenate([np.zeros(1), x[:, 3]], axis=0).astype(np.float32).tolist()

                # TODO padding front
                def pad(x, sd=[]):
                    return np.pad(x, [(0, max_seq_len - df.shape[0] - 1)] + sd, constant_values=0.0)

                inputs = pad(inputs, sd=[(0, 0)])
                targets = pad(targets)

                r_input.append(inputs)
                r_target.append(targets)
                r_stock_name.append(stock_name)
        except:
            traceback.print_exc()

    if len(r_input) == 0:
        raise ValueError("no data input")

    data = {
        "input": np.stack(r_input, 0),
        "target": np.stack(r_target, 0),
        "stock_name": np.stack(r_stock_name, 0)
    }

    np.savez(output_fn, **data)


@click.command()
@click.option('--data-dir', type=str, default="../data/Stocks/a*.txt", show_default=True, help="training data path")
@click.option('--output-fn', type=str, default="/Users/hehehe/PycharmProjects/fast-transformers/data/sample_record_npz/", show_default=True, help="training data path")
@click.option('--min-seq-len', type=int, default=800, show_default=True, help="minimum sequence length")
@click.option('--max-seq-len', type=int, default=1500, show_default=True, help="minimum sequence length")
@click.option('--train-date', type=str, default='2009-01-01', show_default=True, help="example start")
@click.option('--valid-date', type=str, default='2014-01-01', show_default=True, help="example end")
def train(data_dir, min_seq_len, max_seq_len, train_date, valid_date,output_fn):
    text_files = glob.glob(data_dir)
    create_tf_records(text_files, min_seq_len, max_seq_len, train_date=train_date, valid_date=valid_date,output_fn=output_fn)
    print("Pre-processing is done............")


if __name__ == "__main__":
    train()
