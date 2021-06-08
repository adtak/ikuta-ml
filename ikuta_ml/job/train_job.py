import argparse
import datetime as dt
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict

from ikuta_ml.ml.preprocess import PreprocessResult
from ikuta_ml.ml.seq2seq import Seq2Seq, Seq2SeqSetting


def main():
    args = _parse_args()
    input_dir_path = Path(args.input_dir)
    output_dir_path = Path(args.output_dir) / args.label
    os.makedirs(output_dir_path, exist_ok=False)

    preprocess_result = PreprocessResult.load(input_dir_path, 'preprocess_result.pickle')

    encoder_x, decoder_x, decoder_y = preprocess_result.create_train_data(
        140,
        Seq2Seq.SOS_TOKEN,
        Seq2Seq.EOS_TOKEN
    )

    setting = Seq2SeqSetting(
        timestep=140,
        lstm_units=32,
        embedding_input=max(preprocess_result.i2w_dict)+1,  # max index + 1
        embedding_output=64,
        dence_units=max(preprocess_result.i2w_dict)+1,
    )
    seq2seq = Seq2Seq.for_train(setting)

    hist = seq2seq.fit(encoder_x[:5], decoder_x[:5], decoder_y[:5], 1, 10)
    print(hist)

    _save_loss_plot(hist, output_dir_path)

    seq2seq.save(output_dir_path)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', default='datasets')
    parser.add_argument('-o', '--output_dir', default='train_results')
    parser.add_argument('-l', '--label', default=dt.datetime.now().strftime('%Y%m%d_%H%M%S'))

    return parser.parse_args()


def _save_loss_plot(hist: Dict, output_dir_path: Path):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(15, 5))

    pd.DataFrame(hist)[['loss']].plot(ax=ax)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.legend()

    fig.savefig(output_dir_path / "loss.jpg")


if __name__ == '__main__':
    main()
