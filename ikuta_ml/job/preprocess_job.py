import argparse
from pathlib import Path

from ikuta_ml.crawler.twitter_crawler import Conversation
from ikuta_ml.ml.preprocess import Preprocesser
from ikuta_ml.util.file_util import read_csv


def main():
    args = _parse_args()

    df = read_csv(Path(args.input_dir), args.input_file_name)
    # TODO: @はpyknpでindexエラーになるので対象外。対象外にする処理を他に移動したい。
    df = df.dropna()
    mask = df['tweet'].str.contains('@') | df['reply_tweet'].str.contains('@')
    df = df[~mask]
    mask = (df['tweet'].str.len() > 1) & (df['reply_tweet'].str.len() > 1)
    df = df[mask]
    mask = (df['tweet'].str.len() < 70) & (df['reply_tweet'].str.len() < 70)
    df = df[mask]

    converted_data = Conversation.from_df(df)

    preprocesser = Preprocesser(converted_data)
    preprocess_result = preprocesser.run()

    preprocess_result.save(Path(args.output_dir), args.output_file_name)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='datasets')
    parser.add_argument('-i', '--input_file_name', required=True)
    parser.add_argument('--output_dir', default='datasets')
    parser.add_argument('-o', '--output_file_name', default='preprocess_result.pickle')
    return parser.parse_args()


if __name__ == '__main__':
    main()
