import argparse
from pathlib import Path

from ikuta_ml.crawler.twitter_crawler import Conversation
from ikuta_ml.ml.preprocess import Preprocesser
from ikuta_ml.util.file_util import read_csv


def main():
    args = _parse_args()

    df = read_csv(Path(args.input_dir), '20210604_085437_conversation_0.csv')
    converted_data = Conversation.from_df(df.dropna())

    preprocesser = Preprocesser(converted_data)
    preprocess_result = preprocesser.run()

    preprocess_result.save(Path(args.output_dir), 'preprocess_result.pickle')


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', required=True)
    parser.add_argument('-o', '--output_dir', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    main()
