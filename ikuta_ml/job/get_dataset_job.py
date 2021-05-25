import argparse
import datetime as dt
from pathlib import Path

import ikuta_ml.util.file_util as fu
from ikuta_ml.crawler.twitter_crawler import TwitterClawler


def main():
    args = _parse_args()

    output_file_name = \
        dt.datetime.now().strftime('%Y%m%d_%H%M%S') + '_conversation.csv'

    crawler = TwitterClawler()
    results = crawler.get_conversation(args.keyword, args.limit)

    fu.write_csv_from_list(results, Path(args.output_dir), output_file_name)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--keyword', required=True)
    parser.add_argument('-o', '--output_dir', required=True)
    parser.add_argument('-l', '--limit', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    main()
