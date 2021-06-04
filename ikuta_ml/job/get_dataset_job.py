import argparse
from pathlib import Path

from ikuta_ml.crawler.twitter_crawler import TwitterClawler


def main():
    args = _parse_args()

    crawler = TwitterClawler(Path(args.output_dir))
    crawler.get_conversation(args.keyword, args.since_id, args.limit)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--keyword', required=True)
    parser.add_argument('-o', '--output_dir', required=True)
    parser.add_argument('-s', '--since_id')
    parser.add_argument('-l', '--limit', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    main()
