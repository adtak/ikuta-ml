import argparse

from ikuta_ml.crawler.twitter_crawler import TwitterClawler


def main():
    args = _parse_args()

    crawler = TwitterClawler()
    results = crawler.get_conversation(args.keyword)
    for r in results:
        print(r)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--keyword', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    main()
