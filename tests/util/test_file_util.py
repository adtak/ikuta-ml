from pathlib import Path

import ikuta_ml.util.file_util as file_util
from ikuta_ml.crawler.twitter_crawler import Conversation


def test_write_csv_from_list(output_dir):
    test_data = [
        Conversation(1, 'text1_1', 'text1_2', 2, 'text2_1', 'text2_2'),
    ]
    expected = [
        '"tweet_id","tweet","raw_tweet","reply_tweet_id","reply_tweet","raw_reply_tweet"\n',  # noqa E501
        '"1","text1_1","text1_2","2","text2_1","text2_2"\n'
    ]

    file_path = file_util.write_csv_from_list(
        test_data,
        Path(output_dir),
        'test_1.csv',
    )

    if file_path:
        with open(file_path) as f:
            for a, e in zip(f, expected):
                assert a == e
    else:
        assert False


def test_write_csv_from_list_add(output_dir):
    test_data_first = [
        Conversation(1, 'text1_1', 'text1_2', 2, 'text2_1', 'text2_2'),
    ]
    test_data_add = [
        Conversation(3, 'text3_1', 'text3_2', 4, 'text4_1', 'text4_2'),
    ]
    expected = [
        '"tweet_id","tweet","raw_tweet","reply_tweet_id","reply_tweet","raw_reply_tweet"\n',  # noqa E501
        '"1","text1_1","text1_2","2","text2_1","text2_2"\n',
        '"3","text3_1","text3_2","4","text4_1","text4_2"\n'
    ]

    _ = file_util.write_csv_from_list(
        test_data_first,
        Path(output_dir),
        'test_2.csv',
    )

    file_path = file_util.write_csv_from_list(
        test_data_add,
        Path(output_dir),
        'test_2.csv',
        mode='a'
    )

    if file_path:
        with open(file_path) as f:
            for a, e in zip(f, expected):
                assert a == e
    else:
        assert False
