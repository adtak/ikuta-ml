import datetime as dt
import os
import tweepy
from dataclasses import dataclass
from pandas import DataFrame
from pathlib import Path
from string import Template
from typing import List

import ikuta_ml.util.file_util as fu
from ikuta_ml.util.string_util import cleanse_tweet


@dataclass()
class Conversation:
    tweet_id: str
    tweet: str
    raw_tweet: str
    reply_tweet_id: str
    reply_tweet: str
    raw_reply_tweet: str

    @classmethod
    def from_df(cls, df: DataFrame) -> List['Conversation']:
        return [cls(
            series['tweet_id'],
            series['tweet'],
            series['raw_tweet'],
            series['reply_tweet_id'],
            series['reply_tweet'],
            series['raw_reply_tweet'],
        ) for _, series in df.iterrows()]


class TwitterClawler:
    def __init__(
        self,
        output_dir_path: Path,
        max_records_per_file: int = 10_000,
    ) -> None:

        self.api = self._init_tweepy()

        self.output_dir_path = output_dir_path
        self.output_file_name_template = Template(
            dt.datetime.now().strftime('%Y%m%d_%H%M%S') + '_conversation_${number}.csv'
        )

        self.max_records_per_file = max_records_per_file

    def _init_tweepy(self):
        api_key = os.environ['API_KEY']
        api_secret = os.environ['API_SECRET']
        access_token = os.environ['ACCESS_TOKEN']
        access_token_secret = os.environ['ACCESS_TOKEN_SECRET']

        auth = tweepy.OAuthHandler(api_key, api_secret)
        auth.set_access_token(access_token, access_token_secret)

        return tweepy.API(
            auth,
            retry_count=5,
            wait_on_rate_limit=True,
            wait_on_rate_limit_notify=True
        )

    def get_conversation(
            self,
            keyword: str,
            limit: int = 0,
            **kwargs,
    ) -> List[Conversation]:
        result_type = kwargs.get('result_type', 'recent')
        tweepy_cursor = tweepy.Cursor(
            self.api.search, q=keyword, result_type=result_type, **kwargs
        ).items(limit)

        conversations = []
        cnt = 0

        for result in tweepy_cursor:
            tweet_id = result.in_reply_to_status_id
            if not tweet_id:
                continue

            try:
                tweet = self.api.get_status(tweet_id)
            except tweepy.TweepError as e:
                print(e)
                continue

            conversations.append(
                Conversation(
                    tweet.id,
                    cleanse_tweet(tweet.text),
                    tweet.text,
                    result.id,
                    cleanse_tweet(result.text),
                    result.text,
                )
            )

            if len(conversations) >= self.max_records_per_file:
                fu.write_csv_from_list(
                    conversations,
                    self.output_dir_path,
                    self.output_file_name_template.substitute(number=cnt)
                )
                conversations = []
                cnt += 1

        return conversations
