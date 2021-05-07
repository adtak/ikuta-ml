import os
import tweepy
from dataclasses import dataclass
from typing import List


@dataclass()
class Conversation:
    tweet_id: str
    tweet: str
    responce_id: str
    responce: str


class TwitterClawler:
    def __init__(self) -> None:
        api_key = os.environ['API_KEY']
        api_secret = os.environ['API_SECRET']
        access_token = os.environ['ACCESS_TOKEN']
        access_token_secret = os.environ['ACCESS_TOKEN_SECRET']

        auth = tweepy.OAuthHandler(api_key, api_secret)
        auth.set_access_token(access_token, access_token_secret)

        self.api = tweepy.API(
            auth,
            wait_on_rate_limit=True,
            wait_on_rate_limit_notify=True
        )

    def get_conversation(
            self,
            keyword: str,
            count: int = 100
    ) -> List[Conversation]:

        results = self.api.search(
            q=[keyword],
            count=count,
            result_type='recent'
        )
        conversations = []

        for result in results:
            tweet_id = result.in_reply_to_status_id
            if not tweet_id:
                continue

            try:
                tweet = self.api.get_status(tweet_id)
            except tweepy.TweepError as e:
                print(e)
                continue

            conversations.append(
                Conversation(tweet.id, tweet.text, result.id, result.text)
            )

        return conversations
