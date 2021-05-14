import emoji
import re
import sys


def cleanse_tweet(tweet: str) -> str:
    tweet = tweet.replace('\n', '。')
    tweet = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+', '', tweet)
    tweet = tweet.translate(
        dict.fromkeys(range(0x10000, sys.maxunicode + 1), '。')
    )
    tweet = emoji.get_emoji_regexp().sub('。', tweet)
    if tweet.find('#') != -1:
        tweet = tweet[0:tweet.find('#')]

    tweet += '。'
    tweet = re.sub('。+', '。', tweet)

    return tweet
