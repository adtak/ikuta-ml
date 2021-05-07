import emoji
import re
import sys


def cleanse_tweet(tweet: str):
    tweet = tweet.replace('\n', '')
    tweet = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+', '', tweet)
    tweet = tweet.translate(
        dict.fromkeys(range(0x10000, sys.maxunicode + 1), '。')
    )
    tweet = emoji.get_emoji_regexp().sub(u'', tweet)
    if tweet.find('#') != -1:
        tweet = tweet[0:tweet.find('#')]
