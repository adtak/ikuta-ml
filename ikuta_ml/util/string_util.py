import emoji
import re
import sys


def cleanse_tweet(tweet: str) -> str:
    # 改行の除去
    tweet = tweet.replace('\n', '')
    # URLリンクの除去
    tweet = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+', '', tweet)
    # emojiを句点に変換
    tweet = tweet.translate(
        dict.fromkeys(range(0x10000, sys.maxunicode + 1), '。')
    )
    tweet = emoji.get_emoji_regexp().sub('。', tweet)
    # ハッシュタグ除去
    if tweet.find('#') != -1:
        tweet = tweet[0:tweet.find('#')]
    # @除去
    tweet = re.sub('@.*? ', '', tweet)
    # 連続した句点をまとめる
    tweet = re.sub('。+', '。', tweet)
    # 句点+？or！は？or！に変換
    tweet = re.sub('。?([？！])。?', r'\1', tweet)
    # 半角スペースを全角スペースに変換
    tweet = tweet.replace(' ', '　')

    return tweet
