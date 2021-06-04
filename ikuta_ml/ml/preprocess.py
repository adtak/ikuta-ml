import numpy as np
from dataclasses import dataclass
from tensorflow.keras.utils import to_categorical
from typing import Dict, List, Tuple

import ikuta_ml.util.preprocess_util as pu
from ikuta_ml.crawler.twitter_crawler import Conversation


@dataclass
class PreprocessResult:
    tweet_converted_idx: List[List[int]]
    replay_converted_idx: List[List[int]]
    w2i_dict: Dict[str, int]
    i2w_dict: Dict[int, str]

    def create_train_data(
        self,
        maxlen: int,
        sos_index: int,
        eos_index: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        encoder_inputs = pu.pad_post_zero(
            input=list(map(lambda x: [sos_index] + x, self.tweet_converted_idx)),
            maxlen=maxlen
        )
        decoder_inputs = pu.pad_post_zero(
            input=list(map(lambda x: [sos_index] + x, self.replay_converted_idx)),
            maxlen=maxlen
        )
        # maxlenを超えている場合、paddingする際に切り捨てるが、EOSが切り捨てられると困るのであらかじめ対応
        decoder_labels = pu.pad_post_zero(
            input=list(map(lambda x: x[:maxlen-1] + [eos_index], self.replay_converted_idx)),
            maxlen=maxlen
        )

        return (
            encoder_inputs,
            decoder_inputs,
            # indexをone-hotに変換
            to_categorical(decoder_labels, max(self.i2w_dict)+1)
        )


class Preprocesser:
    def __init__(self, raw_data: List[Conversation]):
        self.raw_data = raw_data
        self.threshold_unk = 5
        self.sos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3

    def run(self) -> PreprocessResult:
        tweet_list = [c.tweet for c in self.raw_data]
        replay_list = [c.reply_tweet for c in self.raw_data]

        splitted_tweet_list = pu.split_texts_morphemes(tweet_list)
        splitted_replay_list = pu.split_texts_morphemes(replay_list)

        all_words = \
            self._flatten(splitted_tweet_list) \
            + self._flatten(splitted_replay_list)

        w2i_dict, i2w_dict = self._create_dict(all_words)

        tweet_converted_idx = \
            self._convert_texts_to_index(splitted_tweet_list, w2i_dict)
        replay_converted_idx = \
            self._convert_texts_to_index(splitted_replay_list, w2i_dict)

        return PreprocessResult(
            tweet_converted_idx,
            replay_converted_idx,
            w2i_dict,
            i2w_dict
        )

    def _flatten(self, text_list: List[List[str]]) -> List[str]:
        result: List[str] = []
        for splitted_text in text_list:
            result.extend(splitted_text)
        return result

    def _create_dict(
        self,
        word_list: List[str],
    ) -> Tuple[Dict[str, int], Dict[int, str]]:

        def count_arise(word_list: List[str]) -> Dict[str, int]:
            # 単語毎に発生回数を記録するDict
            counter: Dict[str, int] = dict()
            for word in word_list:
                count = counter.get(word, 0)
                counter[word] = count + 1
            return counter

        counter = count_arise(word_list)
        word_set = set(word_list)

        unk_words = {
            word for word, count in counter.items()
            if count < self.threshold_unk
        }
        word_set = {w for w in word_set if w not in unk_words}

        print(f'Num of unk_words: {len(unk_words)}')
        print(f'Num of word_set: {len(word_set)}')

        # 単語->インデックスの辞書
        # 0はpadding, 1, 2, 3はSOS, EOS, UNKで使用するのでindexは4から開始する
        w2i_dict = {w: i+4 for i, w in enumerate(word_set)}
        w2i_dict['SOS'], w2i_dict['EOS'], w2i_dict['UNK'] = \
            self.sos_idx, self.eos_idx, self.unk_idx
        # インデックス->単語の辞書
        i2w_dict = {w2i_dict[w]: w for w in w2i_dict}

        return w2i_dict, i2w_dict

    def _convert_texts_to_index(
        self,
        texts: List[List[str]],
        w2i_dict: Dict[str, int],
    ) -> List[List[int]]:
        return [pu.convert_to_index(t, w2i_dict) for t in texts]
