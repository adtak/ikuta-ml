import numpy as np
from pyknp import Juman
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import Dict, List


def split_text_morphemes(text: str) -> List[str]:
    jumanpp = Juman()
    return [m.midasi for m in jumanpp.analysis(text)]


def split_texts_morphemes(texts: List[str]) -> List[List[str]]:
    return [split_text_morphemes(t) for t in texts]


def convert_to_index(
    splitted_text: List[str],
    w2i_dict: Dict[str, int],
) -> List[int]:
    return [w2i_dict.get(word, w2i_dict['UNK']) for word in splitted_text]


def pad_post_zero(
    input: List[List[str]],
    maxlen: int = 140,
) -> np.ndarray:
    """
    >>> sequence = [[1], [2, 3], [4, 5, 6]]
    >>> pad_post_zero(sequence)
    array([[1, 0, 0],
            [2, 3, 0],
            [4, 5, 6]], dtype=int32)
    """
    return pad_sequences(
        input, maxlen=maxlen, padding='post', truncating='post')
