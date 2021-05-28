from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM
from numpy import array


def try_simple():
    input = Input(shape=(3, 1))
    lstm = LSTM(1)(input)

    model = Model(input, lstm)

    data = array([0.1, 0.2, 0.3]).reshape((1, 3, 1))
    result = model.predict(data)

    # 最終的な出力のみ
    # [[0.12181703]]
    # (1, 1)
    print(result)
    print(result.shape)


def try_return_sequences():
    input = Input(shape=(3, 1))
    lstm = LSTM(1, return_sequences=True)(input)

    model = Model(input, lstm)

    data = array([0.1, 0.2, 0.3]).reshape((1, 3, 1))

    # 全てのtimestepでhを出力
    # [[[0.00286665]
    #   [0.00810191]
    #   [0.01525954]]]
    # (1, 3, 1)
    result = model.predict(data)
    print(result)
    print(result.shape)


def try_return_state():
    input = Input(shape=(3, 1))
    lstm, state_h, state_c = LSTM(1, return_state=True)(input)

    model = Model(input, [lstm, state_h, state_c])

    data = array([0.1, 0.2, 0.3]).reshape((1, 3, 1))

    # 最終的な出力と、最終的なhとcが出力。当然、最終的な出力=最終的なh
    # [
    #   array([[-0.06940044]], dtype=float32),
    #   array([[-0.06940044]], dtype=float32),
    #   array([[-0.12146951]], dtype=float32)
    # ]
    # (1, 1)
    # (1, 1)
    # (1, 1)
    result = model.predict(data)
    print(result)
    print(result[0].shape)
    print(result[1].shape)
    print(result[2].shape)


def try_return_sequences_n_state():
    input = Input(shape=(3, 1))
    lstm, state_h, state_c = LSTM(
        1, return_sequences=True, return_state=True)(input)

    model = Model(input, [lstm, state_h, state_c])

    data = array([0.1, 0.2, 0.3]).reshape((1, 3, 1))

    # 全てのtimestepにおけるhと、最終的なhとcが出力
    # [
    #   array([[[0.01437917],
    #         [0.04396546],
    #         [0.08838214]]], dtype=float32),
    #   array([[0.08838214]], dtype=float32),
    #   array([[0.15572777]], dtype=float32)
    # ]
    # (1, 3, 1)
    # (1, 1)
    # (1, 1)
    result = model.predict(data)
    print(result)
    print(result[0].shape)
    print(result[1].shape)
    print(result[2].shape)


if __name__ == '__main__':
    # try_simple()
    # try_return_sequences()
    # try_return_state()
    try_return_sequences_n_state()
