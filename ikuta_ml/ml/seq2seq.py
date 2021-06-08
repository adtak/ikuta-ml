import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Final

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Embedding, Input, LSTM


@dataclass
class Seq2SeqSetting:
    timestep: int
    lstm_units: int
    embedding_input: int
    embedding_output: int
    dence_units: int


class Seq2Seq:
    MASKING_TOKEN: Final[int] = 0
    SOS_TOKEN: Final[int] = 1
    EOS_TOKEN: Final[int] = 2
    UNK_TOKEN: Final[int] = 3

    def __init__(
        self,
    ) -> None:
        pass

    @classmethod
    def for_train(cls, settings: Seq2SeqSetting):
        seq2seq = cls()
        seq2seq.settings = settings

        seq2seq._init_layer()
        encoder_in, encoder_states = seq2seq._create_train_model()
        seq2seq._create_predict_model(encoder_in, encoder_states)

        return seq2seq

    @classmethod
    def for_predict(cls, output_dir_path: Path):
        seq2seq = cls()
        seq2seq.pred_encoder_model = load_model(output_dir_path / 'pred_encoder_model')
        seq2seq.pred_decoder_model = load_model(output_dir_path / 'pred_encoder_model')
        return seq2seq

    def _init_layer(self):
        self.embedding_layer = Embedding(
            self.settings.embedding_input,
            self.settings.embedding_output,
            mask_zero=True,
            name='shared_embedding'
        )
        self.encoder_lstm = LSTM(
            self.settings.lstm_units,
            return_state=True,
            name='encoder_lstm'
        )
        self.decoder_lstm = LSTM(
            self.settings.lstm_units,
            return_state=True,
            return_sequences=True,
            name='decoder_lstm'
        )
        self.decoder_dense = Dense(
            self.settings.dence_units,
            activation='softmax',
            name='decoder_dense'
        )

    def _create_train_model(self):
        encoder_in = Input(shape=(self.settings.timestep,), name='encoder_input')
        x = self.embedding_layer(encoder_in)
        _, encoder_state_h, encoder_state_c = self.encoder_lstm(x)
        encoder_states = [encoder_state_h, encoder_state_c]

        decoder_in = Input(shape=(self.settings.timestep,), name='decoder_input')
        x = self.embedding_layer(decoder_in)
        x, _, _ = self.decoder_lstm(x, initial_state=encoder_states)
        decoder_out = self.decoder_dense(x)

        self.train_model = Model([encoder_in, decoder_in], decoder_out)
        self.train_model.compile(loss="mean_squared_error", optimizer="sgd")

        return encoder_in, encoder_states

    def _create_predict_model(self, encoder_in, encoder_states):
        self.pred_encoder_model = Model(encoder_in, encoder_states)

        decoder_state_h_in = Input(shape=(self.settings.lstm_units,))
        decoder_state_c_in = Input(shape=(self.settings.lstm_units,))
        decoder_states_in = [decoder_state_h_in, decoder_state_c_in]

        decoder_in = Input(shape=(1,))
        x = self.embedding_layer(decoder_in)
        x, decoder_state_h, decoder_state_c = self.decoder_lstm(x, initial_state=decoder_states_in)
        decoder_out = self.decoder_dense(x)

        self.pred_decoder_model = Model(
            [decoder_in] + decoder_states_in,
            [decoder_out, decoder_state_h, decoder_state_c]
        )

    def fit(
        self,
        encoder_x: np.ndarray,  # encoder_x.shape => (batches, timestep)
        decoder_x: np.ndarray,  # decoder_x.shape => (batches, timestep)
        decoder_y: np.ndarray,  # decoder_y.shape => (batches, timestep, index)
        batch_size: int,
        epochs: int,
    ) -> Dict:  # {'loss': [2.5682957129902206e-05]}
        history = self.train_model.fit(
            [encoder_x, decoder_x],
            decoder_y,
            batch_size=batch_size,
            epochs=epochs,
        )
        return history.history  # epoch毎のloss

    def predict(self, input, i2w_dict: Dict[int, str]) -> List[str]:
        result = []
        state_value = self.pred_encoder_model.predict(input)
        input = [np.array([[Seq2Seq.SOS_TOKEN]])]

        for _ in range(self.settings.timestep):
            y, h, c = self.pred_decoder_model.predict(input + state_value)

            if y[0][0] == Seq2Seq.EOS_TOKEN:
                break

            max_index = np.argmax(y[0][0])
            result.append(i2w_dict[max_index])

            state_value = [h, c]
            input = [np.array([[max_index]])]

        return result

    def save(self, output_dir_path: Path):
        self.pred_encoder_model.save(
            str(output_dir_path / 'pred_encoder_model')
        )
        self.pred_decoder_model.save(
            str(output_dir_path / 'pred_decoder_model')
        )
