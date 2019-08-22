# -*- coding: utf-8 -*-
import numpy as np
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Masking
from tensorflow.keras.layers import Bidirectional, GRU, CuDNNLSTM
from tensorflow.keras.layers import TimeDistributed, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

from base import convert_words_to_neural_vectors
from base import get_common_parses, measure_quality
from base import ANSWER_REVERSE_MAPPER, pretty_morpheme

import random
import logging

mask_value = 0.0


class NeuralModel(object):

    def __init__(self, num_of_lstms, dropouts, units):
        self.train_raw_data = None
        self.test_raw_data = None
        self.num_of_lstms = num_of_lstms
        self.dropouts = dropouts
        self.units = units
        self.models = []

    def prepare(self, from_dict_path, to_dict_path, split_part=0.2):
        with open(from_dict_path, 'r') as fd, open(to_dict_path, 'r') as td:
            self.common_words = get_common_parses(fd, td)
        logging.info("Total common words %s", len(self.common_words))
        random.shuffle(self.common_words)
        train_part_len = int(len(self.common_words) * (1 - split_part))
        self.train_raw_data = self.common_words[:train_part_len]
        self.test_raw_data = self.common_words[train_part_len:]
        logging.info("Train raw part len %s", len(self.train_raw_data))
        logging.info("Test raw part len %s", len(self.test_raw_data))

        self.train_data_x = []
        self.train_data_y = []
        self.maxlen = 0
        self.inner_maxlen = 0

        for train_word, answer_word in self.train_raw_data:
            X, Y = convert_words_to_neural_vectors(train_word, answer_word)
            for elem in X:
                self.inner_maxlen = max(len(elem), self.inner_maxlen)
            self.train_data_x.append(np.array(X, dtype=np.int8))
            self.train_data_y.append(np.array(Y, dtype=np.int8))
            if len(Y) > self.maxlen:
                self.maxlen = len(Y)

        self.test_data_x = []
        self.test_data_y = []
        for test_word, answer_word in self.test_raw_data:
            X, Y = convert_words_to_neural_vectors(test_word, answer_word)
            for elem in X:
                self.inner_maxlen = max(len(elem), self.inner_maxlen)
            self.test_data_x.append(np.array(X, dtype=np.int8))
            self.test_data_y.append(np.array(Y, dtype=np.int8))

            if len(Y) > self.maxlen:
                self.maxlen = len(Y)
        logging.info("Total train letters %s", len(self.train_data_x))
        logging.info("Total test letters %s", len(self.test_data_x))
        logging.info("One feature vector size %s", len(self.train_data_x[0]))
        logging.info("Max sequence length %s", self.maxlen)
        logging.info("Inner maxlen %s", self.inner_maxlen)

        self.train_data_x = pad_sequences(self.train_data_x, dtype=np.int8, padding='post', value=mask_value, maxlen=self.maxlen)
        self.train_data_y = pad_sequences(self.train_data_y, dtype=np.int8, padding='post', value=mask_value, maxlen=self.maxlen)
        self.test_data_x = pad_sequences(self.test_data_x, dtype=np.int8, padding='post', value=mask_value, maxlen=self.maxlen)
        self.test_data_y = pad_sequences(self.test_data_y, dtype=np.int8, padding='post', value=mask_value, maxlen=self.maxlen)
        self._build_model()

    def _build_model(self):
        for i in range(self.num_of_lstms):
            input_vec = Input(shape=(self.maxlen, self.inner_maxlen))
            bidi1 = Bidirectional(CuDNNLSTM(self.units[i], return_sequences=True))(input_vec)
            do1 = Dropout(self.dropouts[i])(bidi1)
            bidi2 = Bidirectional(CuDNNLSTM(self.units[i], return_sequences=True))(do1)
            do2 = Dropout(self.dropouts[i])(bidi2)
            output = TimeDistributed(Dense(len(ANSWER_REVERSE_MAPPER) + 1, activation='softmax'))(do2)
            self.models.append(Model(inputs=[input_vec], outputs=[output]))
            self.models[-1].compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    def train(self):
        es = EarlyStopping(monitor='val_acc', patience=10, verbose=1)
        for model in self.models:
            logging.info("Starting train")
            model.fit(self.train_data_x, self.train_data_y, epochs=45, verbose=2, callbacks=[es])
            logging.info("Finished")

    def save(self, paths):
        for path, model in zip(paths, self.models):
            model.save(path)

    def load(self, paths):
        for path in paths:
            self.models.append(tensorflow.keras.models.load_model(path))

    def test(self, print_errors):

        prediction_classes = np.mean([model.predict(self.test_data_x) for model in self.models], axis=0).argmax(axis=-1)
        test_words = [w[0].split('\t')[0] for w in self.test_raw_data]

        predictions = []
        real_answers = []
        for i, word in enumerate(test_words):
            cutted_prediction = prediction_classes[i][:len(word)]
            cutted_true = self.test_data_y[i][:len(word)].argmax(axis=-1)
            print(cutted_prediction)
            print(cutted_true)
            answer = [ANSWER_REVERSE_MAPPER[int(num)] for num in cutted_prediction]
            correct = [ANSWER_REVERSE_MAPPER[int(num)] for num in cutted_true]
            predictions.append(answer)
            real_answers.append(correct)

            if print_errors and answer != correct:
                print('expect:', pretty_morpheme(word, answer), 'get:', pretty_morpheme(word, correct))

        return measure_quality(predictions, real_answers, test_words)

    def classify(self, raw_data):
        Xs = []
        words = []
        for raw_word in raw_data:
            X = convert_word_to_neural_vectors(raw_word, self.window_size)
            word = raw_word.split('\t')[0]
            Xs += X
            words.append(word)
        prediction_classes = np.mean([model.predict(Xs) for model in self.models], axis=0).argmax(axis=-1)
        result = []
        start = 0
        for i, word in enumerate(words):
            cutted_prediction = prediction_classes[i][:len(word)]
            answer = [ANSWER_REVERSE_MAPPER[int(num)] for num in cutted_prediction]
            result.append(word + '\t' + pretty_morpheme(word, answer))
        return result
