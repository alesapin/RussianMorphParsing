# -*- coding: utf-8 -*-
from catboost import CatBoostClassifier
from base import convert_words_to_features_vectors, convert_word_to_features_vectors
from base import get_common_parses, measure_quality
from base import ANSWER_REVERSE_MAPPER, pretty_morpheme

import random
import logging


class ForestModel(object):
    def __init__(self, learning_rate, depth, iterations, thread_count, window_size):
        self.window_size = window_size
        self.train_raw_data = None
        self.test_raw_data = None
        self.learning_rate = learning_rate
        self.depth = depth
        self.model = None
        self.iterations = iterations
        self.thread_count = thread_count

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
        self.unique_classes = set([])
        for train_word, answer_word in self.train_raw_data:
            X, Y = convert_words_to_features_vectors(train_word, answer_word, self.window_size)
            self.train_data_x += X
            self.train_data_y += Y
            for letter in Y:
                self.unique_classes.add(letter)

        self.test_data_x = []
        self.test_data_y = []
        for test_word, answer_word in self.test_raw_data:
            X, Y = convert_words_to_features_vectors(test_word, answer_word, self.window_size)
            self.test_data_x += X
            self.test_data_y += Y
            for letter in Y:
                self.unique_classes.add(letter)

        logging.info("Total train letters %s", len(self.train_data_x))
        logging.info("Total test letters %s", len(self.test_data_x))
        logging.info("One feature vector size %s", len(self.train_data_x[0]))
        logging.info("Unique classes are %s", ', '.join(ANSWER_REVERSE_MAPPER[label] for label in self.unique_classes))
        logging.info("Unique classes count %s", len(self.unique_classes))
        self.model = CatBoostClassifier(
            learning_rate=self.learning_rate,
            depth=self.depth,
            loss_function='MultiClass',
            iterations=self.iterations,
            classes_count=len(self.unique_classes),
            thread_count=self.thread_count,
            task_type='GPU',
        )

    def train(self):
        logging.info("Starting train")
        self.model.fit(self.train_data_x, self.train_data_y, cat_features=list(range(len(self.train_data_x[0]))))

    def save(self, path):
        self.model.save_model(path)

    def load(self, path):
        self.model.load_model(path)

    def classify(self, raw_data):
        Xs = []
        words = []
        for raw_word in raw_data:
            X = convert_word_to_features_vectors(raw_word, self.window_size)
            word = raw_word.split('\t')[0]
            Xs += X
            words.append(word)
        prediction_classes = self.model.predict(Xs)
        result = []
        start = 0
        for word in words:
            end = start + len(word)
            answer = [ANSWER_REVERSE_MAPPER[int(num)] for num in prediction_classes[start:end]]
            result.append(word + '\t' + pretty_morpheme(word, answer))
            start = end
        return result

    def test(self, print_errors):
        prediction_classes = self.model.predict(self.test_data_x)
        test_words = [w[0].split('\t')[0] for w in self.test_raw_data]

        start = 0
        predictions = []
        real_answers = []
        for word in test_words:
            end = start + len(word)
            answer = [ANSWER_REVERSE_MAPPER[int(num)] for num in prediction_classes[start:end]]
            correct = [ANSWER_REVERSE_MAPPER[int(num)] for num in self.test_data_y[start:end]]
            predictions.append(answer)
            real_answers.append(correct)

            if print_errors and answer != correct:
                print('expect:', pretty_morpheme(word, answer), 'get:', pretty_morpheme(word, correct))
            start = end

        return measure_quality(predictions, real_answers, test_words)
