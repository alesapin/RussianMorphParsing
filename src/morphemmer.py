#-*- coding: utf-8 -*-
import base
import time
import logging
from neural_morph_segm import Partitioner
from forest_morph_segment import CatboostSpliter

class Morphemmer(object):

    @staticmethod
    def get_name():
        raise Exception("Method get_name is unimplemented")

    def train(self, train_words):
        if any(word.unlabeled for word in train_words):
            raise Exception("Train set has unlabeled words")

        logging.info("Will train on %s words", len(train_words))

        start = time.time()
        self._train_impl(train_words)
        end = time.time()
        return end - start

    def predict(self, sample_words):

        logging.info("Will predict %s words", len(sample_words))
        start = time.time()
        result = self._predict_impl(sample_words)
        end = time.time()

        if any(word.unlabeled for word in sample_words):
            raise Exception("Output predict set has unlabeled words")
        return result, end - start

    def _train_impl(self, train_words):
        raise Exception("Method _train_impl is unimplemented")

    def _predict_impl(self, sample_words):
        raise Exception("Method _predict_impl is unimplemented")

    def evaluate(self, words):
        if any(word.unlabeled for word in words):
            raise Exception("Evaluation set has unlabeled words")

        logging.info("Will evaluate %s words", len(words))

        output_words = self._predict_impl(words)

        return self._evaluate(output_words, words)

    def _evaluate(self, output_words, words):
        return {
            #"morfessor_evaluate": base.morfessor_evaluate(output_words, words),
            "quality": base.measure_quality(output_words, [w.get_labels() for w in words])
        }

class NeuralMorphemmer(Morphemmer):
    def __init__(self, params):
        self.partitioner = Partitioner(**params)

    def _prepare_input_data(self, words):
        source = []
        targets = []
        for word in words:
            source.append(word.get_word())
            targets.append(word.get_labels())
        return source, targets

    @staticmethod
    def get_name():
        return "neural"

    def _train_impl(self, train_words):
        inputs, targets = self._prepare_input_data(train_words)
        self.partitioner.train(inputs, targets)

    def _predict_impl(self, sample_words):
        inputs, _ = self._prepare_input_data(sample_words)
        predicted_targets = self.partitioner._predict_probs(inputs)
        return [elem[0] for elem in predicted_targets]

class ForestMorphemmer(Morphemmer):
    def __init__(self, params):
        self.classifier = CatboostSpliter(params['xmorphy_path'])

    @staticmethod
    def get_name():
        return "catboost"

    def _train_impl(self, train_words):
        self.classifier.train(train_words)

    def _predict_impl(self, sample_words):
        return self.classifier.classify(sample_words)
