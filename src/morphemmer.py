#-*- coding: utf-8 -*-
import base
import time
import logging
from neural_morph_segm import Partitioner

class Morphemmer(object):

    def get_name(self):
        raise Exception("Method get_name is unimplemented")

    def train(self, train_words):
        if any(word.unlabeled for word in train_words)
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

        if any(word.unlabeled for word in sample_words)
            raise Exception("Output predict set has unlabeled words")
        return result, end - start

    def _train_impl(self, train_words):
        raise Exception("Method _train_impl is unimplemented")

    def _predict_impl(self, sample_words):
        raise Exception("Method _predict_impl is unimplemented")

    def evaluate(self, words):
        if any(word.unlabeled for word in train_words):
            raise Exception("Evaluation set has unlabeled words")

        logging.info("Will evaluate %s words", len(words))

        output_words = self._predict_impl(words)

        return self._evaluate(words, output_words)

    def _evaluate(self, words, output_words):
        return {
            "morfessor_evaluate": base.morfessor_evaluate(output_words, words),
            "quality": base.measure_quality(output_words, words)
        }

def NeuralMorphemmer(Morphemmer):
    def __init__(self, params):
        self.partitioner = Partitioner(**params)

    def _prepare_input_data(self, words):
        source = []
        targets = []
        for word in words:
            source.append(word.get_word())
            targets.append(word.get_labels())
        return source, targets

    def get_name(self):
        return "Neural morpheme segmentation"

    def _train_impl(self, train_words):
        inputs, targets = self._prepare_input_data(train_words)
        self.partitioner.train(inputs, targets)

    def _predict_impl(self, sample_words):
        inputs, _ = self._prepare_input_data(sample_words)
        return self.partitioner._predict_probs(inputs)
