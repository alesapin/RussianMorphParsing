# -*- coding: utf-8 -*-
import base
from base import parse_word
import time
import logging
from neural_morph_segm import Partitioner
from forest_morph_segment import CatboostSpliter
import morfessor


class Morphemmer(object):
    def __init__(self, params, dict_name):
        pass

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
            "quality": base.measure_quality(
                output_words,
                [w.get_labels() for w in words],
                words)
        }


class NeuralMorphemmer(Morphemmer):
    def __init__(self, params, dict_name):
        model_params = params.setdefault('model_params', {})
        self.partitioner = Partitioner(**model_params)

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
    def __init__(self, params, dict_name):
        if dict_name == 'cross_lexica':
            path = params["cross_lexica_morph_info"]
        else:
            path = params["tikhonov_morph_info"]
        self.classifier = CatboostSpliter(path)

    @staticmethod
    def get_name():
        return "catboost"

    def _train_impl(self, train_words):
        self.classifier.train(train_words)

    def _predict_impl(self, sample_words):
        return self.classifier.classify(sample_words)


class MorfessorMorphemmer(Morphemmer):
    def __init__(self, params, dict_name):
        self.unlabeled_data_path = params['unlabeled_data']

    @staticmethod
    def get_name():
        return "morfessor"

    def _train_impl(self, train_words):
        pass

    def _predict_impl(self, sample_words):
        result = []
        for word in sample_words:
            text = word.get_word()
            result.append(w.morphemes)
        return result

    def _evaluate(self, output_words, words):
        def evaluator(output_words, words):
            TP, FP, FN = 0, 0, 0

            metrics = ["Точность", "Полнота", "F1-мера"]
            for pred, corr in zip(output_words, words):
                corr_boundaries = [0]
                for morpheme in corr.morphemes:
                    corr_boundaries.append(corr_boundaries[-1] + len(morpheme))
                corr_boundaries.pop(0)
                pred_boundaries = [0]
                for morpheme in pred:
                    pred_boundaries.append(pred_boundaries[-1] + len(morpheme))

                common = [x for x in corr_boundaries if x in pred_boundaries]
                TP += len(common)
                FN += len(corr_boundaries) - len(common)
                FP += len(pred_boundaries) - len(common)
            results = [TP / (TP+FP), TP / (TP+FN), TP / (TP + 0.5*(FP+FN))]
            answer = list(zip(metrics, results))
            return answer
        return {
            "quality": evaluator(output_words, words)
        }


class CrossMorphyMorpemmer(Morphemmer):

    def __init__(self, params, dict_name):
        self.words = {}
        if dict_name == 'cross_lexica':
            path = params['xmorphy_cross_data']
        else:
            path = params['xmorphy_tikhonov_data']
        with open(path, 'r') as fdata:
            for line in fdata:
                word, parse = line.strip().split('\t')
                self.words[word] = parse_word(line.strip())

    @staticmethod
    def get_name():
        return "xmorphy"

    def _train_impl(self, train_words):
        pass

    def _predict_impl(self, sample_words):
        result = []
        for word in sample_words:
            result.append(self.words[word.get_word()].get_labels())
        return result
