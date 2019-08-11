#-*- coding: utf-8 -*-
from morphemmer import ForestMorphemmer, NeuralMorphemmer, MorfessorMorphemmer, CrossMorphyMorpemmer, LSTMMorphemmer
from base import parse_word
import logging
import argparse
import json
import random

random.seed(42)

AVAILABLE_MODELS = {
    'GBDT': ForestMorphemmer,
    'CNN': NeuralMorphemmer,
    'Morfessor': MorfessorMorphemmer,
    'CrossMorphy': CrossMorphyMorpemmer,
    'LSTM': LSTMMorphemmer,
}

def split_words(words, factor):
    random.shuffle(words)
    len_train = int(len(words) * (1 - factor))
    return words[:len_train], words[len_train:]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s.%(msecs)03d %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")

    parser = argparse.ArgumentParser(description="Test several models for morphemic segmentation")
    parser.add_argument("--models-config", required=True)
    parser.add_argument("--train-data-path", required=True)
    parser.add_argument("--test-data-path", required=True)
    parser.add_argument("--model", choices=AVAILABLE_MODELS.keys(), required=True)
    parser.add_argument(
        "--dictionary", choices=('cross_lexica', 'tikhonov',), required=True)
    parser.add_argument("--load", default="")

    args = parser.parse_args()

    with open(args.models_config, 'r') as config_file:
        models_config = json.load(config_file)
    logging.info("Config initialized")

    ModelType = AVAILABLE_MODELS[args.model]
    logging.info("Creating model %s", ModelType.get_name())
    params = models_config.setdefault(ModelType.get_name(), {})
    model = ModelType(params, args.dictionary)
    words = []
    counter = 0
    logging.info("Start loading data from file %s", args.train_data_path)
    train_part = []
    with open(args.train_data_path, 'r') as data:
        for num, line in enumerate(data):
            counter += 1
            train_part.append(parse_word(line.strip()))
            if counter % 1000 == 0:
                logging.info("Loaded %ld words", counter)


    logging.info("Start loading data from file %s", args.test_data_path)
    test_part = []
    with open(args.test_data_path, 'r') as data:
        for num, line in enumerate(data):
            counter += 1
            test_part.append(parse_word(line.strip()))
            if counter % 1000 == 0:
                logging.info("Loaded %ld words", counter)

    logging.info("Totally loaded %ld words", len(words))
    logging.info("Split dataset on train %ld words and test %ld words", len(train_part), len(test_part))

    logging.info("Training model '%s'", model.get_name())
    model.train(train_part)
    logging.info("Train of model '%s' finished", model.get_name())

    evaluation_results = {}
    logging.info("Evaluating model '%s'", model.get_name())
    evaluation_results[model.get_name()] = model.evaluate(test_part)
    logging.info("Evaluation of model '%s' finished", model.get_name())

    print(evaluation_results)
