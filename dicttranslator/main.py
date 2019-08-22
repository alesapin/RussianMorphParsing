import argparse
from multiprocessing import cpu_count
from catboost_model import ForestModel
from neural_model import NeuralModel
from base import get_new_parses_for_first
import logging
import random

random.seed(42)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    parser = argparse.ArgumentParser(description="Dict converter model")
    parser.add_argument("--convert-from", required=True)
    parser.add_argument("--convert-to", required=True)
    parser.add_argument("--model", choices=("LSTM", "GBDT"))
    parser.add_argument("--learning-rate", default=0.03) #
    parser.add_argument("--iterations", default=7000) #
    parser.add_argument("--thread-count", default=cpu_count()) #
    parser.add_argument("--depth", default=8) #
    parser.add_argument("--window-size", default=5) #
    parser.add_argument("--split-factor", default=0.2)
    parser.add_argument("--num_lstms", default=1) #
    parser.add_argument("--dropouts", default=[0.2], nargs='+') #
    parser.add_argument("--units", default=[512], nargs='+') #
    parser.add_argument("--output-model-name")
    parser.add_argument("--load-model")
    parser.add_argument("--classify", action='store_true')

    args = parser.parse_args()
    if args.model == "GBDT":
        model = ForestModel(args.learning_rate, args.depth, args.iterations, args.thread_count, args.window_size)
    elif args.model == "LSTM":
        model = NeuralModel(args.num_lstms, args.dropouts, args.units)
    model.prepare(args.convert_from, args.convert_to, args.split_factor)
    if args.load_model:
        model.load(args.load_model)
    else:
        model.train()
    if args.output_model_name:
        model.save([ args.output_model_name + str(i) + '.h5' for i in range(args.num_lstms)])
    quality = model.test(not args.classify)
    for name, value in quality:
        print(name, ":", value)

    if args.classify:
        with open(args.convert_from, 'r') as f1, open(args.convert_to, 'r') as f2:
            new_queries = get_new_parses_for_first(f2, f1)
            logging.info("Total new words %s", len(new_queries))
            result = model.classify(new_queries)
            print('\n'.join(result))


