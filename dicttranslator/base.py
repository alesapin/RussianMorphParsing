#-*- coding: utf-8 -*-
from collections import defaultdict
from tensorflow.keras.utils import to_categorical
import re

def find_all(stroka, find):
    return [m.start() for m in re.finditer(find, stroka)]

ANSWER_FORWARD_MAPPER = {
    'B-PREF': 1,
    'PREF': 2,
    'B-ROOT': 3,
    'ROOT': 4,
    'B-SUFF': 5,
    'SUFF': 6,
    'B-END': 7,
    'END': 8,
    'B-POSTFIX': 9,
    'POSTFIX': 10,
    'B-LINK': 11,
    'HYPH': 12,
}

ANSWER_REVERSE_MAPPER = {
    v:k for k, v in ANSWER_FORWARD_MAPPER.items()
}

LETTERS = {
    'о': 1,
    'е': 2,
    'а': 3,
    'и': 4,
    'н': 5,
    'т': 6,
    'с': 7,
    'р': 8,
    'в': 9,
    'л': 10,
    'к': 11,
    'м': 12,
    'д': 13,
    'п': 14,
    'у': 15,
    'я': 16,
    'ы': 17,
    'ь': 18,
    'г': 19,
    'з': 20,
    'б': 21,
    'ч': 22,
    'й': 23,
    'х': 24,
    'ж': 25,
    'ш': 26,
    'ю': 27,
    'ц': 28,
    'щ': 29,
    'э': 30,
    'ф': 31,
    'ъ': 32,
    'ё': 33,
    '-': 34,
}

def _get_prev(word, index, default):
    if index <= 0:
        return default
    return word[index - 1]

def _get_next(word, index, default):
    if index >= len(word) - 1:
        return default
    return word[index + 1]

def _get_features_for_single_letter(word, labels, index, window_size):
    result = []
    for i in reversed(range(window_size)):
        result.append(_get_prev(word, index - i, 'NO_LETTER'))
        result.append(_get_prev(labels, index - i, 'NO_LABEL'))

    result.append(word[index])
    result.append(labels[index])

    for i in range(window_size):
        result.append(_get_next(word, index + i, 'NO_LETTER'))
        result.append(_get_next(labels, index + i, 'NO_LABEL'))

    return result

def _get_neural_for_single_letter(word, labels, index):
    result = to_categorical(LETTERS[word[index]], num_classes=len(LETTERS) + 1).tolist()
    result += to_categorical(ANSWER_FORWARD_MAPPER[labels[index]], num_classes=len(ANSWER_FORWARD_MAPPER) + 1).tolist()
    return result


def split_word_to_word_labels(raw_data):
    word, parse = raw_data.split('\t')
    parts = parse.split('/')
    labels = []
    for part in parts:
        part_text, label = part.split(':')
        cur_labels = [label] * len(part_text)
        cur_labels[0] = 'B-' + cur_labels[0]
        labels += cur_labels
    return word, labels

def convert_words_to_features_vectors(raw_data_inp, raw_data_outp, window_size):
    X = []
    feature_word, feature_labels = split_word_to_word_labels(raw_data_inp)
    answer_word, answer_labels = split_word_to_word_labels(raw_data_outp)
    if feature_word != answer_word:
        raise Exception("Words to train are not the same {} != {}".format(feature_word, answer_word))
    for index, _ in enumerate(feature_word):
        X.append(_get_features_for_single_letter(feature_word, feature_labels, index, window_size))
    Y = [ ANSWER_FORWARD_MAPPER[label] for label in answer_labels ]
    return X, Y

def convert_words_to_neural_vectors(raw_data_inp, raw_data_outp):
    X = []
    feature_word, feature_labels = split_word_to_word_labels(raw_data_inp)
    answer_word, answer_labels = split_word_to_word_labels(raw_data_outp)
    if feature_word != answer_word:
        raise Exception("Words to train are not the same {} != {}".format(feature_word, answer_word))
    for index, _ in enumerate(feature_word):
        X.append(_get_neural_for_single_letter(feature_word, feature_labels, index))
    Y = [ to_categorical(ANSWER_FORWARD_MAPPER[label], num_classes=len(ANSWER_FORWARD_MAPPER) + 1) for label in answer_labels]
    return X, Y

def convert_word_to_features_vectors(raw_data_inp, window_size):
    X = []
    feature_word, feature_labels = split_word_to_word_labels(raw_data_inp)
    for index, _ in enumerate(feature_word):
        X.append(_get_features_for_single_letter(feature_word, feature_labels, index, window_size))
    return X

def get_common_parses(f1, f2):
    joiner = defaultdict(list)
    for f in (f1, f2):
        for line in f:
            stripped = line.strip()
            word, parse = stripped.split('\t')
            joiner[word].append(stripped)

    return [tuple(pair) for pair in joiner.values() if len(pair) == 2]

def get_new_parses_for_first(f1, f2):
    not_new = set([])
    for line in f1:
        word = line.strip().split('\t')[0]
        not_new.add(word)
    result = []
    for line in f2:
        stripped = line.strip()
        word, parse = stripped.split('\t')
        if word not in not_new and len(find_all(stripped, 'ROOT')) == 1 and '-' not in stripped:
            result.append(stripped)
    return result

def measure_quality(predicted_targets, targets, words):
    TP, FP, FN, equal, total = 0, 0, 0, 0, 0
    BE = ['{}-{}'.format(x, y) for x in "B" for y in ["ROOT", "PREF", "SUFF", "END", "LINK", "None"]]
    corr_words = 0
    for corr, pred, word in zip(targets, predicted_targets, words):
        corr_len = len(corr)
        pred_len = len(pred)
        boundaries = [i - 1 for i in range(corr_len) if i != 0 and corr[i] in BE]
        pred_boundaries = [i - 1 for i in range(pred_len) if i != 0 and pred[i] in BE]
        common = [x for x in boundaries if x in pred_boundaries]
        TP += len(common)
        FN += len(boundaries) - len(common)
        FP += len(pred_boundaries) - len(common)
        equal += sum(int(x==y) for x, y in zip(corr, pred))
        total += len(corr)
        corr_words += (corr == pred)
    metrics = ["Precision", "Recall", "F1", "Accuracy", "Word accuracy"]
    results = [TP / (TP+FP), TP / (TP+FN), TP / (TP + 0.5*(FP+FN)),
               equal / total, corr_words / len(targets)]
    return list(zip(metrics, results))

def pretty_morpheme(word, labels):
    start = 0
    result = []
    if len(word) == 1:
        return word + ':' + labels[0]

    current_label = labels[0].split('B-')[1]
    current_word = word[0]
    word_index = 1
    for label in labels[1:]:
        if label.startswith('B-'):
            result.append(current_word + ':' + current_label)
            current_word = word[word_index]
            current_label = label.split('B-')[1]
        else:
            current_word += word[word_index]
        word_index += 1

    result.append(current_word + ':' + current_label)
    return '/'.join(result)
