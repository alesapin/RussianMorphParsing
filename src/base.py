#-*- coding: utf-8 -*-
from enum import Enum
from itertools import product

class MorphemeLabel(Enum):
    PREF = 'PREF'
    ROOT = 'ROOT'
    SUFF = 'SUFF'
    END = 'END'
    LINK = 'LINK'
    HYPH = 'HYPH'
    POSTFIX = 'POSTFIX'
    NONE = None


class Morpheme(object):
    def __init__(self, part_text, label, begin_pos):
        self.part_text = part_text
        self.length = len(part_text)
        self.begin_pos = begin_pos
        self.label = label
        self.end_pos = self.begin_pos + self.length

    def __len__(self):
        return self.length

    def get_labels(self):
        if self.length == 1:
            return 'S-' + self.label
        result = ['B-' + self.label]
        result += ['M-' + self.label for _ in self.part_text[1:-1]]
        result += ['E-' + self.label]
        return result

    def __str__(self):
        return self.part_text + ':' + self.label

    @property
    def unlabeled(self):
        return not self.label


class Word(object):
    def __init__(self, morphemes=[]):
        self.morphemes = morphemes

    def append_morpheme(self, morpheme):
        self.morphemes.append(morpheme)

    def get_word(self):
        return ''.join([morpheme.part_text for morpheme in self.morphemes])

    def parts_count(self):
        return len(self.morphemes)

    def suffix_count(self):
        return len([morpheme for morpheme in self.morphemes if morpheme.label == MorphemeLabel.SUFFIX])

    def get_labels(self):
        result = []
        for morpheme in self.morphemes:
            result += morpheme.get_labels()
        return result

    def __str__(self):
        return '/'.join([str(morpheme) for morpheme in self.morphemes])

    @property
    def unlabeled(self):
        return all(p.unlabeled for p in self.morphemes)

def parse_morpheme(str_repr, position):
    text, label = str_repr.split(':')
    return Morpheme(text, MorphemeLabel[label], position)

def parse_word(str_repr):
    parts = str_repr.split('/')
    morphemes = []
    global_index = 0
    for part in parts:
        morphemes.append(parse_morpheme(part, global_index))
        global_index += len(part)
    return Word(morphemes)

def morfessor_evaluate(prediction, target):
    if len(prediction) != len(target):
        raise Exception("Prediction and target sets are not same len")
    def calc_prop_distance(ref, pred):
        if len(ref) == 0:
            return 1.0
        diff = len(set(ref) - set(pred))
        return (len(ref) - diff) / float(len(ref))

    def get_array_repr(word):
        result = []
        shift = 0
        for morpheme in word.morphemes:
            shitf += len(morpheme)
            result.append(shift)
        return result

    sum_precision = 0
    sum_recall = 0
    for pred_word, target_word in zip(prediction, target):
        prediction_array = get_array_repr(pred_word)
        target_array = get_array_repr(target_word)
        sum_precision += max(calc_prop_distance(r, p) for p, r in product(prediction_array, target_array))
        sum_recall += max(calc_prop_distance(p, r) for p, r in product(prediction_array, value_array))

    precision = sum_precision / len(prediction)
    recall = sum_recall / len(prediction)
    f_score = 2.0 / (1.0 / precision + 1.0 / recall)
    return [("Precision", precision), ("Recall", recall), ("F1", f_score)]

def measure_quality(prediction, targets):
    TP, FP, FN, equal, total = 0, 0, 0, 0, 0
    SE = ['{}-{}'.format(x, y) for x in "SE" for y in ["ROOT", "PREF", "SUFF", "END", "LINK", "None"]]
    corr_words = 0
    for corr, pred in zip(targets, predicted_targets):
        corr_len = len(corr)
        pred_len = len(pred)
        boundaries = [i for i in range(corr_len) if corr[i] in SE]
        pred_boundaries = [i for i in range(pred_len) if pred[i] in SE]
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


