#-*- coding: utf-8 -*-
from catboost import CatBoostClassifier
import subprocess
import json
from base import MorphemeLabel, BMES_PREFIXES
from tempfile import NamedTemporaryFile
import logging

VOWELS = {
    'а', 'и', 'е', 'ё', 'о', 'у', 'ы', 'э', 'ю', 'я'
}
CIPH = {
    'о': int(1000000 * 0.10983),
    'е': int(1000000 * 0.08483),
    'а': int(1000000 * 0.07998),
    'и': int(1000000 * 0.07367),
    'н': int(1000000 * 0.067),
    'т': int(1000000 * 0.06318),
    'с': int(1000000 * 0.05473),
    'р': int(1000000 * 0.04746),
    'в': int(1000000 * 0.04533),
    'л': int(1000000 * 0.04343),
    'к': int(1000000 * 0.03486),
    'м': int(1000000 * 0.03203),
    'д': int(1000000 * 0.02977),
    'п': int(1000000 * 0.02804),
    'у': int(1000000 * 0.02615),
    'я': int(1000000 * 0.02001),
    'ы': int(1000000 * 0.01898),
    'ь': int(1000000 * 0.01735),
    'г': int(1000000 * 0.01687),
    'з': int(1000000 * 0.01641),
    'б': int(1000000 * 0.01592),
    'ч': int(1000000 * 0.0145),
    'й': int(1000000 * 0.01208),
    'х': int(1000000 * 0.00966),
    'ж': int(1000000 * 0.0094),
    'ш': int(1000000 * 0.00718),
    'ю': int(1000000 * 0.00639),
    'ц': int(1000000 * 0.00486),
    'щ': int(1000000 * 0.00361),
    'э': int(1000000 * 0.00331),
    'ф': int(1000000 * 0.00267),
    'ъ': int(1000000 * 0.00037),
    'ё': int(1000000 * 0.00013),
    '-': int(0),
}
def get_word_info(word, morf_info):
    def _if_none(x):
        if x is "_":
            return ''
        return x
    normal_form = _if_none(morf_info['lemma'])
    pos = _if_none(morf_info['speech_part'])
    case = _if_none(morf_info['case'])
    gender = _if_none(morf_info['gender'])
    number = _if_none(morf_info['number'])
    tense = _if_none(morf_info['tense'])
    length = len(word)
    stem_length = 0
    for letter1, letter2 in zip(word, normal_form.upper()):
        if letter1 != letter2:
            break
        stem_length += 1
    return (pos, case, gender, number, tense, length, stem_length)

def _get_prev(word, index):
    if index <= 0:
        return ''
    return word[index - 1]

def _get_next(word, index):
    if index >= len(word) - 1:
        return ''
    return word[index + 1]

class CatboostSpliter(object):

    PARTS_MAPPING = dict(
        [(v.value, num) for num, v in enumerate(MorphemeLabel) if v.value is not None] + [('B-SUFF', len(MorphemeLabel) - 1)]
    )

    def __init__(self, xmorphy_path):
        print(self.PARTS_MAPPING)
        self.model = CatBoostClassifier(learning_rate=0.1, depth=8, loss_function='MultiClass', classes_count=len(self.PARTS_MAPPING), task_type ="GPU")
        self.xmorphy_path = xmorphy_path

    def _get_parse_repr(self, word, morph_info):
        global_index = 0
        result_x = []
        result_y = []
        word_text = word.get_word()
        word_features = get_word_info(word_text, morph_info)
        for part in word.morphemes:
            part_text = part.part_text
            part_answer = part.get_simple_labels()
            for answer, letter in zip(part_answer, part_text):
                prev_letter = _get_prev(word_text, global_index)
                next_letter = _get_next(word_text, global_index)
                prev_prev_letter = _get_prev(word_text, global_index - 1)
                next_next_letter = _get_next(word_text, global_index + 1)
                next_next_next_letter = _get_next(word_text, global_index + 2)
                prev_prev_prev_letter = _get_prev(word_text, global_index - 2)
                if letter in VOWELS:
                    vow = "VOWEL"
                else:
                    vow = "CONSONANT"
                letter_features = [letter, vow, global_index, CIPH[letter],
                                   prev_prev_prev_letter, prev_prev_letter, prev_letter,
                                   next_letter, next_next_letter, next_next_next_letter]
                letter_features += word_features
                result_x.append(letter_features)
                result_y.append(self.PARTS_MAPPING[answer])
                global_index += 1
        return result_x, result_y

    def _get_morph_info(self, words):
        f = NamedTemporaryFile()
        for word in words:
            f.write((word.get_word() + "\n").encode('utf-8'))

        logging.info("Starting morphological analyzer")
        with open('../data/cross_result.json', 'r') as anal_data:
            output = json.load(anal_data)
        #output = subprocess.check_output("{} -i {} -j -d".format(self.xmorphy_path, f.name), shell=True)
        logging.info("Finished morphological analyzer")
        morph_info = {}
        for k, info in output.items():
            _, word = k.split("_")
            morph_info[word] = info['0']
        return morph_info

    def _prepare_words(self, words):
        morph_info = self._get_morph_info(words)
        result_x, result_y = [], []
        for word in words:
            word_text = word.get_word()
            if word_text not in morph_info:
                raise Exception("Word {} is not found in infos".format(word_text))
            word_x, word_answer = self._get_parse_repr(word, morph_info[word_text])
            result_x += word_x
            result_y += word_answer
        return result_x, result_y

    def train(self, words):
        x, y = self._prepare_words(words)
        print('X:' + str('\n'.join([str(v) for v in x[:10]])))
        print('Y:' + str(y[:10]))
        self.model.fit(x, y, cat_features=[0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

    def _transform_classification(self, parse):
        parts = []
        print('Got:' + str(parse))
        current_part = [parse[0]]
        for num, letter in enumerate(parse[1:]):
            index = num + 1
            if letter == 'SUFF' and parse[index - 1] == 'B-SUFF':
                current_part.append(letter)
            elif letter != parse[index - 1]:
                parts.append(current_part)
                current_part = [letter]
            else:
                current_part.append(letter)
        for part in parts:
            if part[0] == 'B-SUFF':
                part[0] = 'SUFF'
            if len(part) == 1:
                part[0] = 'S-' + part[0]
            else:
                part[0] = 'B-' + part[0]
                part[-1] = 'E-' + part[-1]
                for num, letter in enumerate(part[1:-1]):
                    part[num] = 'M-' + letter
        result = []
        for part in parts:
            result += part
        return result


    def classify(self, words):
        x, _ = self._prepare_words(words)
        pred_class = self.model.predict(x)
        print('PredClassLen:' + str(len(pred_class)))
        print('X len:' + str(len(x)))

        start = 0
        reverse_mapping = {v:k for k, v in self.PARTS_MAPPING.items()}
        print(reverse_mapping)
        result = []
        for word in words:
            word_len = len(word)
            end = start + word_len
            print(word.get_word())
            print('Real:' + str(word.get_simple_labels()))
            print('Start:' + str(start))
            print('End:' + str(end))
            print('WOrd:' + str(len(word)))
            parse = self._transform_classification([reverse_mapping[int(num)] for num in pred_class[start:end]])
            print("transformed:" + str(parse))
            result.append(parse)
            start = end
        return result
