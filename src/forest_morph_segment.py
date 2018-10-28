#-*- coding: utf-8 -*-
from catboost import CatBoostClassifier
import subprocess
import json
from base import MorphemeLabel
from tempfile import NamedTemporaryFile

VOWELS = {
    'А', 'И', 'Е', 'Ё', 'О', 'У', 'Ы', 'Э', 'Ю', 'Я'
}
CIPH = {
    'О': int(1000000 * 0.10983),
    'Е': int(1000000 * 0.08483),
    'А': int(1000000 * 0.07998),
    'И': int(1000000 * 0.07367),
    'Н': int(1000000 * 0.067),
    'Т': int(1000000 * 0.06318),
    'С': int(1000000 * 0.05473),
    'Р': int(1000000 * 0.04746),
    'В': int(1000000 * 0.04533),
    'Л': int(1000000 * 0.04343),
    'К': int(1000000 * 0.03486),
    'М': int(1000000 * 0.03203),
    'Д': int(1000000 * 0.02977),
    'П': int(1000000 * 0.02804),
    'У': int(1000000 * 0.02615),
    'Я': int(1000000 * 0.02001),
    'Ы': int(1000000 * 0.01898),
    'Ь': int(1000000 * 0.01735),
    'Г': int(1000000 * 0.01687),
    'З': int(1000000 * 0.01641),
    'Б': int(1000000 * 0.01592),
    'Ч': int(1000000 * 0.0145),
    'Й': int(1000000 * 0.01208),
    'Х': int(1000000 * 0.00966),
    'Ж': int(1000000 * 0.0094),
    'Ш': int(1000000 * 0.00718),
    'Ю': int(1000000 * 0.00639),
    'Ц': int(1000000 * 0.00486),
    'Щ': int(1000000 * 0.00361),
    'Э': int(1000000 * 0.00331),
    'Ф': int(1000000 * 0.00267),
    'Ъ': int(1000000 * 0.00037),
    'Ё': int(1000000 * 0.00013)
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

def CatboostSpliter(object):

    PARTS_MAPPING = {
        part:num for num, part in enumerate(MorphemeLabel) if part.value is not None
    }

    def __init__(self, xmorphy_path):
        self.xmorphy_path = xmorphy_path
        self.model = CatBoostClassifier(learning_rate=0.1, depth=8, thread_count=12, loss_function='MultiClass', classes_count=len(self.PARTS_MAPPING))

    def _get_parse_repr(self, word, morph_info):
        global_index = 0
        result_x = []
        result_y = []
        word_text = word.get_word()
        word_features = get_word_info(word_text, morph_info)
        for part in word.morphemes:
            part_text = part.part_text
            part_answer = part.get_labels()
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
                result_y.append(PARTS_MAPPING[answer])
                global_index += 1
        return result_x, result_y

    def _get_morph_info(self, words):
        f = NamedTemporaryFile()
        for word in words:
            f.write(word.get_word() + "\n")

        output = subprocess.check_output("{} -i {} -j -d".format(self.xmorphy_path, f.name), shell=True)
        morph_info = {}
        for k, info in json.loads(output):
            _, word = k.split("_")
            morph_info[word] = info[0]

    def _prepare_words(self, words):
        morph_info = self._get_morph_info(words)
        result_x, result_y = [], []
        for word in words:
            word_text = word.get_word()
            if word_text not in morph_info:
                raise Exception("Word {} is not found in infos".format(word_text))
            word_x, word_answer = self._get_parse_repr(word, morph_info[word_text])
            result_x.append(word_x)
            result_y.append(result_y)
        return result_x, result_y

    def train(self, words):
        x, y = self._prepare_words(words)
        self.model.fit(x, y, cat_features=[0,1,4,5,6,7,8,9,10,11,12,13,14,15])

    def classify(self, words):
        x, _ = self._prepare_words(words)
        pred_class = self.model.predict(x)
        start = 0
        label_list = list(MorphemeLabel)
        for word in words:
            word_len = len(word)
            parse = [label_list[num] for num in pred_class[start : start+word_len]]
            start += word_len
