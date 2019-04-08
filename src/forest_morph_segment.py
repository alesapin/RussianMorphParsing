# -*- coding: utf-8 -*-
from catboost import CatBoostClassifier
import catboost
import subprocess
import json
import ahocorasick
from base import MorphemeLabel
import numpy as np
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
    for letter1, letter2 in zip(word, normal_form):
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

def get_next_morpheme_types(morpheme_type):
    """
    Определяет, какие морфемы могут идти за текущей.
    """
    if morpheme_type == "PREF":
        return ["END", "LINK", "ROOT", "PREF", "END", "HYPH", "POSTFIX"]
    if morpheme_type == "ROOT":
        return ["SUFF", "END", "LINK", "ROOT", "HYPH", "POSTFIX"]
    if morpheme_type == "SUFF":
        return ["SUFF", "END", "HYPH", "POSTFIX", "LINK"]
    if morpheme_type == "LINK":
        return ["ROOT", "HYPH", "PREF"]
    if morpheme_type == "HYPH":
        return ["ROOT", "PREF"]
    if morpheme_type == "POSTFIX":
        return ["END", "POSTFIX"]
    return ["END", "POSTFIX"]

def get_next_morpheme(morpheme):
    if '-' in morpheme:
        _, morpheme_type = morpheme.split("-")
    else:
        morpheme_type = morpheme

    next_types = get_next_morpheme_types(morpheme_type)
    if 'SUFF' in next_types:
        next_types.append('B-SUFF')

    if 'PREF' in next_types:
        next_types.append('B-PREF')
    if 'ROOT' in next_types:
        next_types.append('B-ROOT')

    return next_types

def is_correct_morpheme_sequence(morphemes):
    if morphemes == []:
        return False
    if '-' in morphemes[0]:
        return False
    morpheme_type = morphemes[0]
    if morpheme_type not in ["B-PREF", "PREF", "B-ROOT", "ROOT"]:
        #print("Not correct because first symbol is not root and prefix")
        return False
    if '-' in morphemes[-1]:
        _, morpheme_type = morphemes[-1].split('-')
    else:
        morpheme_type = morphemes[-1]
    if morpheme_type not in ["ROOT", "SUFF", "END", "POSTFIX", "B-SUFF"]:
        #print("Not correct because last symbol is not in:", ["ROOT", "SUFF", "END", "POSTFIX", "B-SUFF"])
        return False
    for i, morpheme in enumerate(morphemes[:-1]):
        if morphemes[i+1] not in get_next_morpheme(morpheme):
            #print("Not correct beause morpheme", morphemes[i+1], '[', i+1, "] not in next list:", get_next_morpheme(morpheme))
            return False
    return True

class Corrector(object):
    def __init__(self, target_symbols):
        self.target_symbols = target_symbols
        self.target_symbol_codes = {v: k for k, v in enumerate(self.target_symbols)}
        self.target_symbols_number = len(target_symbols)
        self.incorrect = 0

    def decode_best(self, probs):
        length = len(probs)
        best_states = np.argmax(probs, axis=1)
        best_labels = [self.target_symbols[state_index] for state_index in best_states]
        if not is_correct_morpheme_sequence(best_labels):
            self.incorrect += 1
            initial_costs = [np.inf] * self.target_symbols_number
            initial_states = [None] * self.target_symbols_number
            initial_costs[best_states[0]], initial_states[best_states[0]] = -np.log(probs[0][best_states[0]]), best_states[0]
            costs, states = [initial_costs], [initial_states]
            for i in range(length - 1):
                state_order = np.argsort(costs[-1])
                curr_costs = [np.inf] * self.target_symbols_number
                prev_states = [None] * self.target_symbols_number
                inf_count = self.target_symbols_number
                for prev_state in state_order:
                    if np.isinf(costs[-1][prev_state]):
                        break
                    possible_states = self.get_possible_next_states(prev_state)
                    for state in possible_states:
                        if np.isinf(curr_costs[state]):
                            curr_costs[state] = costs[-1][prev_state] - np.log(probs[i + 1, state])
                            prev_states[state] = prev_state
                            inf_count -= 1
                    if inf_count == 0:
                        break
                costs.append(curr_costs)
                states.append(prev_states)
            possible_states = [num for num, k in enumerate(self.target_symbols) if k != 'PREF']
            best_states = [min(possible_states, key=(lambda x: costs[-1][x]))]
            for j in range(length - 1, 0, -1):
                best_states.append(states[j][best_states[-1]])
            best_states = best_states[::-1]
        probs_to_return = np.zeros(shape=(length, self.target_symbols_number), dtype=np.float32)
        for j, state in enumerate(best_states[:-1]):
            possible_states = self.get_possible_next_states(state)
            probs_to_return[j, possible_states] = probs[j + 1, possible_states]
        return [self.target_symbols[i] for i in best_states]

    def get_possible_next_states(self, state_index):
        state = self.target_symbols[state_index]
        next_states = get_next_morpheme(state)
        return [self.target_symbol_codes[x] for x in next_states if x in self.target_symbol_codes]


class CatboostSpliter(object):

    PARTS_MAPPING = dict(
        [
            (v.value, num) for num, v in enumerate(MorphemeLabel) if v.value is not None] + [('B-SUFF', len(MorphemeLabel) - 1), ('B-PREF', len(MorphemeLabel)), ('B-ROOT', len(MorphemeLabel) + 1)]
    )
    F_AUTOMATA_CACHE = {}
    B_AUTOMATA_CACHE = {}

    def __init__(self, morph_info_path, model_path):
        self.morph_info_path = morph_info_path

        self.model = CatBoostClassifier(learning_rate=0.03, depth=9, loss_function='MultiClass', classes_count=len(self.PARTS_MAPPING))
        if model_path is None:
            self.loaded = False
        else:
            self.model.load_model(model_path)
            self.loaded = True
        self.corrector = Corrector([v.value for v in MorphemeLabel if v.value is not None] + ['B-SUFF', 'B-PREF', 'B-ROOT'])

    def build_automations(self, words):
        forward_automation = ahocorasick.Automaton()
        backward_automation = ahocorasick.Automaton()
        for word in words:
            forward_automation.add_word(word, 1)

        for word in words:
            backward_automation.add_word(''.join(reversed(word)), 1)
        forward_automation.make_automaton()
        backward_automation.make_automaton()
        return forward_automation, backward_automation

    def get_forward_feature(self, automata, word, letter_index, forward=True):
        if letter_index == 0:
            return 0
        part = word[:letter_index]
        cache = self.F_AUTOMATA_CACHE if forward else self.B_AUTOMATA_CACHE
        if part not in cache:
            it = automata.items(part);
            res = set([])
            for child_word, _ in it:
                if len(child_word) == letter_index:
                    continue
                next_letter = child_word[letter_index]
                res.add(next_letter)
            cache[part] = len(res)
        return cache[part]

    def get_backward_feature(self, automata, word, letter_index):
        return self.get_forward_feature(automata, ''.join(reversed(word)), len(word) - (letter_index + 1), False)

    def _get_parse_repr(self, word, morph_info, fb, bb):
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
                next_next_next_next_letter = _get_next(word_text, global_index + 3)
                prev_prev_prev_prev_letter = _get_prev(word_text, global_index - 3)
                n_next_next_next_next_letter = _get_next(word_text, global_index + 4)
                p_prev_prev_prev_prev_letter = _get_prev(word_text, global_index - 4)
                if letter in VOWELS:
                    vow = "VOWEL"
                elif letter != '-':
                    vow = "CONSONANT"
                else:
                    vow = "_"
                harris_forward = self.get_forward_feature(fb, word_text, global_index)
                harris_backward = self.get_backward_feature(bb, word_text, global_index)
                letter_features = [
                    letter, vow, global_index, CIPH[letter],
                    p_prev_prev_prev_prev_letter,
                    prev_prev_prev_prev_letter,
                    prev_prev_prev_letter, prev_prev_letter, prev_letter,
                    next_letter, next_next_letter, next_next_next_letter,
                    next_next_next_next_letter,
                    n_next_next_next_next_letter,
                    harris_forward, harris_backward,
                ]
                letter_features += word_features
                result_x.append(letter_features)
                result_y.append(self.PARTS_MAPPING[answer])
                global_index += 1

        return result_x, result_y

    def _get_morph_info(self, words):
        logging.info("Starting morphological analyzer")
        with open(self.morph_info_path, 'r') as anal_data:
            output = json.load(anal_data)
        logging.info("Finished morphological analyzer")
        morph_info = {}
        for k, info in output.items():
            _, word = k.split("_")
            morph_info[word] = info['0']
        return morph_info

    def _prepare_words(self, words):
        morph_info = self._get_morph_info(words)
        result_x, result_y = [], []
        for i, word in enumerate(words):
            word_text = word.get_word()
            if word_text not in morph_info:
                raise Exception("Word {} is not found in infos".format(word_text))
            word_x, word_answer = self._get_parse_repr(word, morph_info[word_text], self.fb, self.bb)
            result_x += word_x
            result_y += word_answer
            i += 1
        return result_x, result_y

    def train(self, words):
        logging.info("Start building automations")
        self.fb, self.bb = self.build_automations([word.get_word() for word in words])
        logging.info("Finished building automations")
        if not self.loaded:
            x, y = self._prepare_words(words)
            self.model.fit(x, y, cat_features=[0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20])
        else:
            logging.info("Model already trained")

    def _transform_classification(self, parse):
        parts = []
        #print('Got:' + str(parse))
        current_part = [parse[0]]
        for num, letter in enumerate(parse[1:]):
            index = num + 1
            if letter == 'SUFF' and parse[index - 1] == 'B-SUFF':
                current_part.append(letter)
            elif letter == 'PREF' and parse[index - 1] == 'B-PREF':
                current_part.append(letter)
            elif letter == 'ROOT' and parse[index - 1] == 'B-ROOT':
                current_part.append(letter)
            elif letter != parse[index - 1] or letter.startswith('B-'):
                parts.append(current_part)
                current_part = [letter]
            else:
                current_part.append(letter)
        if current_part:
            parts.append(current_part)

        for part in parts:
            if part[0] == 'B-PREF':
                part[0] = 'PREF'
            if part[0] == 'B-SUFF':
                part[0] = 'SUFF'
            if part[0] == 'B-ROOT':
                part[0] = 'ROOT'
            if len(part) == 1:
                part[0] = 'S-' + part[0]
            else:
                part[0] = 'B-' + part[0]
                part[-1] = 'E-' + part[-1]
                for num, letter in enumerate(part[1:-1]):
                    part[num+1] = 'M-' + letter
        result = []
        for part in parts:
            result += part
        return result


    def classify(self, words):
        x, _ = self._prepare_words(words)
        pred_class = self.model.predict(x)
        pred_probs = self.model.predict_proba(x)

        start = 0
        reverse_mapping = {v:k for k, v in self.PARTS_MAPPING.items()}
        result = []
        counter = 0
        for word in words:
            word_len = len(word)
            end = start + word_len
            raw_parse = [reverse_mapping[int(num)] for num in pred_class[start:end]]
            raw_probs = np.asarray([elem for elem in pred_probs[start:end]])
            corrected_parse = self.corrector.decode_best(raw_probs)
            corrected = False
            if raw_parse != corrected_parse:
                counter += 1
                #print(word.get_word(),'\t', '|'.join(raw_parse), '\t', '|'.join(corrected_parse))
                corrected = True
            parse = self._transform_classification(raw_parse)
            result.append(parse)
            start = end
        print("TotalCorrection:", counter)
        return result
