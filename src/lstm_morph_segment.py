from base import MorphemeLabel
import json
import numpy as np
import ahocorasick
from keras.models import Sequential, Model
from keras.layers import LSTM, Bidirectional, GRU, CuDNNLSTM
from keras.layers import Dense, Input, Concatenate, Masking
from keras.layers import TimeDistributed, Dropout
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from attention import Attention
import datetime
import time

mask_value = 0

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
    morpheme_type = morphemes[0]
    if morpheme_type not in ["B-PREF", "PREF", "B-ROOT", "ROOT"]:
        return False
    if '-' in morphemes[-1]:
        _, morpheme_type = morphemes[-1].split('-')
    else:
        morpheme_type = morphemes[-1]
    if morpheme_type not in ["ROOT", "SUFF", "END", "POSTFIX", "B-SUFF"]:
        return False
    for i, morpheme in enumerate(morphemes[:-1]):
        if morphemes[i+1] not in get_next_morpheme(morpheme):
            return False
    return True

class Corrector(object):
    def __init__(self, target_symbols):
        self.target_symbols = [''] + target_symbols
        self.target_symbol_codes = {v: k for k, v in enumerate(self.target_symbols)}
        self.target_symbols_number = len(self.target_symbols)
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

class LSTMSpliter(object):
    F_AUTOMATA_CACHE = {}
    B_AUTOMATA_CACHE = {}

    PARTS_MAPPING = {
        'PREF': 1,
        'ROOT': 2,
        'SUFF': 3,
        'END': 4,
        'LINK': 5,
        'HYPH': 6,
        'POSTFIX': 7,
        'B-SUFF': 8,
        'B-PREF': 9,
        'B-ROOT': 10,
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

    VOWELS = {
        'а', 'и', 'е', 'ё', 'о', 'у', 'ы', 'э', 'ю', 'я'
    }

    CIPH = {
        'о': 0.10983,
        'е': 0.08483,
        'а': 0.07998,
        'и': 0.07367,
        'н': 0.067,
        'т': 0.06318,
        'с': 0.05473,
        'р': 0.04746,
        'в': 0.04533,
        'л': 0.04343,
        'к': 0.03486,
        'м': 0.03203,
        'д': 0.02977,
        'п': 0.02804,
        'у': 0.02615,
        'я': 0.02001,
        'ы': 0.01898,
        'ь': 0.01735,
        'г': 0.01687,
        'з': 0.01641,
        'б': 0.01592,
        'ч': 0.0145,
        'й': 0.01208,
        'х': 0.00966,
        'ж': 0.0094,
        'ш': 0.00718,
        'ю': 0.00639,
        'ц': 0.00486,
        'щ': 0.00361,
        'э': 0.00331,
        'ф': 0.00267,
        'ъ': 0.00037,
        'ё': 0.00013,
        '-': 0.0,
    }


    def _get_morph_info(self, words):
        with open(self.morph_info_path, 'r') as morph_data:
            output = json.load(morph_data)
            return output

    def __init__(self, models_number, dropout, morph_info,
                 layers, optimizer, activation, validation_split, epochs):
        self.models_number = models_number
        self.dropout = dropout
        self.morph_info_path = morph_info
        self.layers = layers
        self.optimizer = optimizer
        self.activation = activation
        self.validation_split = validation_split
        self.epochs = epochs
        self.maxlen = None
        self.pos_dct = {}
        self.case_dct = {}
        self.gender_dct = {}
        self.number_dct = {}
        self.tense_dct = {}
        self.models = []
        self.corrector = Corrector(
            ['PREF', 'ROOT', 'SUFF', 'END', 'LINK',
             'HYPH', 'POSTFIX', 'B-SUFF', 'B-PREF', 'B-ROOT'])


    def _get_word_info(self, word, morf_info):
        def _if_none(x, dct):
            info = ''
            if x is not "_":
                info = x
            if info not in dct:
                dct[info] = len(dct)
            return dct[info]

        normal_form = morf_info['lemma']
        pos = to_categorical(_if_none(morf_info['speech_part'], self.pos_dct), num_classes=18)
        case = to_categorical(_if_none(morf_info['case'], self.case_dct), num_classes=13)
        gender = to_categorical(_if_none(morf_info['gender'], self.gender_dct), num_classes=4)
        number = to_categorical(_if_none(morf_info['number'], self.number_dct), num_classes=3)
        tense = to_categorical(_if_none(morf_info['tense'], self.tense_dct), num_classes=4)
        length = len(word)
        stem_length = 0
        for letter1, letter2 in zip(word, normal_form):
            if letter1 != letter2:
                break
            stem_length += 1
        return pos.tolist() + case.tolist() + gender.tolist() + number.tolist() + tense.tolist() + [stem_length]

    def _get_parse_repr(self, word, morph_info):
        features = []
        word_text = word.get_word()
        word_morph_info = self._get_word_info(word_text, morph_info[word.get_word()])
        for index, letter in enumerate(word_text):
            letter_features = word_morph_info.copy()
            vovelty = 0
            if letter in self.VOWELS:
                vovelty = 1
            letter_features.append(vovelty)
            letter_features += to_categorical(self.LETTERS[letter], num_classes=len(self.LETTERS) + 1).tolist()
            features.append(letter_features)

        X = np.array(features, dtype=np.int8)
        Y = np.array([to_categorical(self.PARTS_MAPPING[label], num_classes=len(self.PARTS_MAPPING) + 1) for label in word.get_simple_labels()])
        return X, Y

    def _pad_sequences(self, Xs, Ys):
        if not self.maxlen:
            newXs = pad_sequences(Xs, dtype=np.int8, padding='post', value=mask_value)
            newYs = pad_sequences(Ys, padding='post', value=mask_value)
            maxlen1 = max(len(f) for f in newXs)
            maxlen2 = max(len(f) for f in newYs)
            assert(maxlen1 == maxlen2)
            self.maxlen = maxlen1
        else:
            newXs = pad_sequences(Xs, padding='post', dtype=np.int8, maxlen=self.maxlen)
            newYs = pad_sequences(Ys, padding='post', maxlen=self.maxlen)
        return newXs, newYs, self.maxlen

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
            it = automata.items(part)
            res = set([])
            for child_word, _ in it:
                if len(child_word) == letter_index:
                    continue
                next_letter = child_word[letter_index]
                res.add(next_letter)
            cache[part] = len(res)
        return cache[part]

    def get_backward_feature(self, automata, word, letter_index):
        return self.get_forward_feature(
            automata, ''.join(reversed(word)),
            len(word) - (letter_index + 1), False)

    def _prepare_words(self, words):
        morph_info = self._get_morph_info(words)
        print(list(morph_info.items())[0])
        result_x, result_y = [], []
        for i, word in enumerate(words):
            word_x, word_answer = self._get_parse_repr(word, morph_info)
            result_x.append(word_x)
            result_y.append(word_answer)

        return self._pad_sequences(result_x, result_y)

    def _build_model(self, input_maxlen):
        inp = Input(shape=(input_maxlen, 79))
        inputs = [inp]
        prev_do = None
        for drop, units in zip(self.dropout, self.layers):
            bidi = Bidirectional(CuDNNLSTM(units, return_sequences=True))(inp)
            do = Dropout(drop)(bidi)
            if prev_do is None:
                inp = Concatenate(axis=-1)([do, inp])
            else:
                inp = Concatenate(axis=-1)([do, prev_do])
            prev_do = do

        outputs = [TimeDistributed(
            Dense(len(self.PARTS_MAPPING) + 1, activation=self.activation))(inp)]
        self.models.append(Model(inputs, outputs=outputs))
        self.models[-1].compile(loss='categorical_crossentropy',
                                optimizer=self.optimizer, metrics=['acc'])

    def train(self, words):
        self.fb, self.bb = self.build_automations([word.get_word() for word in words])
        x, y, self.maxlen = self._prepare_words(words)
        for i in range(self.models_number):
            self._build_model(self.maxlen)
        es = EarlyStopping(monitor='val_acc', patience=10, verbose=1)
        for i, model in enumerate(self.models):
            model.fit(x, y, epochs=self.epochs, verbose=2,
                      callbacks=[es], validation_split=self.validation_split)
            model.save("keras_model_{}.h5".format(int(time.time())))

    def _transform_classification(self, parse):
        parts = []
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
        print("Total models:", len(self.models))
        x, _, _ = self._prepare_words(words)
        pred = np.mean([model.predict(x) for model in self.models], axis=0)
        pred_class = pred.argmax(axis=-1)
        reverse_mapping = {v: k for k, v in self.PARTS_MAPPING.items()}
        result = []
        corrected = 0
        for i, word in enumerate(words):
            cutted_prediction = pred_class[i][:len(word.get_word())]
            raw_parse = [reverse_mapping[int(num)] for num in cutted_prediction]
            raw_probs = pred[i][:len(word.get_word())]
            corrected_parse = self.corrector.decode_best(raw_probs)
            if corrected_parse != raw_parse:
                print("Word:", word.get_word())
                print("Raw:", raw_parse)
                print("Corrected:", corrected_parse)
                corrected += 1
                raw_parse = corrected_parse
            parse = self._transform_classification(raw_parse)
            result.append(parse)
        print("Totally corrected:", corrected)
        return result

    def measure_time(self, words, times):
        x, _, _ = self._prepare_words(words)
        result = []
        for i in range(times):
            start = time.time()
            pred = np.mean([model.predict(x) for model in self.models], axis=0)
            finish = time.time()
            result.append(finish - start)
        return result
