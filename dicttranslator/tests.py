#-*- coding: utf-8 -*-
import unittest
from base import split_word_to_word_labels, convert_word_to_features_vectors, get_common_parses

class TestBaseMethods(unittest.TestCase):

    def test_split(self):
        word1, labels1 = split_word_to_word_labels('абажур	абажур:ROOT')
        self.assertEqual(word1, 'абажур')
        self.assertEqual(labels1, ['B-ROOT', 'ROOT', 'ROOT', 'ROOT', 'ROOT', 'ROOT'])
        word2, labels2 = split_word_to_word_labels('абонент	абон:ROOT/ент:SUFF')
        self.assertEqual(word2, 'абонент')
        self.assertEqual(labels2, ['B-ROOT', 'ROOT', 'ROOT', 'ROOT', 'B-SUFF', 'SUFF', 'SUFF'])
        word3, labels3 = split_word_to_word_labels('автоматчик	автомат:ROOT/ч:SUFF/ик:SUFF')
        self.assertEqual(word3, 'автоматчик')
        self.assertEqual(labels3, ['B-ROOT', 'ROOT', 'ROOT', 'ROOT', 'ROOT', 'ROOT', 'ROOT', 'B-SUFF', 'B-SUFF', 'SUFF'])

    def test_convert(self):
        X, Y = convert_word_to_features_vectors(
            'абажур	абажур:ROOT',
            'абажур	а:PREF/бажур:ROOT',
            2
        )
        self.assertEqual(len(X), len('абажур'))
        self.assertEqual(len(Y), len('абажур'))
        self.assertEqual(len(X[0]), 10)
        self.assertEqual(X[0], ['NO_LETTER', 'NO_LABEL', 'NO_LETTER', 'NO_LABEL', 'а', 'B-ROOT', 'б', 'ROOT', 'а', 'ROOT'])
        self.assertEqual(Y, [0, 2, 3, 3, 3, 3])

    def test_get_common_parses(self):
        f1 = [
            'абажур	абажур:ROOT',
            'абажурчик	абажур:ROOT/чик:SUFF',
            'абдуктивный	аб:PREF/дукт:ROOT/ив:SUFF/н:SUFF/ый:END',
        ]
        f2 = [
            'абажур	а:PREF/бажур:ROOT',
            'абажурчик	абажур:ROOT/чик:SUFF',
            'абонент	абон:ROOT/ент:SUFF',
        ]
        c = get_common_parses(f1, f2)
        def find_word(common, parse):
            for p1, p2 in common:
                if p1 == parse or p2 == parse:
                    return True
            return False

        self.assertTrue(find_word(c, 'абажур	а:PREF/бажур:ROOT'))
        self.assertTrue(find_word(c, 'абажур	абажур:ROOT'))
        self.assertTrue(find_word(c, 'абажурчик	абажур:ROOT/чик:SUFF'))
        self.assertFalse(find_word(c, 'абдуктивный	аб:PREF/дукт:ROOT/ив:SUFF/н:SUFF/ый:END'))
        self.assertFalse(find_word(c, 'абонент	абон:ROOT/ент:SUFF'))


if __name__ == '__main__':
    unittest.main()

