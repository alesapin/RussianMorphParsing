#-*- coding: utf-8 -*-
from enum import Enum

class MorphemeLabel(Enum):
    PREF = 'PREF'
    ROOT = 'ROOT'
    SUFF = 'SUFF'
    END = 'END'
    LINK = 'LINK'
    HYPH = 'HYPH'
    POSTFIX = 'POSTFIX'

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
        result = ['B-' + self.label]
        result += [self.label for _ in self.part_text[1:]]
        return result

    def __str__(self):
        return self.part_text + ':' + self.label


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

def parse_morpheme(str_repr, position):
    text, label = str_repr.split(':')
    return Morpheme(text, label, position)

def parse_word(str_repr):
    parts = str_repr.split('/')
    morphemes = []
    global_index = 0
    for part in parts:
        morphemes.append(parse_morpheme(part, global_index))
        global_index += len(part)
    return Word(morphemes)

