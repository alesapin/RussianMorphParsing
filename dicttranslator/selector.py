import argparse
import logging

def opos(parse):
    pos = 0
    for elem in parse.split('/'):
        part, t = elem.split(':')
        if t == 'LINK':
            return pos
        pos += len(part)
    return -1

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    parser = argparse.ArgumentParser(description="Common words selector")
    candidates = {}
    with open('tikhonov.txt', 'r') as f:
        for line in f:
            if 'о:LINK' in line and not '-' in line:
                word, parse = line.strip().split('\t')
                candidates[word] = parse

    cross = {}
    with open('cross_lexica.txt', 'r') as f:
        for line in f:
            word, parse = line.strip().split('\t')
            cross[word] = parse

    result = []
    for word in candidates:
        if candidates[word].count('о:LINK') >= 2:
            continue
        pos = opos(candidates[word])
        left = word[:pos]
        right = word[pos + 1:]
        if left in cross and right not in cross:
            left_parse, right_parse = candidates[word].split('/о:LINK/')
            print('{}\t{}/о:LINK/{}'.format(word, cross[left], right_parse))
        #if right in cross and left not in cross:
        #    left_parse, right_parse = candidates[word].split('/-:HYPH/')
        #    print('{}\t{}/-:HYPH/{}'.format(word, left_parse, cross[right]))
        #elif right in cross:
        #    print(word, candidates[word], right, cross[right])
