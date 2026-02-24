#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import collections
import numpy as np


def uni_bleu(references, sentence):
    """[summary]

    Args:
        references ([type]): [description]
        sentence ([type]): [description]

    Returns:
        [type]: [description]
    """
    aux_d = {}
    for i in references:
        for j in i:
            if (j not in aux_d.keys() or
                    aux_d[j] < i.count(j)):
                aux_d[j] = i.count(j)
    x = {x: 0 for x in sentence}
    words_s = {x: sentence.count(x) for x in sentence}
    for i in references:
        for j in x.keys():
            if j in i:
                x[j] = words_s[j]
    for j in x.keys():
        if j in aux_d.keys():
            x[j] = min(aux_d[j], x[j])
    t = np.argmin(
        [abs(len(x) - len(sentence)) for x in references])
    num_s = len(references[t])
    if len(sentence) > len(
            references[t]):
        tr = 1
    else:
        tr = np.exp(
            1 - float(num_s) / len(
                sentence))
    return tr * sum(x.values()) / len(sentence)
