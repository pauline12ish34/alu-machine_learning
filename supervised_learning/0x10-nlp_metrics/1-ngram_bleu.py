#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def ngram_bleu(references, sentence, n):
    """[summary]

    Args:
        references ([type]): [description]
        sentence ([type]): [description]
        n ([type]): [description]

    Returns:
        [type]: [description]
    """
    references = n_g_t(references, n)
    sentence = n_g_t(sentence, n)
    dict_sen = {}
    for x in sentence:
        dict_sen[x] = dict_sen.get(x, 0) + 1
    max_dict = {}
    for reference in references:
        this_ref = {}
        for x in reference:
            this_ref[x] = this_ref.get(x, 0) + 1
        for x in this_ref:
            max_dict[x] = max(
                max_dict.get(x, 0), this_ref[x])
    in_ref = 0
    for x in dict_sen:
        in_ref += min(
            max_dict.get(x, 0), dict_sen[x])
    aux = np.argmin(
        np.abs([len(ref) - len(sentence)
                for ref in references]))
    aux = len(references[aux])
    if len(sentence) >= aux:
        brevity = 1
    else:
        brevity = np.exp(
            1 - (aux + n - 1) / (
                len(sentence) + n - 1))
    return brevity * in_ref / len(sentence)


def n_g_t(corpus, n):
    """[summary]

    Args:
        corpus ([type]): [description]
        n ([type]): [description]

    Returns:
        [type]: [description]
    """
    aut_n = 0

    if type(corpus[0]) is not list:
        corpus = [corpus]
        aut_n = 1

    t_c = []

    for line in corpus:

        f_s = []
        for x in range(len(line) - n + 1):
            str_gram = ""

            for i in range(n):
                if i != 0:
                    str_gram += " "
                str_gram += line[x + i]
            f_s.append(str_gram)

        t_c.append(f_s)

    if aut_n:
        return t_c[0]

    return t_c
