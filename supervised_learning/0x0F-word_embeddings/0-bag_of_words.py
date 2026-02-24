#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """[summary]

    Args:
        sentences ([type]): [description]
        vocab ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    vec = CountVectorizer(vocabulary=vocab)
    x = vec.fit_transform(sentences)
    return x.toarray(), vec.get_feature_names()
