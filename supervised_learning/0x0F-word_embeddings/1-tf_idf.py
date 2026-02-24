#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """[summary]

    Args:
        sentences ([type]): [description]
        vocab ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    vec = TfidfVectorizer(vocabulary=vocab)
    x = vec.fit_transform(sentences)
    return x.toarray(), vec.get_feature_names()
