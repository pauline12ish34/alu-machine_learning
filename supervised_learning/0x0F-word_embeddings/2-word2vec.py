#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow.keras as keras
from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5,
                   window=5, negative=5,
                   cbow=True, iterations=5,
                   seed=0, workers=1):
    """[summary]

    Args:
        sentences ([type]): [description]
        size (int, optional): [description]. Defaults to 100.
        min_count (int, optional): [description]. Defaults to 5.
        window (int, optional): [description]. Defaults to 5.
        negative (int, optional): [description]. Defaults to 5.
        cbow (bool, optional): [description]. Defaults to True.
        iterations (int, optional): [description]. Defaults to 5.
        seed (int, optional): [description]. Defaults to 0.
        workers (int, optional): [description]. Defaults to 1.

    Returns:
        [type]: [description]
    """
    w_2v = Word2Vec(sentences,
                    size=size,
                    window=window,
                    min_count=min_count,
                    negative=negative,
                    workers=workers,
                    sg=cbow,
                    seed=seed,
                    iter=iterations)
    w_2v.train(
        sentences,
        total_examples=w_2v.corpus_count,
        epochs=w_2v.epochs
    )
    return w_2v
