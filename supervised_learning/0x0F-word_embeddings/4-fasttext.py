#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
from gensim.models import FastText


def fasttext_model(sentences, size=100,
                   min_count=5, negative=5,
                   window=5, cbow=True,
                   iterations=5, seed=0,
                   workers=1):
    """[summary]

    Args:
        sentences ([type]): [description]
        size (int, optional): [description]. Defaults to 100.
        min_count (int, optional): [description]. Defaults to 5.
        negative (int, optional): [description]. Defaults to 5.
        window (int, optional): [description]. Defaults to 5.
        cbow (bool, optional): [description]. Defaults to True.
        iterations (int, optional): [description]. Defaults to 5.
        seed (int, optional): [description]. Defaults to 0.
        workers (int, optional): [description]. Defaults to 1.

    Returns:
        [type]: [description]
    """
    f_t = FastText(
        size=size, window=window,
        min_count=min_count,
        negative=negative,
        sg=cbow, seed=seed,
        workers=workers
    )
    f_t.build_vocab(sentences=sentences)
    f_t.train(
        sentences=sentences,
        total_examples=len(sentences),
        epochs=iterations
    )
    return f_t
