#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow as tf
import tensorflow_hub as hub
import os


def semantic_search(corpus_path, sentence):
    """[summary]

    Args:
        corpus_path ([type]): [description]
        sentence ([type]): [description]

    Returns:
        [type]: [description]
    """
    embeding_model = hub.load(
        "https://tfhub.dev/google/universal-sentence-encoder-large/5")
    aux = -2
    best_match = ''
    rfi_data = os.listdir(corpus_path)
    rfi_data = [x for x in rfi_data if x.endswith('.md')]
    for ref_file in rfi_data:
        with open(
            corpus_path + "/" +
                ref_file) as f:
            text_data = f.read()
            ref_embed = embeding_model([text_data])
        sim = tf.tensordot(
            embeding_model(
                [sentence]),
            ref_embed,
            axes=[[1],
                  [1]])
        if aux < sim:
            aux = sim
            best_match = text_data
    return best_match
