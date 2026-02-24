#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """[summary]

    Args:
        question ([type]): [description]
        reference ([type]): [description]

    Returns:
        [type]: [description]
    """
    mod = 'bert-large-uncased-whole-word-masking-finetuned-squad'
    tokenizer = BertTokenizer.from_pretrained(mod)
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
    question_tokens = tokenizer.tokenize(question)
    ref_tokens = tokenizer.tokenize(reference)
    tokens = (['[CLS]'] + question_tokens + ['[SEP]'] + ref_tokens
              + ['[SEP]'])
    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
    attent_mask = [1] * len(input_word_ids)
    token_type_ids = ([0] * (1 + len(question_tokens) + 1)
                      + [1] * (len(ref_tokens) + 1))
    input_word_ids, attent_mask, token_type_ids = map(
        lambda t: tf.expand_dims(
            tf.convert_to_tensor(
                t, dtype=tf.int32), 0),
        (input_word_ids,
         attent_mask,
         token_type_ids))
    outputs = model(
        [input_word_ids,
         attent_mask,
         token_type_ids]
    )
    short_start = tf.argmax(
        outputs[0][0][1:]) + 1
    short_end = tf.argmax(
        outputs[1][0][1:]) + 1
    answer_tokens = tokens[short_start: short_end + 1]
    answer = tokenizer.convert_tokens_to_string(
        answer_tokens)
    if len(answer) <= 1:
        return None
    return answer
