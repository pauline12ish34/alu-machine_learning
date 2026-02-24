#!/usr/bin/env python3
"""[summary]
"""
qa = __import__('0-qa').question_answer
semantic_search = __import__('3-semantic_search').semantic_search


def question_answer(corpus_path):
    """[summary]

    Args:
        corpus_path ([type]): [description]
    """
    exit = ["exit",
            "quit",
            "goodbye",
            "bye",
            "adios",
            "ciao",
            "sayounara",
            "poka",
            "tschuss",
            "hei hei",
            "zaijian",
            "do skorovo",
            "adieu"]
    while 1:
        input_text = input("Q: ").lower()
        if input_text in exit:
            print("A: Goodbye")
            break

        ref = semantic_search(corpus_path,
                              input_text)
        if len(ref) == 0:
            print("A: Sorry, I do not understand your question.")
            continue
        answer = qa(input_text, ref)
        if answer is None:
            print("A: Sorry, I do not understand your question.")
        else:
            print("A: {}".format(answer))
