#!/usr/bin/env python3
"""[summary]
"""
question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """[summary]

    Args:
        reference ([type]): [description]
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

        answer = question_answer(input_text, reference)
        if answer is None:
            print("Q: Sorry, I do not understand your question.")
        else:
            print("A: {}".format(answer))
