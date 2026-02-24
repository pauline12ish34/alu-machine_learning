#/usr/bin/env python3
"""[summary]
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
    print("A:")
