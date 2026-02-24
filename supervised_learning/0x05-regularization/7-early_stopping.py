#!/usr/bin/env python3
"""[summary]
    """


def early_stopping(cost, opt_cost, threshold, patience, count):
    """[summary]

    Args:
        cost ([type]): [description]
        opt_cost ([type]): [description]
        threshold ([type]): [description]
        patience ([type]): [description]
        count ([type]): [description]

    Returns:
        [type]: [description]
    """
    if (opt_cost - cost) > threshold:
        return False, 0
    count += 1
    if count < patience:
        return False, count
    return True, count
