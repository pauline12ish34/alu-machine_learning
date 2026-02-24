#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """[summary]

    Args:
        x ([type]): [description]
        layer_sizes (list, optional): [description]. Defaults to [].
        activations (list, optional): [description]. Defaults to [].

    Returns:
        [type]: [description]
    """
    l_output = create_layer(x, layer_sizes[0], activations[0])
    for i in range(1, len(layer_sizes)):
        l_output = create_layer(l_output, layer_sizes[i], activations[i])
    return l_output
