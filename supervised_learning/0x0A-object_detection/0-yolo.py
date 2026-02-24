#!/usr/bin/env python3
"""[summary]
"""
import tensorflow.keras as K


class Yolo:
    """[summary]
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """[summary]

        Args:
            model_path ([type]): [description]
            classes_path ([type]): [description]
            class_t ([type]): [description]
            nms_t ([type]): [description]
            anchors ([type]): [description]
        """

        self.model = K.models.load_model(
            model_path, {'GlorotUniform': K.initializers.glorot_uniform()})

        self.class_names = open(classes_path,
                                'r').read().splitlines()
        self.nms_t = nms_t
        self.anchors = anchors
        self.class_t = class_t
