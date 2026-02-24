#!/usr/bin/env python3
"""[summary]
Returns:
    [type]: [description]
"""
import tensorflow.keras as K
import numpy as np


def sigmoid(x):
    """[summary]
    Args:
        x ([type]): [description]
    Returns:
        [type]: [description]
    """
    return 1 / (1 + np.exp(-x))


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
        custom_objects = {'GlorotUniform': K.initializers.glorot_uniform()}
        self.model = K.models.load_model(model_path, custom_objects)
        self.class_names = open(classes_path, 'r').read().splitlines()
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """[summary]
        Args:
            outputs ([type]): [description]
            image_size ([type]): [description]
        Returns:
            [type]: [description]
        """
        data = ([], [], [])
        all_anchor_sizes = self.anchors
        anchor = 0

        img_h = image_size[0]
        img_w = image_size[1]

        for output in outputs:
            anchor_sizes = all_anchor_sizes[anchor]
            anchor += 1

            boxes = np.zeros(output[:, :, :, 0:4].shape)
            boxes[:, :, :, :] = output[:, :, :, 0:4]
            box_confidences = sigmoid(output[:, :, :, np.newaxis, 4])
            box_class_probs = sigmoid(output[:, :, :, 5:])

            gh = output.shape[0]
            gw = output.shape[1]
            for i in range(output.shape[0]):
                for j in range(output.shape[1]):
                    cy = i
                    cx = j
                    boxes[i, j, :, 0] = (sigmoid(output[i, j, :, 0]) + cx) / gw
                    boxes[i, j, :, 1] = (sigmoid(output[i, j, :, 1]) + cy) / gh

            inp_h = self.model.input.shape[2].value
            inp_w = self.model.input.shape[1].value

            pw = anchor_sizes[:, 0]
            ph = anchor_sizes[:, 1]

            boxes[:, :, :, 2] = pw * np.exp(output[:, :, :, 2]) / inp_w
            boxes[:, :, :, 3] = ph * np.exp(output[:, :, :, 3]) / inp_h

            coordinates = np.zeros(boxes.shape)
            coordinates[:, :, :, :] = boxes[:, :, :, :]

            bx = boxes[:, :, :, 0]
            by = boxes[:, :, :, 1]
            bw = boxes[:, :, :, 2]
            bh = boxes[:, :, :, 3]

            coordinates[:, :, :, 0] = (bx - bw / 2) * img_w
            coordinates[:, :, :, 1] = (by - bh / 2) * img_h
            coordinates[:, :, :, 2] = (bx + bw / 2) * img_w
            coordinates[:, :, :, 3] = (by + bh / 2) * img_h

            data[0].append(coordinates)
            data[1].append(box_confidences)
            data[2].append(box_class_probs)
        return data
