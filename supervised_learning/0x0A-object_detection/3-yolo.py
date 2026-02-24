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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """[summary]

        Args:
            boxes ([type]): [description]
            box_confidences ([type]): [description]
            box_class_probs ([type]): [description]

        Returns:
            [type]: [description]
        """
        f_boxes1 = []
        box_c_1 = []
        scores_list = []
        for out_idx in range(len(boxes)):
            box = boxes[out_idx]
            box_confidence = box_confidences[out_idx]
            box_class_prob = box_class_probs[out_idx]
            obj_class_prob = box_class_prob * box_confidence
            box_classes = np.argmax(obj_class_prob, axis=-1)
            box_scores = np.max(obj_class_prob, axis=-1, keepdims=False)
            mask = box_scores >= self.class_t
            filtered_boxes = box[mask]
            box_classes = box_classes[mask]
            box_scores = box_scores[mask]
            f_boxes1.extend(filtered_boxes.tolist())
            box_c_1.extend(box_classes.tolist())
            scores_list.extend(box_scores.tolist())
        f_boxes1 = np.array(f_boxes1)
        box_c_1 = np.array(box_c_1)
        scores_list = np.array(scores_list)
        return f_boxes1, box_c_1, scores_list

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """[summary]

        Args:
            filtered_boxes ([type]): [description]
            box_classes ([type]): [description]
            box_scores ([type]): [description]

        Returns:
            [type]: [description]
        """
        i_nd_aux = np.lexsort((-box_scores, box_classes))
        p_box_1 = np.array([filtered_boxes[i] for i in i_nd_aux])
        p_box_c_2 = np.array([box_classes[i] for i in i_nd_aux])
        scores_box_1 = np.array([box_scores[i] for i in i_nd_aux])
        _, class_counts = np.unique(p_box_c_2, return_counts=True)
        i = 0
        accumulated_count = 0
        for class_count in class_counts:
            while i < accumulated_count + class_count:
                j = i + 1
                while j < accumulated_count + class_count:
                    tmp = self.iou(p_box_1[i],
                                   p_box_1[j])
                    if tmp > self.nms_t:
                        p_box_1 = np.delete(p_box_1, j,
                                            axis=0)
                        scores_box_1 = np.delete(scores_box_1,
                                                 j, axis=0)
                        p_box_c_2 = (np.delete
                                     (p_box_c_2,
                                      j, axis=0))
                        class_count -= 1
                    else:
                        j += 1
                i += 1
            accumulated_count += class_count
        return p_box_1, p_box_c_2, scores_box_1

    @staticmethod
    def iou(box1, box2):
        """[summary]

        Args:
            box1 ([type]): [description]
            box2 ([type]): [description]

        Returns:
            [type]: [description]
        """
        X__1 = max(box1[0], box2[0])
        Y__2 = max(box1[1], box2[1])
        x__2 = min(box1[2], box2[2])
        y__2 = min(box1[3], box2[3])
        area_1 = max(y__2 - Y__2, 0) * max(x__2 - X__1, 0)
        area_2 = (box1[3] - box1[1]) * (box1[2] -
                                        box1[0]) + (box2[3] - box2[1]) * \
            (box2[2] - box2[0]) - area_1

        return area_1 / area_2

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
