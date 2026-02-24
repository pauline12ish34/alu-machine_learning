#!/usr/bin/env python3
"""[summary]
    Returns:
        [type]: [description]
    """
import tensorflow.keras as K
import numpy as np
import cv2
import glob


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

    def filter_boxes(self, boxes, box_confidences, box_class_prot_s):
        """[summary]
        Args:
            boxes ([type]): [description]
            box_confidences ([type]): [description]
            box_class_prot_s ([type]): [description]
        Returns:
            [type]: [description]
        """
        f_boxes1 = []
        box_c_1 = []
        scores_list = []
        for out_idx in range(len(boxes)):
            box = boxes[out_idx]
            box_confidence = box_confidences[out_idx]
            box_class_prob = box_class_prot_s[out_idx]
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

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """[summary]
        Args:
            image ([type]): [description]
            boxes ([type]): [description]
            box_classes ([type]): [description]
            box_scores ([type]): [description]
            file_name ([type]): [description]
        """
        for i in range(len(boxes)):
            score = "{:.2f}".format(box_scores[i])
            s_init = (int(boxes[i, 0]), int(boxes[i, 1]))
            es_end = (int(boxes[i, 2]), int(boxes[i, 3]))
            image = cv2.rectangle(image,
                                  s_init, es_end,
                                  (255, 0, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (int(boxes[i, 0]), int(boxes[i, 1] - 5))
            image = cv2.putText(image,
                                self.class_names[box_classes[i]] + score,
                                org,
                                font,
                                0.5,
                                (0, 0, 255), 1,
                                cv2.LINE_AA)
        cv2.imshow(file_name, image)
        if cv2.waitKey(0) == ord('s'):
            os.mkdir('detections') if not os.path.isdir('detections') else None
            os.chdir('detections')
            cv2.imwrite(file_name, image)
            os.chdir('../')
        cv2.destroyAllWindows()

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
    def load_images(folder_path):
        """[summary]
        Args:
            folder_path ([type]): [description]
        Returns:
            [type]: [description]
        """
        path_i1 = glob.glob(folder_path + "/*")
        l_path_2 = [cv2.imread(i) for i in path_i1]

        return l_path_2, path_i1

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

    def preprocess_images(self, images):
        """[summary]
        Args:
            images ([type]): [description]
        Returns:
            [type]: [description]
        """
        im_list_1 = []
        im_shapes_1 = []
        for i in images:
            im_shapes_1.append([i.shape[0],
                                i.shape[1]]
                               )
            new_size = (self.model.input.shape[1],
                        self.model.input.shape[2]
                        )
            img_resized = (cv2.resize(i,
                                      new_size,
                                      interpolation=cv2.INTER_CUBIC)) / 255
            im_list_1.append(img_resized)
        return (np.array(im_list_1),
                np.array(im_shapes_1)
                )

    def predict(self, folder_path):
        """[summary]

        Args:
            folder_path ([type]): [description]

        Returns:
            [type]: [description]
        """
        imgs, paths = self.load_images(folder_path)
        p_imgs, i_shapes = self.preprocess_images(imgs)

        y_hat = self.model.predict(p_imgs)
        num_grids = len(y_hat)
        output = []

        for i in range(y_hat[0].shape[0]):

            aux_o = []
            for j in range(num_grids):
                aux_o.append(y_hat[j][i])

            res, res_c, res_p = self.process_outputs(aux_o, i_shapes[i])
            fil, t_o, t_s = self.filter_boxes(res, res_c, res_p)

            boxes, box_class, box_scores = self.non_max_suppression(
                fil, t_o, t_s)
            path = paths[i].split("/")[-1]

            output.append((boxes, box_class, box_scores))

            self.show_boxes(imgs[i], boxes, box_class, box_scores, path)

        return output, paths

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
            box_class_prot_s = sigmoid(output[:, :, :, 5:])

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

            res = boxes[:, :, :, 0]
            by = boxes[:, :, :, 1]
            bw = boxes[:, :, :, 2]
            bh = boxes[:, :, :, 3]

            coordinates[:, :, :, 0] = (res - bw / 2) * img_w
            coordinates[:, :, :, 1] = (by - bh / 2) * img_h
            coordinates[:, :, :, 2] = (res + bw / 2) * img_w
            coordinates[:, :, :, 3] = (by + bh / 2) * img_h

            data[0].append(coordinates)
            data[1].append(box_confidences)
            data[2].append(box_class_prot_s)
        return data
