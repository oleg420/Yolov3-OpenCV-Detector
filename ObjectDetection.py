import cv2
import numpy as np

import torch
import torchvision


class ObjectDetection:
    def __init__(self, model_path, cfg_path, classes_path, size=416, device='cpu'):
        self.detector = cv2.dnn.readNet(model_path, cfg_path)

        if device is 'cpu':
            self.detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        elif device is 'cuda':
            self.detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

        self.classes = open(classes_path, 'r').read().splitlines()
        self.scale = float(1 / 255)
        self.size = size

    def __get_output_layers(self, detector):
        layer_names = detector.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in detector.getUnconnectedOutLayers()]
        return output_layers

    def __call__(self, image, threshold=0.5, nms_threshold=0.15):
        height, width, _ = image.shape
        blob = cv2.dnn.blobFromImage(image, self.scale,
                                     (self.size, self.size),
                                     (0, 0, 0), True, crop=False)

        self.detector.setInput(blob)
        results = self.detector.forward(self.__get_output_layers(self.detector))

        records = []

        for result in results:
            r = result[:, 5:] > threshold
            ro, co = np.where(r == True)
            ro = np.unique(ro)
            r = result[ro]

            for result in r:
                class_scores = result[5:]
                cls = np.argmax(class_scores)
                conf = class_scores[cls]

                x = int(result[0] * width)
                y = int(result[1] * height)
                w = int(result[2] * width)
                h = int(result[3] * height)

                x_min = int(x - (w / 2))
                y_min = int(y - (h / 2))
                x_max = int(x + (w / 2))
                y_max = int(y + (h / 2))

                records.append([x_min, y_min, x_max, y_max, cls, float(conf)])

        records = np.array(records)
        if records.size and nms_threshold != 0:
            boxes = torch.tensor(records[:, :4])
            conf = torch.tensor(records[:, 5])
            indices = np.array(torchvision.ops.nms(boxes, conf, nms_threshold))
            records = records[indices]

        return records

    def draw(self, image, results):
        for r in results:
            cv2.rectangle(image, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), (255, 255, 255), 2)
            cv2.putText(image, f'{self.get_class(int(r[4]))} - {round(r[5], 2)}', (int(r[0]) - 10, int(r[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return image

    def get_class(self, index):
        return self.classes[index]

    # def image_resize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
    #     if width is None and height is None:
    #         return image
    #
    #     (h, w) = image.shape[:2]
    #
    #     if width is None:
    #         r = height / float(h)
    #         dim = (int(w * r), height)
    #     else:
    #         r = width / float(w)
    #         dim = (width, int(h * r))
    #
    #     return cv2.resize(image, dim, interpolation=inter)
