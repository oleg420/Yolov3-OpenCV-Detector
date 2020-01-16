import cv2
import numpy as np
import argparse
import datetime
import warnings 

class Detector:
    def __init__(self, config, weights, classes, nn_input=416, backend='cpu'):
        self.__detector = cv2.dnn.readNet(weights, config)

        if backend is 'cpu':
            self.__detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self.__detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        elif backend is 'cuda':
            if self.__cv_version_check():
                self.__detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.__detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            else:
                self.__detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                self.__detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                warnings.warn('This OpenCV version is incompatible with CUDA backend. Required OpenCV 4.2.0 or higher')
                warnings.warn(cv2.__version__)
        else:
            self.__detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self.__detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        classes_file = open(classes, 'r')
        self.__classes = [line.strip() for line in classes_file.readlines()]
        self.__scale = float(1/255)
        self.__nn_input = nn_input

    def __cv_version_check(self):
        v = cv2.__version__
        v = [int(v.split('.')[0]), int(v.split('.')[1])]

        if v[0] >= 4 and v[1] >= 2:
            return True
        else:
            return False

    def __get_output_layers(self, detector):
        layer_names = detector.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in detector.getUnconnectedOutLayers()]
        return output_layers

    def detect(self, image, confidence_threshold=0.5):
        height, width, _ = image.shape
        blob = cv2.dnn.blobFromImage(image, self.__scale,
                                     (self.__nn_input, self.__nn_input),
                                     (0, 0, 0), True, crop=False)
        
        self.__detector.setInput(blob)
        results = self.__detector.forward(self.__get_output_layers(self.__detector))

        classes = []
        class_confidences = []
        boxes = []

        for result in results:
            for detection in result:
                class_scores = detection[5:]
                class_ID = np.argmax(class_scores)
                class_confidence = class_scores[class_ID]

                if class_confidence > confidence_threshold:
                    xCenter = int(detection[0] * width)
                    yCenter = int(detection[1] * height)

                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = xCenter - w / 2
                    y = yCenter - h / 2

                    classes.append(self.__classes[class_ID])
                    class_confidences.append(float(class_confidence))
                    boxes.append([x, y, w, h])

        return boxes, class_confidences, classes

    def NMS(self, image, boxes, class_confidences, confidence_threshold, nms_threshold):
        height, width, _ = image.shape
        indices = cv2.dnn.NMSBoxes(boxes, class_confidences, confidence_threshold, nms_threshold)

        r_boxes = []

        for i in indices:
            i = i[0]
            box = boxes[i]
            x_min, y_min = round(box[0]), round(box[1])
            x_max, y_max = round(x_min + box[2]), round(y_min + box[3])

            if x_min < 0:
                x_min = 0
            if x_min > width:
                x_min = width - 1

            if y_min < 0:
                y_min = 0
            if y_min > height:
                y_min = height - 1

            if x_max < 0:
                x_max = 0
            if x_max > width:
                x_max = width - 1

            if y_max < 0:
                y_max = 0
            if y_max > height:
                y_max = height - 1

            r_boxes.append([int(x_min), int(y_min), int(x_max), int(y_max)])

        return r_boxes

    def draw(self, image, label, box, label_color=(255, 255, 255), rec_color=(255, 255, 255),
             label_thickness=2, rec_thickness=2, label_size=0.5):
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), rec_color, rec_thickness)
        cv2.putText(image, label, (box[0] - 10, box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    label_size, label_color, label_thickness)
        return image

    def imageResize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        if width is None and height is None:
            return image

        (h, w) = image.shape[:2]

        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        return cv2.resize(image, dim, interpolation=inter)
