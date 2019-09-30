import cv2
import numpy as np
import argparse
import datetime

class Detector:
    def __init__(self, config, weights, classes, nn_input=416):
        self.__detector = cv2.dnn.readNet(weights, config)

        self.__detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.__detector.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)

        classesFile = open(classes, 'r')
        self.__classes = [line.strip() for line in classesFile.readlines()]
        self.__scale = float(1/255)
        self.__nn_input = nn_input

    def __getOutputLayers(self, detector):
        layerNames = detector.getLayerNames()
        outputLayers = [layerNames[i[0] - 1] for i in detector.getUnconnectedOutLayers()]
        return outputLayers

    def getClasses(self):
        return self.__classes

    def detect(self, image, confidenceThreshold=0.5):
        height, width, _ = image.shape
        blob = cv2.dnn.blobFromImage(image, self.__scale,
                                     (self.__nn_input, self.__nn_input),
                                     (0, 0, 0), True, crop=False)
        self.__detector.setInput(blob)
        results = self.__detector.forward(self.__getOutputLayers(self.__detector))

        classIDs = []
        classConfidences = []
        boxes = []

        for result in results:
            for detection in result:
                classScores = detection[5:]
                classID = np.argmax(classScores)
                classConfidence = classScores[classID]
                if classConfidence > confidenceThreshold:
                    xCenter = int(detection[0] * width)
                    yCenter = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = xCenter - w / 2
                    y = yCenter - h / 2
                    classIDs.append(classID)
                    classConfidences.append(float(classConfidence))
                    boxes.append([x, y, w, h])

        return boxes, classConfidences, classIDs

    def NMSCompress(self, image, boxes, classIDs, classConfidences, confidenceThreshold, nmsThreshold):
        height, width, _ = image.shape
        indices = cv2.dnn.NMSBoxes(boxes, classConfidences, confidenceThreshold, nmsThreshold)

        rClasses = []
        rBoxes = []

        for i in indices:
            i = i[0]
            box = boxes[i]
            xMin, yMin = round(box[0]), round(box[1])
            xMax, yMax = round(xMin + box[2]), round(yMin + box[3])

            if xMin < 0:
                xMin = 0
            if xMin > width:
                xMin = width - 1

            if yMin < 0:
                yMin = 0
            if yMin > height:
                yMin = height - 1

            if xMax < 0:
                xMax = 0
            if xMax > width:
                xMax = width - 1

            if yMax < 0:
                yMax = 0
            if yMax > height:
                yMax = height - 1

            rClasses.append(self.__classes[classIDs[i]])
            rBoxes.append([int(xMin), int(yMin), int(xMax), int(yMax)])

        return rBoxes, classConfidences, rClasses

    def draw(self, image, label, box, label_color=(255, 255, 255), rec_color=(255, 255, 255),
             label_thickness=2, rec_thickness=2, label_size=0.5):
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), rec_color, rec_thickness)
        cv2.putText(image, label, (box[0] - 10, box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    label_size, label_color, label_thickness)
        return image

    def imageResize(image, width=None, height=None, inter=cv2.INTER_AREA):
        """
        Resize image with saving original proportions.
        :param image: Image itself.
        :param width: New image width.
        :param height: New image height.
        :param inter: Interpolation method.
        :return: New resized image.
        """
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Yolov3 Detector Super')
    parser.add_argument('--image', type=str, required=True, help='Image source')
    parser.add_argument('--config', type=str, required=True, help='Path to YOLOv3 cfg file')
    parser.add_argument('--weights', type=str, required=True, help='Path to YOLOv3 weights file')
    parser.add_argument('--classes', type=str, required=True, help='Path to YOLOv3 class file')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold value')
    parser.add_argument('--nms_threshold', type=float, default=0.15, help='NMS threshold value')
    parser.add_argument('--nn_input', type=str, default='320,416,512', help='Input size of YOLOv3 CNN')
    parser.add_argument('--nms_union', type=int, default=1)
    args = parser.parse_args()

    print('Threshold: %f' % args.threshold)
    print('NMS threshold: %f' % args.nms_threshold)
    print('NMS union: %d' % args.nms_union)
    print('Input size: %s' % args.nn_input)

    nn_inputs = args.nn_input.split(',')

    yoloDetectors = []
    for i in range(len(nn_inputs)):
        yoloDetectors.append(
            Detector(config=args.config, weights=args.weights, classes=args.classes, nn_input=int(nn_inputs[i])))

    image = cv2.imread(args.image)
    height, width, _ = image.shape

    print('========= [Path: %s] =========' % args.image)
    print('W: %d H: %d' % (width, height))

    rawBoxes = []
    rawConfidences = []
    rawClasses = []

    now = datetime.datetime.now()
    for i in range(len(nn_inputs)):
        loop = datetime.datetime.now()
        tmpBoxes, tmpConfidences, tmpClasses = yoloDetectors[i].detect(image, confidenceThreshold=args.threshold)

        if not args.nms_union:
            tmpBoxesStage1, tmpConfidencesStage1, tmpClassesStage1 = yoloDetectors[0].NMSCompress(image, tmpBoxes,
                                                                                                  tmpClasses,
                                                                                                  tmpConfidences,
                                                                                                  confidenceThreshold=args.threshold,
                                                                                                  nmsThreshold=args.nms_threshold)
            rawBoxes += tmpBoxesStage1
            rawConfidences += tmpConfidencesStage1
            rawClasses += tmpClassesStage1
        else:
            rawBoxes += tmpBoxes
            rawConfidences += tmpConfidences
            rawClasses += tmpClasses
        print('Loop %d(%s) compute time: %s' % (i, nn_inputs[i], str(datetime.datetime.now() - loop)))

    if args.nms_union:
        boxes, confidences, classes = yoloDetectors[0].NMSCompress(image, rawBoxes, rawClasses, rawConfidences,
                                                                   confidenceThreshold=args.threshold,
                                                                   nmsThreshold=args.nms_threshold)
    else:
        boxes, confidences, classes = rawClasses, rawConfidences, rawBoxes

    print('Total compute time: %s' % str(datetime.datetime.now() - now))

    for i in range(len(classes)):
        image = yoloDetectors[0].draw(image, '', boxes[i], label_size=1)

    print('Founded objects: %d' % len(boxes))
    for i in range(len(boxes)):
        print('%s (%f): %s' % (classes[i], round(confidences[i], 2), str(boxes[i])))

    cv2.imshow('Detect', image)
    cv2.waitKey()
    cv2.destroyAllWindows()
