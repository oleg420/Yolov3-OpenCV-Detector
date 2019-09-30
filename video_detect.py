import cv2
import numpy as np
import argparse

from Detector import Detector

def isNum(num):
    try:
        return int(num)
    except ValueError:
        return str(num)

parser = argparse.ArgumentParser('Yolov3 Detector Super')
parser.add_argument('--source', type=str, required=True, help='Image source. (Webcam number or URL)')
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
    yoloDetectors.append(Detector(config=args.config, weights=args.weights, classes=args.classes, nn_input=int(nn_inputs[i])))

source = isNum(args.source)
cap = cv2.VideoCapture(source)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    _, image = cap.read()
    rawBoxes = []
    rawConfidences = []
    rawClasses = []

    for i in range(len(nn_inputs)):
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

    if args.nms_union:
        boxes, confidences, classes = yoloDetectors[0].NMSCompress(image, rawBoxes, rawClasses, rawConfidences,
                                                                   confidenceThreshold=args.threshold,
                                                                   nmsThreshold=args.nms_threshold)
    else:
        boxes, confidences, classes = rawClasses, rawConfidences, rawBoxes

    for i in range(len(classes)):
        image = yoloDetectors[0].draw(image, '%s: %f' % (classes[i], round(confidences[i], 2)), boxes[i])

    cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
