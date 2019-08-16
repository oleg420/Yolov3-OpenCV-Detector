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
parser.add_argument('--nms_threshold', type=float, default=0.25, help='NMS threshold value')
parser.add_argument('--nn_input', type=str, default='320,416,512', help='Input size of YOLOv3 CNN')
args = parser.parse_args()

print('Threshold: %f' % args.threshold)
print('NMS threshold: %f' % args.nms_threshold)
print('Input size: %s' % args.nn_input)

configs = args.config.split(',')
nn_inputs = args.nn_input.split(',')
if len(configs) != len(nn_inputs):
    print('Error')
    exit(1)

yoloDetectors = []
for i in range(len(nn_inputs)):
    yoloDetectors.append(Detector(config=configs[i], weights=args.weights, classes=args.classes, nn_input=int(nn_inputs[i])))

source = isNum(args.source)
cap = cv2.VideoCapture(source)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    _, image = cap.read()
    boxes = []
    classConfidences = []
    classIDs = []

    for i in range(len(nn_inputs)):
        tmpBoxes, tmpClassConfidences, tmpClassIDs = yoloDetectors[i].detect(image, confidenceThreshold=args.threshold)
        boxes += tmpBoxes
        classConfidences += tmpClassConfidences
        classIDs += tmpClassIDs

    classes, confidences, boxes = yoloDetectors[0].NMSCompress(image, boxes, classIDs, classConfidences, confidenceThreshold=args.threshold, nmsThreshold=args.nms_threshold)

    for i in range(len(classes)):
        image = yoloDetectors[0].draw(image, '%s: %f' % (classes[i], round(confidences[i], 2)), boxes[i])

    cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
