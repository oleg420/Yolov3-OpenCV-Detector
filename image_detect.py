import argparse
import glob
import os
import cv2
import datetime

from Detector import Detector


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

parser = argparse.ArgumentParser('Detect vehicles on 2d-image, using YOLOv3 CNN')
parser.add_argument('--path', type=str, required=True, help='Path with images')
parser.add_argument('--config', type=str, required=True, help='Path to YOLOv3 cfg file')
parser.add_argument('--weights', type=str, required=True, help='Path to YOLOv3 weights file')
parser.add_argument('--classes', type=str, required=True, help='Path to YOLOv3 class file')
parser.add_argument('--threshold', type=float, default=0.5, help='Threshold value')
parser.add_argument('--nms_threshold', type=float, default=0.4, help='NMS threshold value')
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

imagePaths = glob.glob(os.path.abspath(args.path) + '/*.jpg')
imagePaths += glob.glob(os.path.abspath(args.path) + '/*.png')

imagePaths.sort()

for path in imagePaths:
    image = cv2.imread(path)
    height, width, _ = image.shape

    if height > 720:
        image = imageResize(image=image, height=720)
        height, width, _ = image.shape
    elif width > 1280:
        image = imageResize(image=image, width=1080)
        height, width, _ = image.shape

    print('========= [Path: %s] =========' % path)
    print('W: %d H: %d' % (width, height))

    boxes = []
    classConfidences = []
    classIDs = []

    now = datetime.datetime.now()
    now = datetime.datetime.now()
    for i in range(len(nn_inputs)):
        loop = datetime.datetime.now()
        tmpBoxes, tmpClassConfidences, tmpClassIDs = yoloDetectors[i].detect(image, confidenceThreshold=args.threshold)
        boxes += tmpBoxes
        classConfidences += tmpClassConfidences
        classIDs += tmpClassIDs
        print('Loop %d(%s) compute time: %s' % (i, nn_inputs[i],str(datetime.datetime.now() - loop)))

    classes, confidences, boxes = yoloDetectors[0].NMSCompress(image, boxes, classIDs, classConfidences,
                                                               confidenceThreshold=args.threshold,
                                                               nmsThreshold=args.nms_threshold)
    print('Total compute time: %s' % str(datetime.datetime.now() - now))

    for i in range(len(classes)):
        image = yoloDetectors[0].draw(image, '', boxes[i], label_size=1)

    print('Founded objects: %d' % len(boxes))
    for i in range(len(boxes)):
        print('%s (%f): %s' % (classes[i], round(confidences[i], 2), str(boxes[i])))

    cv2.imshow('%s' % path, image)
    print('Enter anything to continue (q to exit)')
    if cv2.waitKey() & 0xFF == ord('q'):
        print('Terminating')
        exit(0)
    cv2.destroyAllWindows()

cv2.waitKey()
cv2.destroyAllWindows()
