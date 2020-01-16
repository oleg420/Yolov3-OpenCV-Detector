import argparse
import glob
import os
import cv2
import datetime

from Detector import Detector

def arg_source(x):
    try:
        return int(x)
    except ValueError:
        return str(x)

def arg_backend(x):
    if x.upper().lower() in ['gpu', 'cuda', 'nvidia']:
        return 'cuda'
    elif x.upper().lower() in ['cpu']:
        return 'cpu'
    else:
        return 'cpu'

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Yolov3 Detector')
    parser.add_argument('--path', type=arg_source, required=True, help='Path with .jpg and .png photos')
    parser.add_argument('--config', type=str, required=True, help='Path to Yolov3 cfg file')
    parser.add_argument('--weights', type=str, required=True, help='Path to Yolov3 weights file')
    parser.add_argument('--classes', type=str, required=True, help='Path to Yolov3 class file')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold value')
    parser.add_argument('--nms_threshold', type=float, default=0.15, help='NMS threshold value')
    parser.add_argument('--nn_input', type=int, default=416, help='Input size of Yolov3 CNN')
    parser.add_argument('--backend', type=arg_backend, default='cpu', help='Select OpenCV backend (CPU or CUDA)')
    args = parser.parse_args()
    print(args)
    print()
    
    yolo_detector = Detector(config=args.config, weights=args.weights, classes=args.classes, backend=args.backend, nn_input=args.nn_input)
    
    imagePaths = glob.glob(os.path.abspath(args.path) + '/*.jpg')
    imagePaths += glob.glob(os.path.abspath(args.path) + '/*.png')
    imagePaths.sort()

    for path in imagePaths:
        image = cv2.imread(path)
        height, width, _ = image.shape

        if height > 720:
            image = yolo_detector.imageResize(image=image, height=720)
            height, width, _ = image.shape
        elif width > 1280:
            image = yolo_detector.imageResize(image=image, width=1080)
            height, width, _ = image.shape

        print('Image: %s' % path)

        raw_boxes, confidences, classes = yolo_detector.detect(image, confidence_threshold=args.threshold)
        boxes = yolo_detector.NMS(image, raw_boxes, confidences,
                                  confidence_threshold=args.threshold,
                                  nms_threshold=args.nms_threshold)

        for (obj_class, confidences, box) in zip(classes, confidences, boxes):
            image = yolo_detector.draw(image, '%s: %.2f' % (obj_class, confidences), box)
            print('%s: %.2f - %s' % (obj_class, confidences, str(box)))

        cv2.imshow('%s' % path, image)

        print('Enter anything to continue (q to exit)')
        if cv2.waitKey() & 0xFF == ord('q'):
            print('Terminating')
            exit(0)
        cv2.destroyAllWindows()
        print()

    cv2.waitKey()
    cv2.destroyAllWindows()
