import cv2
import argparse

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
    parser.add_argument('--source', type=arg_source, required=True, help='Video source (Web-camera number or file or URL)')
    parser.add_argument('--config', type=str, required=True, help='Path to Yolov3 cfg file')
    parser.add_argument('--weights', type=str, required=True, help='Path to Yolov3 weights file')
    parser.add_argument('--classes', type=str, required=True, help='Path to Yolov3 class file')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold value')
    parser.add_argument('--nms_threshold', type=float, default=0.15, help='NMS threshold value')
    parser.add_argument('--nn_input', type=int, default=416, help='Input size of Yolov3 CNN')
    parser.add_argument('--backend', type=arg_backend, default='cpu', help='Select OpenCV backend (CPU or CUDA)')
    args = parser.parse_args()
    print(args)

    yolo_detector = Detector(config=args.config, weights=args.weights, classes=args.classes, backend=args.backend, nn_input=args.nn_input)

    cap = cv2.VideoCapture(args.source)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        _, image = cap.read()

        results = yolo_detector.detect(image, threshold=args.threshold, nms_threshold=args.nms_threshold)
        image = yolo_detector.draw(image, results)

        cv2.imshow('Image', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
