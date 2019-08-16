Yolov3 Detector Super

Detect objects on image using different image resolutions. (Edit WIDTH and HEIGTH in .cfg file)
If you like to use only 1 resolution - just select 1 resolution))

Requirements:
- Python 3 or higher
- OpenCV 3.4 or higher

Flags example:
--config=yolov3-tiny-320.cfg,yolov3-tiny-416.cfg,yolov3-tiny-512.cfg
--weights=yolov3-tiny.weights
--classes=classes.names
--path=/images
--nn_input=320,416,512
--threshold=0.5
