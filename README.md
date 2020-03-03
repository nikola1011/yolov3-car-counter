# YoloV3 Car counter
This is a demo project that uses YoloV3 neural network to count vehicles on a given video. The detection happens every x frames where x can be specified. Other times the dlib library is used for tracking previously detected vehicles. Furthermore, you can edit confidence detection level, number of frames to count vehicle as detected before removing it from trackable list, the maximum distance from centroid (CentroidTracker class).

YoloV3 model is pretrained and downloaded (Internet connection is required for the download process).
## Dependencies
Install dependencies via pip specified by requirements.txt file.
The code is tested and run with Python 3.7.4 and Python 3.5.6 on Ubuntu 18.04.3 LTS.
(Windows 10 platforms should also be able to run the project)
## Demo
You can see the demo of the project via the gif below.
![Gif of a demo project could not be loaded](https://github.com/nikola1011/yolov3-car-counter/blob/master/demo-yolov3-dlib-window-rec.gif)
