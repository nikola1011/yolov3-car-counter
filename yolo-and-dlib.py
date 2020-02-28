# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2017 Taehoon Lee

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Created on Mon Sep 10 19:40:49 2018

@author: Baakchsu
"""


#import tensorflow as tf
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
import tensornets as nets
import cv2
import numpy as np
import time
import dlib
import tensorflow.compat.v1 as tf

#https://github.com/theislab/scgen/issues/14
tf.disable_v2_behavior() 


inputs = tf.placeholder(tf.float32, [None, 416, 416, 3])
model = nets.YOLOv3COCO(inputs, nets.Darknet19)
#model = nets.YOLOv2(inputs, nets.Darknet19)
#model = nets.TinyYOLOv2VOC(inputs, nets.Darknet19)

ct = CentroidTracker(maxDisappeared=5, maxDistance=50)
trackers = []
trackableObjects = {}


#frame=cv2.imread("D://pyworks//yolo//truck.jpg",1)
trackers = []
#tracker = dlib.correlation_tracker()
skip_frames = 10
total = 0

classes={'0':'person','1':'bicycle','2':'car','3':'bike','5':'bus','7':'truck'}
list_of_classes=[0,1,2,3,5,7]
with tf.Session() as sess:
    sess.run(model.pretrained())
#"D://pyworks//yolo//videoplayback.mp4"    
    #video_path = "/home/nikola/Videos/Relaxing highway traffic.mp4"
    #video_path = "/home/nikola/Videos/Road traffic video for object detection and tracking.mp4"
    #video_path = "/home/nikola/Videos/UK Motorway M25 Trucks, Lorries, Cars Highway.mp4"
    video_path = "/home/nikola/Videos/M6 Motorway Traffic.mp4"
    cap = cv2.VideoCapture(video_path)
    width =  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
    skiped_cnt = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        img=cv2.resize(frame,(416,416))
        #img = frame
        rects = []
        if skiped_cnt == skip_frames:
            # DETECTING
            trackers = []

            skiped_cnt = 0
            
            imge=np.array(img).reshape(-1,416,416,3) #.reshape(-1, width, height, 3) #
            start_time=time.time()
            preds = sess.run(model.preds, {inputs: model.preprocess(imge)})
        
            print("--- %s seconds ---" % (time.time() - start_time)) 
            boxes = model.get_boxes(preds, imge.shape[1:3])
            cv2.namedWindow('image',cv2.WINDOW_NORMAL)

            cv2.resizeWindow('image', 700,700)
            #print("--- %s seconds ---" % (time.time() - start_time)) 
            np_boxes = np.array(boxes)
            for j in list_of_classes:
                count =0
                if str(j) in classes:
                    lab=classes[str(j)]
                if len(np_boxes) !=0:
                    for i in range(len(np_boxes[j])):
                        box=np_boxes[j][i] 
                        if np_boxes[j][i][4]>=.40:
                            count += 1
                            cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,255,0),1)
                            cv2.putText(img, lab, (box[0],box[1]), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), lineType=cv2.LINE_AA)
                            
                            startX, startY, endX, endY = box[0], box[1], box[2], box[3]
                            # construct a dlib rectangle object from the bounding
				            # box coordinates and then start the dlib correlation
				            # tracker
                            tracker = dlib.correlation_tracker()
                            rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
                            tracker.start_track(img, rect)

                            # add the tracker to our list of trackers so we can
				            # utilize it during skip frames
                            trackers.append(tracker)

                print(lab,": ",count)
        
        else:
            # TRACKING
            skiped_cnt += 1
            for tracker in trackers:
                tracker.update(img)
                pos = tracker.get_position()

			    # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                # add the bounding box coordinates to the rectangles list
                rects.append((startX, startY, endX, endY))


                #tracker.update(img)
                rect = tracker.get_position()
                pt1 = (int(rect.left()), int(rect.top()))
                pt2 = (int(rect.right()), int(rect.bottom()))
                img = cv2.rectangle(img, pt1, pt2, (255, 0, 0), 2)


        # use the centroid tracker to associate the (1) old object
	    # centroids with (2) the newly computed object centroids
        objects = ct.update(rects)


        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # check to see if a trackable object exists for the current
            # object ID
            to = trackableObjects.get(objectID, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)

            # otherwise, there is a trackable object so we can utilize it
            # to determine direction
            else:
                # the difference between the y-coordinate of the *current*
                # centroid and the mean of *previous* centroids will tell
                # us in which direction the object is moving (negative for
                # 'up' and positive for 'down')
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                # check to see if the object has been counted or not
                if not to.counted:
                    # if the direction is negative (indicating the object
                    # is moving up) AND the centroid is above the center
                    # line, count the object
                    #if direction < 0 and centroid[1] < H // 2:
                    #	totalUp += 1
                    #	to.counted = True

                    # if the direction is positive (indicating the object
                    # is moving down) AND the centroid is below the
                    # center line, count the object
                    #elif direction > 0 and centroid[1] > H // 2:
                    #	totalDown += 1
                    #	to.counted = True
                    total += 1
                    to.counted = True

            # store the trackable object in our dictionary
            trackableObjects[objectID] = to

            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(img, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(img, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            total_str = "Total counted: " + str(total)
            cv2.putText(img, total_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


        cv2.imshow("image",img)  
        key = cv2.waitKey(1) & 0xFF
        if key  == ord('q'):
            break          
        elif key == ord('p'):
            cv2.waitKey(0) # PAUSE (Enter any key to continue)




cap.release()
cv2.destroyAllWindows()    
