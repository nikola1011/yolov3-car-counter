# TODO: Refactor all this Nikola .. Make this code pretty 

from tracking.centroidtracker import CentroidTracker
from tracking.trackableobject import TrackableObject
import tensornets as nets
import cv2
import numpy as np
import time
import dlib
import tensorflow.compat.v1 as tf
import os

# For 'disable_v2_behavior' see https://github.com/theislab/scgen/issues/14
tf.disable_v2_behavior() 

# Image size must be '416x416' as YoloV3 network expects that specific image size as input
img_size = 416
inputs = tf.placeholder(tf.float32, [None, img_size, img_size, 3])
model = nets.YOLOv3COCO(inputs, nets.Darknet19)
#TODO: [nikola1011] Try these models out, or delete them
#model = nets.YOLOv2(inputs, nets.Darknet19)
#model = nets.TinyYOLOv2VOC(inputs, nets.Darknet19)

ct = CentroidTracker(maxDisappeared=5, maxDistance=50) # Look into 'CentroidTracker' for further info about parameters
trackers = [] # List of all dlib trackers
trackableObjects = {} # TODO: [nikola1011] Explain this variable 
skip_frames = 10 # Numbers of frames to skip from detecting
confidence_level = 0.40 # The confidence level of a detection
total = 0 # Total number of detected objects from classes of interest

#TODO: [nikola1011] Add some video to local 'videos' folder and upload to GitHub. Trimm to some smaller size.
#video_path = "/home/nikola/Videos/Relaxing highway traffic.mp4"
#video_path = "/home/nikola/Videos/Road traffic video for object detection and tracking.mp4"
#video_path = "/home/nikola/Videos/UK Motorway M25 Trucks, Lorries, Cars Highway.mp4"
video_path = "/home/nikola/Videos/M6 Motorway Traffic.mp4"
video_name = os.path.basename(video_path)

# From https://github.com/experiencor/keras-yolo3/blob/master/yolo3_one_file_to_detect_them_all.py#L389
# YoloV3 detects 80 classes represented below
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", \
              "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", \
              "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", \
              "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", \
              "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", \
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", \
              "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", \
              "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", \
              "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", \
              "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# Classes of interest (with their corresponsind indexes for easier looping)
classes = { 1 : 'bicycle', 2 : 'car', 3 : 'motorbike', 5 : 'bus', 7 : 'truck' }

with tf.Session() as sess:
    sess.run(model.pretrained())
    cap = cv2.VideoCapture(video_path)

    # Get video size (just for log purposes)
    width =  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #TODO: [nikola1011] Check this with latest python version (also update requirements.txt)
    # Python 3.5.6 does not support f-strings (next line will generate syntax error)
    #print(f"Loaded {video_path}. Width: {width}, Height: {height}")
    print("Loaded {video_path}. Width: {width}, Height: {height}".format(video_path=video_path, width=width, height=height))
    
    skipped_frames_counter = 0

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            print("Error reading frame. cap.read() returned {ret}".format(ret))
        
        # Frame must be resized to 'img_size' (because that's what YoloV3 accepts as input)
        img = cv2.resize(frame, (img_size, img_size))
        
        tracker_rects = []

        if skipped_frames_counter == skip_frames:
            
            # Detecting happens after number of frames have passes specified by 'skip_frames' variable value
            print("[DETECTING]")
            
            trackers = []
            skipped_frames_counter = 0 # reset counter
            
            np_img = np.array(img).reshape(-1, img_size, img_size, 3)

            start_time=time.time()
            predictions = sess.run(model.preds, {inputs: model.preprocess(np_img)})
            print("Detection took %s seconds" % (time.time() - start_time)) 

            # model.get_boxes returns a 80 element array containing information about detected classes 
            # each element contains a list of detected boxes, confidence level ...
            detections = model.get_boxes(predictions, np_img.shape[1:3])
            np_detections = np.array(detections)

            # Loop only through classes we are interested in
            for class_index in classes.keys():
                local_count = 0
                #TODO: [nikola1011] Consider deleting 'label' variable # Unify 'class' and 'label' term into one
                label = classes[class_index]

                # Loop through detected infos of a class we are interested in
                for i in range(len(np_detections[class_index])):
                    box = np_detections[class_index][i] 

                    if np_detections[class_index][i][4] >= confidence_level:
                        print("Detected ", label, " with confidence of ", np_detections[class_index][i][4])

                        local_count += 1
                        startX, startY, endX, endY = box[0], box[1], box[2], box[3]
                        
                        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 1)
                        cv2.putText(img, label, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), lineType=cv2.LINE_AA)
                        
                        # Construct a dlib rectangle object from the bounding box coordinates and then start the dlib correlation
                        tracker = dlib.correlation_tracker()
                        rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
                        tracker.start_track(img, rect)

                        # Add the tracker to our list of trackers so we can utilize it during skip frames
                        trackers.append(tracker)

                # Write the total number of detected objects for a given class on this frame
                print(label," : ", local_count)
        else:
            # If detection is not happening then track previously detected objects (if any)
            print("[TRACKING]")

            skipped_frames_counter += 1 # Increase the number frames for which we did not use detection

            # Loop through tracker, update each of them and display their rectangle
            for tracker in trackers:
                tracker.update(img)
                pos = tracker.get_position()

			    # Unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                
                # Add the bounding box coordinates to the tracking rectangles list
                tracker_rects.append((startX, startY, endX, endY))
                
                # Draw tracking rectangles
                img = cv2.rectangle(img, (startX, startY), (endX, endY), (255, 0, 0), 1)


        # Use the centroid tracker to associate the (1) old object centroids with (2) the newly computed object centroids
        objects = ct.update(tracker_rects)

        # Loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # Check to see if a trackable object exists for the current object ID
            to = trackableObjects.get(objectID, None)

            if to is None:
                # If there is no existing trackable object, create one
                to = TrackableObject(objectID, centroid)
            else:
                to.centroids.append(centroid)

                # If the object has not been counted, count it and mark it as counted
                if not to.counted:
                    total += 1
                    to.counted = True

            # Store the trackable object in our dictionary
            trackableObjects[objectID] = to

            # Draw both the ID of the object and the centroid of the object on the output frame
            object_id = "ID {}".format(objectID)
            cv2.putText(img, object_id, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.circle(img, (centroid[0], centroid[1]), 2, (0, 255, 0), -1)

            # Display the total count so far
            total_str = "Total counted: " + str(total)
            cv2.putText(img, total_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Display the current frame (with all annotations drawn up to this point)
        # TODO: [nikola1011] Consider transforming detected box coordinates to draw them on original frame and not the 'resized to 416' one
        #cv2.namedWindow(video_name, cv2.WINDOW_NORMAL)
        #cv2.resizeWindow(video_name, 700, 700)
        cv2.imshow(video_name, img)  

        key = cv2.waitKey(1) & 0xFF
        if key  == ord('q'): # QUIT (exits)
            break          
        elif key == ord('p'):
            cv2.waitKey(0) # PAUSE (Enter any key to continue)

cap.release()
cv2.destroyAllWindows()
print("Exited")
