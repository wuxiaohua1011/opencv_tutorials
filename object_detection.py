import numpy as np
import os
from pathlib import Path
from typing import List
import cv2

front_rgb_images_path = Path("/home/michael/Desktop/projects/ROAR/opencv_object_tracking/data/front_rgb")
prototxt_path = Path("/home/michael/Desktop/projects/ROAR/opencv_object_tracking/mobile_net_deploy.prototxt")
caffe_model_path = Path("/home/michael/Desktop/projects/ROAR/opencv_object_tracking/mobilenet_iter_73000.caffemodel")
CONFIDENCE = 0.2

paths: List[Path] = sorted(Path(front_rgb_images_path).iterdir(), key=os.path.getmtime)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt_path.as_posix(), caffe_model_path.as_posix())

for img_path in paths:
    image = cv2.imread(img_path.as_posix())
    try:
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843,
                                     (300, 300), 127.5)
        # print("[INFO] computing object detections...")
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > CONFIDENCE:
                # extract the index of the class label from the `detections`,
                # then compute the (x, y)-coordinates of the bounding box for
                # the object
                idx = int(detections[0, 0, i, 1])
                # if idx != 7:
                #     print(f"{CLASSES[idx]} IS DETECTED")
                # else:
                #     print("CAR DETECTED!!!! ")
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # display the prediction
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                print("[INFO] {}".format(label))
                cv2.rectangle(image, (startX, startY), (endX, endY),
                              COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(image, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            cv2.imshow("output", image)
    except Exception as e:
        print("Error: ", e)
    key = cv2.waitKey(20) & 0xFF
    if key == ord("s"):
        pass

    elif key == ord("q"):
        break
