import cv2
import sys
import os
from pathlib import Path
from typing import List
front_rgb_images_path = Path("/home/michael/Desktop/projects/ROAR/opencv_object_tracking/data/front_rgb")
paths: List[Path] = sorted(Path(front_rgb_images_path).iterdir(), key=os.path.getmtime)

# tracker_used = "goturn"
# tracker_used = "csrt"
# tracker_used = "mil"
# tracker_used = "tld"
# tracker_used = "medianflow"
# tracker_used = "mosse"
tracker_used = "kcf"
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create,
    "goturn": cv2.TrackerGOTURN_create
}

# grab the appropriate object tracker using our dictionary of
# OpenCV object tracker objects
trackers = cv2.MultiTracker_create()

# initialize the bounding box coordinates of the object we are going
# to track
initBB = None

for img_path in paths:
    frame = cv2.imread(img_path.as_posix())
    timer = cv2.getTickCount()
    (H, W) = frame.shape[:2]

    (success, boxes) = trackers.update(frame)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    print(success)
    if success:
        for box in boxes:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # initialize the set of information we'll be displaying on
        # the frame
        info = [
            ("Tracker", tracker_used),
            ("Success", "Yes" if success else "No"),
            ("FPS", "{:.2f}".format(fps))
        ]
        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(10) & 0xFF

    if key == ord("s"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        box = cv2.selectROI("Frame", frame, fromCenter=False,
                            showCrosshair=True)
        # start OpenCV object tracker using the supplied bounding box
        # coordinates, then start the FPS throughput estimator as well

        tracker = OPENCV_OBJECT_TRACKERS[tracker_used]()
        trackers.add(tracker, frame, box)
    elif key == ord("q"):
        break

# close all windows
cv2.destroyAllWindows()
