import cv2
from typing import List
from pathlib import Path
import os

front_rgb_images_path = Path("/home/michael/Desktop/projects/ROAR/opencv_object_tracking/data/front_rgb")
paths: List[Path] = sorted(Path(front_rgb_images_path).iterdir(), key=os.path.getmtime)

folder_name = "ssd_mobilenet_v3_large_coco_2020_01_14"
print(f"Using {folder_name}")

tensorflowNet = cv2.dnn.readNetFromTensorflow(
    'opencv_weights_and_config/' + folder_name + '/frozen_inference_graph.pb',
    'opencv_weights_and_config/' + folder_name + '/model_structure.pbtxt')

# tensorflowNet = cv2.dnn.readNetFromTensorflow(
#     "opencv_weights_and_config/faster",
#     'faster_rcnn_inception_v2/faster_rcnn_inception_v2_coco_2018_01_28.pbtxt'
# )
tensorflowNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
tensorflowNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

for img_path in paths[120:]:
    image = cv2.imread(img_path.as_posix())

    rows, cols, channels = image.shape

    # Use the given image as input, which needs to be blob(s).
    tensorflowNet.setInput(cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False))

    # Runs a forward pass to compute the net output
    networkOutput = tensorflowNet.forward()

    # Loop on the outputs
    for detection in networkOutput[0, 0]:

      score = float(detection[2])
      if score > 0.5:
        left = detection[3] * cols
        top = detection[4] * rows
        right = detection[5] * cols
        bottom = detection[6] * rows
        area = (right - left) * (bottom - top)

        # draw a red rectangle around detected objects
        if area < 10000:
            cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)
        # cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)
    # Show the image with a rectagle surrounding the detected objects
    cv2.imshow('Image', image)

    key = cv2.waitKey(2) & 0xFF
    if key == ord("s"):
      pass

    elif key == ord("q"):
      break

