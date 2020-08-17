from pathlib import Path
import cv2

efficient_det_path = Path("efficientdet-d0/efficientdet-d0.pb")
efficient_det_txt_path = Path("efficientdet-d0/efficientdet-d0.pbtx")
cvNet = cv2.dnn.readNetFromTensorflow(efficient_det_path.as_posix(), efficient_det_txt_path.as_posix())

# Input image
img = cv2.imread('images/dog.jpeg')
rows = img.shape[0]
cols = img.shape[1]
cvNet.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
cvOut = cvNet.forward()

for detection in cvOut[0,0,:,:]:
    score = float(detection[2])
    if score > 0.3:
        left = detection[3] * cols
        top = detection[4] * rows
        right = detection[5] * cols
        bottom = detection[6] * rows
        cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)

cv2.imshow('img', img)
cv2.waitKey()