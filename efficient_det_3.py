import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


import cv2

# Load a model imported from Tensorflow
tensorflowNet = cv2.dnn.readNetFromTensorflow('faster_rcnn_inception_v2/faster_rcnn_inception_v2_coco_2018_01_28.pb',
                                              'faster_rcnn_inception_v2/faster_rcnn_inception_v2_coco_2018_01_28.pbtxt')
tensorflowNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
tensorflowNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
# Input image
img = cv2.imread('data/front_rgb/frame_210.png')
rows, cols, channels = img.shape

# Use the given image as input, which needs to be blob(s).
tensorflowNet.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))

# Runs a forward pass to compute the net output
networkOutput = tensorflowNet.forward()

# Loop on the outputs
for detection in networkOutput[0,0]:

    score = float(detection[2])
    if score > 0.2:

        left = detection[3] * cols
        top = detection[4] * rows
        right = detection[5] * cols
        bottom = detection[6] * rows

        #draw a red rectangle around detected objects
        cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)

# Show the image with a rectagle surrounding the detected objects
cv2.imshow('Image', img)
cv2.waitKey()
cv2.destroyAllWindows()