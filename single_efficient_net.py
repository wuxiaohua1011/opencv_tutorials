from tensorflow import keras

from pathlib import Path
import os
import sys
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import efficientnet.tfkeras
from tensorflow.keras.models import load_model
from keras.applications.imagenet_utils import decode_predictions
import efficientnet
model = efficientnet.tfkeras.EfficientNetB7(weights='imagenet')

# image = imread(Path("./images/dog.jpeg"))
image = imread(Path("./data/front_rgb/frame_201.png"))
# plt.figure(figsize=(10, 10))
# plt.imshow(image)
# plt.show()

image_size = model.input_shape[1]

x = efficientnet.tfkeras.center_crop_and_resize(image, image_size=image_size)
x = efficientnet.tfkeras.preprocess_input(x)
x = np.expand_dims(x, 0)

# make prediction and decode
y = model.predict(x)
result = decode_predictions(y)
print(result)


