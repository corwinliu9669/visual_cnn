import cv2
import tensorflow as tf
import random
import numpy as np
import config
import matplotlib.pyplot as plt


testpath = './sample_images/dog2.jpeg'

np.random.seed(100)
windows = np.random.randint(256, size=(config.h_wind, config.w_wind, 3)).astype(np.uint8)

plt.imshow(windows)
plt.show()

img_origin = cv2.imread(testpath)
