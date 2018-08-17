import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from scipy import stats

npfile = 'save_array.npy'
save_array = np.load(npfile)

box_select = 40

arr_flatten = save_array.reshape(-1)
mode_value = stats.mode(arr_flatten)

arg_sort = np.argsort(arr_flatten)
ind = arg_sort[-box_select]

thres = arr_flatten[ind]
img = cv2.imread("./sample_images/dog3.jpeg")
plt.imshow(img[:, :, [2, 1, 0]])
plt.show()

mask_tmp = np.where(save_array > thres, 255, 0)

mask = np.array(mask_tmp, dtype=np.uint8)
mask_color_tmp = mask.reshape(mask.shape[0], mask.shape[1], 1)
add_img = np.zeros((mask.shape[0], mask.shape[1], 2), dtype=np.uint8)
mask_color = np.dstack((add_img, mask_color_tmp))

mask_resize = cv2.resize(mask_color, (img.shape[1], img.shape[0]))

backtorgb = mask_resize
# backtorgb = cv2.cvtColor(mask_resize,cv2.COLOR_GRAY2RGB)
plt.imshow(backtorgb[:,:,[2,1,0]])
plt.show()

dst = cv2.addWeighted(img,0.5,backtorgb,0.5,0)

plt.imshow(dst[:,:,[2,1,0]])
plt.show()