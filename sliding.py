import cv2
import math
import os
import tensorflow as tf
import random
import numpy as np
import config
import matplotlib.pyplot as plt


def create_windows(height, width, plot_fig=False):
    np.random.seed(100)
    windows = np.random.randint(256, size=(height, width, 3)).astype(np.uint8)
    if plot_fig:
        plt.imshow(windows)
        plt.show()
    return windows


def cal_wind(imgshape, config, output_last = True):
    h_wind = math.ceil(imgshape[0] / config.h_sample_rate)
    w_wind = math.ceil(imgshape[1] / config.w_sample_rate)
    if not output_last:
        return h_wind, w_wind
    else:
        h_wind_last = imgshape[0] - h_wind * (config.h_sample_rate - 1)
        w_wind_last = imgshape[1] - w_wind * (config.w_sample_rate - 1)
        return h_wind, w_wind, h_wind_last, w_wind_last


def cal_location(h_wind, w_wind, i, j, imgshape, config):
    windows_normal = True
    bgn = np.array([h_wind * i, w_wind * j])
    fnl = bgn + np.array([h_wind, w_wind])
    if i == config.h_sample_rate - 1:
        fnl[0] = imgshape[0]
        windows_normal = False
    if j == config.w_sample_rate - 1:
        fnl[1] = imgshape[1]
        windows_normal = False
    return bgn, fnl, windows_normal


if __name__ == '__main__':
    testpath = './sample_images/dog3.jpeg'
    img_origin = cv2.imread(testpath)
    # img_origin = cv2.resize(img_origin, (img_origin.shape[0]//4, img_origin.shape[1]//4))
    height = 299
    img_origin = cv2.resize(img_origin, (int(img_origin.shape[1] * (height / img_origin.shape[0])), height))
    plot_blur = False
    save_img = True
    # window initialization
    imgshape = img_origin.shape
    h_wind, w_wind, h_wind_last, w_wind_last = cal_wind(imgshape, config)
    windows = create_windows(h_wind, w_wind)

    # windows_normal = True
    ind = 0
    for i in range(config.h_sample_rate):
        for j in range(config.w_sample_rate):
            img_tmp = img_origin.copy()
            bgn, fnl, windows_normal = cal_location(h_wind, w_wind, i, j, imgshape, config)
            if windows_normal:
                img_tmp[bgn[0]: fnl[0], bgn[1]: fnl[1]] = windows
            else:
                img_tmp[bgn[0]: fnl[0], bgn[1]: fnl[1]] = windows[0: fnl[0] - bgn[0], 0: fnl[1] - bgn[1]]
            if plot_blur:
                plt.imshow(img_tmp[:, :, [2, 1, 0]])
                plt.show()
            if save_img:
                fname = os.path.join(config.blur_img_dir, '%05d' % ind + '.png')
                cv2.imwrite(fname, img_tmp)
                ind += 1
                if ind % config.save_print == 0:
                    print(f"{ind} images saving complete!")





