import cv2
import os
import tensorflow as tf
import tensorflow.contrib.slim.nets
import config
import numpy as np
slim = tf.contrib.slim
resnet_v2 = slim.nets.resnet_v2
resnet_utils = slim.nets.resnet_utils

inception = slim.nets.inception

g_resnet = tf.Graph()
g_inception = tf.Graph()

with g_resnet.as_default():
    input = tf.placeholder(tf.float32, shape=(None, 299, 299, 3), name='input')
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
        net, endpoints = resnet_v2.resnet_v2_50(input, num_classes=1001)

# with g_inception.as_default():
#     input = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='input')
#     with slim.arg_scope(inception.inception_v1_arg_scope()):
#         net, endpoints = inception.inception_v1(input, num_classes=1001)

model_path_inception = './checkpoints/inception_v1.ckpt'
model_path_resnet = './checkpoints/resnet_v2_50.ckpt'

img = cv2.imread('./sample_images/dog.jpeg')
img_input = cv2.resize(img, (299, 299)).astype("float")

img_input = img_input.reshape([1, 299, 299, 3])

with tf.Session(graph=g_resnet) as sess:
    restorer = tf.train.Saver([i for i in tf.trainable_variables()])
    restorer.restore(sess, model_path_resnet)
    a = sess.run([net], feed_dict={input: img_input})
    print(a)
    




