import tensorflow as tf
import numpy as np
import os
import cv2
import os
import tensorflow as tf
import tensorflow.contrib.slim.nets
import config
import numpy as np
from preprocessing import preprocess
slim = tf.contrib.slim
resnet_v2 = slim.nets.resnet_v2
resnet_utils = slim.nets.resnet_utils




imagedir = './sample_images'
imglist = [os.path.join(imagedir, fname) for fname in os.listdir(imagedir) if ".DS" not in fname]

tot_num = imglist.__len__()

images_val = tf.convert_to_tensor(imglist)
input_queue_val = tf.train.slice_input_producer([images_val])
image_val = tf.read_file(input_queue_val[0])
image_val = tf.image.decode_jpeg(image_val, channels=3)

preprocessed_image_val = preprocess(image_val)

images_val = tf.train.batch([preprocessed_image_val], batch_size=1, allow_smaller_final_batch=True)



# g_resnet = tf.Graph()

# with g_resnet.as_default():
with slim.arg_scope(resnet_utils.resnet_arg_scope()):
    net, endpoints = resnet_v2.resnet_v2_50(images_val, num_classes=1001)

# with g_resnet.as_default():
#     input = tf.placeholder(tf.float32, shape=(None, 299, 299, 3), name='input')
#     with slim.arg_scope(resnet_utils.resnet_arg_scope()):
#         net, endpoints = resnet_v2.resnet_v2_50(input, num_classes=1001)
#
model_path_resnet = './checkpoints/resnet_v2_50.ckpt'


with tf.Session() as sess:
    restorer = tf.train.Saver([i for i in tf.trainable_variables()])
    restorer.restore(sess, model_path_resnet)
    print("restoring complete!")

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    i = 0
    try:
        while not coord.should_stop():
            a = sess.run([net, endpoints['resnet_v2_50/block4'][0,0,0,:]])
            i += 1
            print(a)
            # for j in range(10):
            #     print(image_batch_v.shape, label_batch_v[j])
            if i >= tot_num:
                coord.request_stop()

    except tf.errors.OutOfRangeError:
        print("done")
    finally:
        coord.request_stop()
    coord.join(threads)

