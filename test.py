import tensorflow as tf
import numpy as np
import os
import cv2
import os
import tensorflow as tf
import tensorflow.contrib.slim.nets
import config
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import preprocess
slim = tf.contrib.slim
resnet_v2 = slim.nets.resnet_v2
resnet_utils = slim.nets.resnet_utils




imagedir = './blur_img'
imglist = [os.path.join(imagedir, fname) for fname in os.listdir(imagedir) if ".DS" not in fname]
imglist.sort()
labellist = []
for imgpath in imglist:
    label = int(imgpath.split('/')[-1].split('.')[0])
    labellist.append(label)
tot_num = imglist.__len__()

images_list_tensor = tf.convert_to_tensor(imglist)
label_list_tensor = tf.convert_to_tensor(labellist)
input_queue_val = tf.train.slice_input_producer([images_list_tensor, label_list_tensor], shuffle=False)
image_fname = tf.read_file(input_queue_val[0])
# fname_str = tf.decode_raw(image_fname)
image_val = tf.image.decode_png(image_fname, channels=3)

preprocessed_image_val = preprocess(image_val)

images_val, label = tf.train.batch([preprocessed_image_val, input_queue_val[1]], capacity=10000, batch_size=1, num_threads=10, allow_smaller_final_batch=True)



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

save_array = np.zeros((config.h_sample_rate, config.w_sample_rate))



with tf.Session() as sess:
    restorer = tf.train.Saver([i for i in tf.trainable_variables()])
    restorer.restore(sess, model_path_resnet)
    print("restoring complete!")

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    i = 0

    try:
        while not coord.should_stop():
            #
            # imgname = sess.run(input_queue_val[0])
            # fname = imgname.decode("utf-8")
            # print(fname)
            # img, imglabel = sess.run([images_val, label])
            # print(imglabel)
            # plt.imshow(img[0])
            # raw_code = sess.run(imgname)
            # fname = raw_code[0].decode("utf-8")
            # index = int(fname.split('/')[-1].split('.')[0])
            # print(index)
            # print(fname)
            # # print(fname[0])
            # # print(img.shape)
            # plt.show()
            a, index = sess.run([endpoints['resnet_v2_50/block4'][0,0,0,:], label])
            index = index[0]
            save_array[index//config.w_sample_rate, index % config.h_sample_rate] = a[464]
            # for j in range(10):
            #     print(image_batch_v.shape, label_batch_v[j])
            if i % 10 == 0:
                print(i)
            i += 1
            if i >= tot_num:
                coord.request_stop()



    except tf.errors.OutOfRangeError:
        print("done")
    finally:
        coord.request_stop()
    coord.join(threads)

np.save("save_array.npy", save_array)
