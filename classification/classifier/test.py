import glob
import os
from os import path

import cv2
import numpy as np
import tensorflow as tf
from skimage import io, transform

w = 32
h = 32
c = 1

img_dir = "/Users/macrosea/ws/work/watermark/classifier/tf4wm"
train_path = path.join(img_dir, "train")
# train_path = "/Users/macrosea/ws/work/watermark/classifier/images/old/"
# test_path = "mnist/test/"


def read_image(img_dir):
    label_dir = [path.join(img_dir, x) for x in os.listdir(img_dir) if os.path.isdir(path.join(img_dir, x))]#子檔案目錄
    images = []
    labels = []
    for index,folder in enumerate(label_dir):
        for img in glob.glob(folder+'/*.bmp'):
            print("reading the image:%s"%img)
            image = cv2.imread(img, 0)
            image = cv2.resize(image, (28, 28))
            # image = io.imread(img)
            # image = transform.resize(image,(w,h,c))
            images.append(image)
            labels.append(index)
    return np.asarray(images, dtype = np.float32), np.asarray(labels, dtype=np.int32)



def load_preprosess_image(input_path):
    image = tf.io.read_file(input_path)
    image = tf.image.decode_jpeg(image,channels=3)
    image = tf.image.resize(image,[256,256])
    image = tf.cast(image,tf.float32)
    image = image/255

    return image ,label # return回的都是一个batch一个batch的 ， 一个批次很多张


######


# train_data, train_label = read_image(train_path)
# #test_data,test_label = read_image(test_path)
# print(train_data.shape)

load_preprosess_image("/Users/macrosea/ws/work/watermark/classifier/images/wm/test_wm_03.png")
