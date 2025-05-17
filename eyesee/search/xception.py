# -*- coding: utf-8 -*-

# @Time    : 2019/10/05
# @Author  : macrosea

import os

import keras.backend.tensorflow_backend as KTF
import numpy as np
import tensorflow as tf
from keras.applications.xception import Xception as KerasXception
from keras.applications.xception import \
    preprocess_input as preprocess_input_xception
from keras.preprocessing import image
from numpy import linalg as LA

# LOCAL_TMP_PATH = os.getenv("UPLOAD_FOLDER", "/tmp/")  Xception


class Xception:
    def __init__(self):
        self.input_shape = (299, 299, 3)
        self.weight = 'imagenet'
        self.pooling = 'avg'
        self.load_config()

    def load_config(self):
        # read model config from environment
        self.device_str = os.environ.get("device_id", "/cpu:0")
        self.user_config = tf.ConfigProto(allow_soft_placement=False)
        gpu_mem_limit = float(os.environ.get("gpu_mem_limit", 0.3))
        self.user_config.gpu_options.per_process_gpu_memory_fraction = gpu_mem_limit
        self.user_config.gpu_options.allow_growth = True
        if os.environ.get("log_device_placement", False):
            self.user_config.log_device_placement = True
        # print("device id %s, gpu memory limit: %f" %
        #       (self.device_str, gpu_mem_limit))

        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.device(self.device_str):
                self.session = tf.Session(config=self.user_config, graph=self.graph)
                KTF.set_session(self.session)
                self.model = KerasXception(weights=self.weight,
                                           input_shape=(self.input_shape[0],
                                                        self.input_shape[1],
                                                        self.input_shape[2]),
                                           pooling=self.pooling,
                                           include_top=False)
                self.graph = KTF.get_graph()
                self.session = KTF.get_session()
                self.model.trainable = False
                self.model.predict(np.zeros(
                    (1, self.input_shape[0], self.input_shape[1], 3)))
                self.graph.as_default()
                self.session.as_default()

    def extract_feat(self, img_path):
        img = image.load_img(
            img_path,
            target_size=(
                self.input_shape[0],
                self.input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input_xception(img)
        with self.graph.as_default():
            with self.session.as_default():
                # with tf.device(self.device_str):
                feat = self.model.predict(img)

        norm_feat = feat[0] / LA.norm(feat[0])
        norm_feat = [i.item() for i in norm_feat]
        return norm_feat
