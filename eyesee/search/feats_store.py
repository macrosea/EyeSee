# -*- coding: utf-8 -*-

# @Time    : 2019/10/05
# @Author  : macrosea


import os

import h5py
import numpy as np
#from search.vgg16_keras import VGGNet
from search.xception import Xception as VGGNet


class HDF5Store:
    def __init__(self, file_name):
        self.file_name = file_name
        pass

    def save(self, data_set_name, feats):
        if len(feats.shape) == 1:
            max_shape = (None,)
        else:
            feats_shape = list(feats.shape)
            feats_shape[0] = None
            max_shape = tuple(feats_shape)

        #print("max_shape: {}".format(max_shape))

        with h5py.File(self.file_name, 'a') as h5f:
            if not h5f.__contains__(data_set_name):
                h5f.create_dataset(data_set_name,  data=feats, chunks=True, maxshape=max_shape)
            else:
                h5f[data_set_name].resize((h5f[data_set_name].shape[0] + feats.shape[0]), axis=0)
                h5f[data_set_name][-feats.shape[0]:] = feats
                h5f.flush()

    def show_shape(self, sets_name):
        with h5py.File(self.file_name, 'r') as h5f:
            print(h5f[sets_name].shape)


class ImgFeatsStore(HDF5Store):
    def __init__(self, file_name="/tmp/feature.h5", feats_set_name='feat_sets', path_set_name='name_sets'):
        super().__init__(file_name)
        self.model = VGGNet()
        self.feats_set_name = feats_set_name
        self.path_set_name = path_set_name

    def save_img_feat(self, img_path):
        norm_feat = self.model.extract_feat(img_path)
        img_name = os.path.split(img_path)[1]
        feats = np.array([norm_feat])
        names = np.array([img_path.encode()])
        # print(feats.shape)
        # print(names.shape)
        self.save(self.feats_set_name, feats)
        self.save(self.path_set_name, names)

    def batch_save_img_feat(self, img_dir):
        img_list = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')]
        feats = []
        names = []
        for i, img_path in enumerate(img_list):
            try:
                norm_feat = self.model.extract_feat(img_path)
                img_name = os.path.split(img_path)[1]
                feats.append(norm_feat)
                names.append(img_path.encode())
            except Exception as e:
                print("Couldn't handle {}".format(img_path))
                print('Reason:', e)

            print("extracting feature from image No. %d , %d images in total" % ((i + 1), len(img_list)))

        feats = np.array(feats)
        names = np.array(names)
        # print(feats.shape)
        # print(names.shape)
        self.save(self.feats_set_name, feats)
        self.save(self.path_set_name, names)
