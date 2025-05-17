# -*- coding: utf-8 -*-

# @Time    : 2019/10/05
# @Author  : macrosea

import time
from os.path import join

import h5py
import numpy as np
#from search.vgg16_keras import VGGNet
from search.xception import Xception as VGGNet


class Search:
    __max_res = 6

    def __init__(self, feat_store="/tmp/feature.h5"):
        self.model = VGGNet()
        # read in indexed images' feature vectors and corresponding image names
        h5f = h5py.File(feat_store, 'r')
        self.feats = h5f['feat_sets'][:]
        self.img_name_sets = h5f['name_sets'][:]
        h5f.close()

    def search(self, query_img_path):
        # extract query image's feature, compute similarity score and sort
        start = time.time()
        query_vec = self.model.extract_feat(query_img_path)
        scores = np.dot(query_vec, self.feats.T)
        rank_id = np.argsort(scores)[::-1]
        rank_score = scores[rank_id]
        end = time.time()
        #print(end - start)  # .seconds
        imlist = [self.img_name_sets[index] for i, index in enumerate(rank_id[0:Search.__max_res])]
        #print("top %d images in order are: " % Search.__max_res, imlist)
        return str(imlist[0], encoding='utf-8')

    def search_topk(self, query_img_path, topk):
        query_vec = self.model.extract_feat(query_img_path)
        scores = np.dot(query_vec, self.feats.T)
        rank_id = np.argsort(scores)[::-1]
        #rank_score = scores[rank_id]
        imlist = [str(self.img_name_sets[index], encoding='utf-8') for i, index in enumerate(rank_id[0:topk])]
        return imlist
