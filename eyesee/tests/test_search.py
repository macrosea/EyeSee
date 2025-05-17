#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys

sys.path.append("..")

import h5py
from common.config import IMG_FEATS_STORE_NAME
from search.feats_store import ImgFeatsStore
from search.search import Search

IMG_FEATS_STORE_NAME = "/tmp/feature.h5"

IMG_STORE = "/tmp/ws/work/watermark/img_store/wm-0902"
#img_store = "/tmp/test_img"

def save_single_feats():
    img2 = "/tmp/tmp/images/500px1021494767.jpg"
    #img2 = "/tmp/Downloads/images/1599043401821.jpg"
    store = ImgFeatsStore(IMG_FEATS_STORE_NAME)
    # store.show_shape("feat_sets")
    # store.show_shape("name_sets")
    store.save_img_feat(img2)
    store.show_shape("feat_sets")
    store.show_shape("name_sets")


def batch_save():
    #img_dir = "/tmp/ws/work/watermark/sets_verify"
    img_dir = IMG_STORE
    store = ImgFeatsStore(IMG_FEATS_STORE_NAME)
    store.batch_save_img_feat(img_dir)
    store.show_shape("feat_sets")
    store.show_shape("name_sets")


def search():
    #batch_save()
    searcher = Search(IMG_FEATS_STORE_NAME)
    res = searcher.search('/tmp/wm_test/ia_100000279.jpg')
    print(res)


def test_hf5():
    #file_name = "/tmp/feature.h5"
    with h5py.File(IMG_FEATS_STORE_NAME, 'r') as h5f:
        set_feats = h5f['feat_sets']
        set_names = h5f['name_sets']
        print(set_feats[0])
        print(set_names[:])
        for idx, name in enumerate(set_names[:]):
            if str(name, encoding='utf-8').find('ia_100000279.jpg') != -1:
                print("yes")
                break


if __name__ == '__main__':
    #batch_save()
    save_single_feats()
    #test_hf5()
    #search()
