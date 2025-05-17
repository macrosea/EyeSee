#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys

sys.path.append("..")

#
# from common.file_utils import create_dir
#
# import os
# import shutil
# import logging
import time

# from search.feats_store import ImgFeatsStore
from common.config import IMG_FEATS_STORE_NAME, IMG_STORE
from search.search import Search
# from img_processor.feature_match import FeatureMatch
from tests.helper import (DIR_CROPPED, init_env, save_rect_crop,
                          search_compose_detect, wm_crop)
from wm_client.wm_client import WmClient


def test_single_img():
    init_env()
    #img_path = "/tmp/tmp/detected-error/embeded_ia_1300000183.jpg"
    #
    img_path = '/tmp/ws//work/watermark/img_store/wm-0902/ia_2000000265.jpg'
    img_path = '/tmp/ws//work/watermark/img_store/wm-0902/ia_1000000115.jpg'
    #img_path = "/tmp/tmp/wm/photo_0803/ia_v_0803_400000133.jpg"
    img_path = "/tmp/ws/work/watermark/img_store/wm-0902/ia_1100000203.jpg"

    fn = "ia_1400000279.jpg"
    fn = "ia_200000187.jpg"
    fn = "ia_1600000195.jpg"
    fn = "ia_700000215.jpg"
    fn = "ia_2200000217.jpg"
    img_path = "/tmp/ws/work/watermark/img_store/wm-0902/" + fn
    img_path = "/tmp/tmp/ia_down_img/ia_v_080305_100001784.jpeg"

    f_cropped = wm_crop(img_path)
    print('[test_single_img] : ' + f_cropped)

    try:
        client = WmClient()
        searcher = Search(IMG_FEATS_STORE_NAME)
        start = time.time()
        res = search_compose_detect(f_cropped, searcher, client)
        end = time.time()
        print("res={}; end - start={}".format(res, end - start))
    except Exception as err:
        print(err)







if __name__ == '__main__':
    test_single_img()
