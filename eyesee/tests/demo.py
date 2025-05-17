#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
sys.path.append("..")

from search.feats_store import ImgFeatsStore
from common.config import IMG_FEATS_STORE_NAME, IMG_STORE
from search.search import Search
from wm_client.wm_client import WmClient
from img_processor.feature_match import FeatureMatch
from tests.helper import save_rect_crop, wm_crop, search_compose_detect, init_env, DIR_CROPPED


from common.file_utils import create_dir

import os
import shutil
import logging


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='/tmp/watermark_testing.log', level=logging.INFO, format=LOG_FORMAT)


img_list = [os.path.join(IMG_STORE, f) for f in os.listdir(IMG_STORE) if f.endswith('.jpg')]


def verify_no_wm():
    client = WmClient()
    img_count = len(img_list)
    detected_count = 0
    right_count = 0
    for idx, itm in enumerate(img_list):
        print(itm)
        res = client.detect(itm)
        if int(res[0]) == 200:
            detected_count += 1
            right = 1 if int(res[1]) == 0 else 0
            right_count += right
        print(idx, right_count, img_count)
    print("img_count: {}, detected_count: {}, right_count: {}, acc: {}".format(img_count, detected_count, right_count,  float(right_count)/detected_count ))


def verify_wm():
    dir_embed = "/tmp/embedded"
    create_dir(dir_embed)
    client = WmClient()
    img_count = len(img_list)
    detected_count = 0
    right_count = 0
    for idx, itm in enumerate(img_list):
        #print(itm)
        embedded = client.embed(itm, dir_embed)
        if not embedded:
            print("failed to embed {}".format(itm))
            continue
        #print(embedded)
        res = client.detect(embedded)
        if int(res[0]) == 200:
            detected_count += 1
            if int(res[1]) == 1:
                right_count += 1
            else:
                print(itm, embedded)
                base_name = os.path.basename(embedded)
                shutil.copyfile(embedded, os.path.join("/tmp/tmp/detected-error/", base_name))
        else:
            print(embedded, res)

        print(idx+1, right_count, img_count)
    print("img_count: {}, detected_count: {}, right_count: {}, acc: {}".format(img_count, detected_count, right_count,  float(right_count)/detected_count ))



def verify_crop_wm():
    dir_embed = "/tmp/wm/embedded"
    dir_cropped = "/tmp/wm/cropped"
    dir_compose = "/tmp/wm/compose"

    dir_embed_failed = "/tmp/wm/embed_failed"
    dir_detect_failed = "/tmp/wm/detect_failed"

    dir_except = "/tmp/wm/except"

    create_dir(dir_embed)
    create_dir(dir_cropped)
    create_dir(dir_compose)
    create_dir(dir_embed_failed)
    create_dir(dir_detect_failed)
    create_dir(dir_except)

    client = WmClient()
    img_count = len(img_list)
    detected_count = 0
    right_count = 0
    for idx, itm in enumerate(img_list):
        print("handling " + itm)
        base_name = os.path.basename(itm)

        embedded = client.embed(itm, dir_embed)
        if not embedded:
            print("failed to embed {}".format(itm))
            shutil.copyfile(itm, os.path.join(dir_embed_failed, base_name))
            continue

        cropped = save_rect_crop(embedded, dir_cropped)
        searcher = Search(IMG_FEATS_STORE_NAME)
        res = searcher.search(cropped)
        try:
            fm = FeatureMatch()
            compose_path = os.path.join(dir_compose, base_name)
            fm.compose(res, cropped, compose_path)

            res = client.detect(compose_path)
            if int(res[0]) == 200:
                detected_count += 1
                if int(res[1]) == 1:
                    right_count += 1
                else:
                    print("detect wrong {}".format(itm, compose_path))
                    shutil.copyfile(itm, os.path.join(dir_detect_failed, base_name))
            else:
                print("detect failed st!=200: {}".format(embedded, res))
        except Exception as e:
            print(e)
            shutil.copyfile(itm, os.path.join(dir_except, base_name))

        print(idx+1, right_count, img_count)
    print("img_count: {}, detected_count: {}, right_count: {}, acc: {}".format(img_count, detected_count, right_count,  float(right_count)/detected_count ))


# def test_few_imgs():
#     IMG_STORE = "/tmp/tmp/detected-error"
#     imgs = [os.path.join(IMG_STORE, f) for f in os.listdir(IMG_STORE) if f.endswith('.jpg')]
#     init_env()
#     for idx, itm in enumerate(imgs):
#         print("============== handling " + itm)
#         f_cropped = wm_crop(itm)
#         res = search_compose_detect(f_cropped)
#         if res != 1:
#             print(itm + "!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#         else:
#             print(itm + " @@@@@@@@@@@@@@@@ ")

def test_single_img():
    init_env()
    #img_path = "/tmp/tmp/detected-error/embeded_ia_1300000183.jpg"
    #
    img_path = '/tmp/ws//work/watermark/img_store/wm-0902/ia_2000000265.jpg'
    img_path = '/tmp/ws//work/watermark/img_store/wm-0902/ia_1000000115.jpg'
    f_cropped = wm_crop(img_path)
    print('[test_single_img] : ' + f_cropped)
    res = search_compose_detect(f_cropped)
    if res != 1:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")


def batch_imgs_prepare(dir_imgs):
    img_paths = [os.path.join(dir_imgs, f) for f in os.listdir(dir_imgs) if f.endswith('.jpg')]
    total = len(img_paths)
    except_count = 0
    for idx, itm in enumerate(img_paths):
        try:
            logging.info("batch_imgs_prepare: {} -/- {}; handling {}".format(idx+1, total, itm))
            wm_crop(itm)
        except Exception as e:
            except_count += 1
            logging.error("[batch_imgs_prepare] {}\n{}".format(itm, e))
    logging.info("batch_imgs_prepare finished; total={}, except_count={}".format(total, except_count))
    logging.info("\n\n{}\n\n".format("="*30))


def batch_imgs_detect(dir_imgs, expect):
    img_paths = [os.path.join(dir_imgs, f) for f in os.listdir(dir_imgs) if f.endswith('.jpg')]
    total = len(img_paths)
    right_count = 0
    error_count = 0
    except_count = 0
    client = WmClient()
    searcher = Search(IMG_FEATS_STORE_NAME)
    for idx, itm in enumerate(img_paths):
        try:
            logging.info("batch_imgs_detect: {} -/- {}; handling {}".format(idx+1, total, itm))
            res = search_compose_detect(itm, searcher, client)
            if res == expect:
                right_count += 1
            else:
                error_count += 1
                logging.error('[batch_imgs_detect.expect:{}] error; file: {}'.format(expect, itm))
        except Exception as e:
            except_count += 1
            logging.error("[batch_imgs_detect.expect:{}] Exception:\n file: {}\n{}".format(expect, itm, e))
    logging.info("batch_imgs_detect finished; total={}, right_count={}, error_count={}, except_count={}".format(total, right_count, error_count, except_count))


def test_batch_imgs():
    init_env()
    dir_imgs = "/tmp/ws/work/watermark/img_store/wm-0902"
    batch_imgs_prepare(dir_imgs)
    batch_imgs_detect(DIR_CROPPED, 1)

if __name__ == '__main__':
    test_batch_imgs()
    #test_single_img()
