#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
sys.path.append("..")

from search.feats_store import ImgFeatsStore
from common.config import IMG_FEATS_STORE_NAME
from search.search import Search
from wm_client.wm_client import WmClient
from img_processor.feature_match import FeatureMatch
from common.file_utils import create_dir

import cv2
from os import path

DIR_EMBED = "/tmp/wm/embedded"
DIR_CROPPED = "/tmp/wm/cropped"
DIR_COMPOSE = "/tmp/wm/compose"


def init_env():
    create_dir(DIR_EMBED)
    create_dir(DIR_CROPPED)
    create_dir(DIR_COMPOSE)


def rect_crop(img_path):
    src = cv2.imread(img_path)  # cv2.IMREAD_GRAYSCALE)
    h, w, _ = src.shape
    x_offset = round(0.2 * w)
    y_offset = round(0.2 * h)
    x0 = x_offset
    y0 = y_offset
    x1 = w - x_offset
    y1 = h - y_offset
    cropped = src[y0: y1, x0: x1]
    return cropped


def save_rect_crop(img_path, saved_dir):
    cropped = rect_crop(img_path)
    saved_path = path.join(saved_dir, path.basename(img_path))
    cv2.imwrite(saved_path, cropped)
    return saved_path


def wm_crop(img_path):
    client = WmClient()
    embedded = client.embed(img_path, DIR_EMBED)
    cropped = save_rect_crop(embedded, DIR_CROPPED)
    return cropped


def search_compose_detect(cropped_img_path, searcher, client):
    #searcher = Search(IMG_FEATS_STORE_NAME)
    res_topk = searcher.search_topk(cropped_img_path, 20)
    # print("cropped_img_path: {}".format(cropped_img_path))
    # print("res_topk: {}".format(res_topk))
    fm = FeatureMatch()
    best_match = dict()
    min_l2 = 100000000
    cropped_img = cv2.imread(cropped_img_path, cv2.IMREAD_GRAYSCALE)
    feat_cropped = fm.detect_feature(cropped_img)
    for idx, itm in enumerate(res_topk):
        feat_target = fm.detect_feature(cv2.imread(itm, cv2.IMREAD_GRAYSCALE))
        dx, dy, l2 = fm.computer_delta(feat_target, feat_cropped)
        print(itm, dx, dy, l2)
        if dx < 0 or dy < 0:
            continue

        if l2 < min_l2:
            best_match["f_name"] = itm
            best_match["info"] = (dx, dy, l2)
            min_l2 = l2

        if min_l2 < 0.8: # todo
            break

    if not bool(best_match):
        raise Exception("no_match")

    print("cropped_img_path : {}, best_match: {}".format(cropped_img_path, best_match))
    compose_path = path.join(DIR_COMPOSE, path.basename(cropped_img_path))
    fm.compose(best_match["f_name"], cropped_img_path, compose_path, (best_match["info"][0], best_match["info"][1]))
    resp = client.detect(compose_path)
    if int(resp[0]) != 200:
        raise Exception("detect_error")
    return int(resp[1])



