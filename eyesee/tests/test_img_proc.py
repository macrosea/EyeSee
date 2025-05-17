#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
sys.path.append("..")

from search.feats_store import ImgFeatsStore
from common.config import IMG_FEATS_STORE_NAME
from search.search import Search
from wm_client.wm_client import WmClient
from img_processor.feature_match import FeatureMatch
from tests.helper import save_rect_crop

from common.file_utils import create_dir

import os
import shutil
import cv2 as cv
#

IMG_FEATS_STORE_NAME = "/tmp/feature.h5"


def show_matches():
    fn = "ia_1600000195.jpg"
    fn = "ia_100000017.jpg"
    fn = "ia_1100000203.jpg"
    fn = "ia_1400000279.jpg"
    fn = "ia_200000187.jpg"
    fn = "ia_1600000195.jpg"
    fn = "ia_700000215.jpg"
    fn = "ia_2200000217.jpg"
    img_path = "/tmp/ws/work/watermark/img_store/wm-0902/" + fn
    cropped_path = save_rect_crop(img_path, "/tmp/")
    cropped_path = "/tmp/wm/cropped/" +fn

    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    cropped = cv.imread(cropped_path, cv.IMREAD_GRAYSCALE)

    # img = cv.imread(img_path)
    # cropped = cv.imread(cropped_path)
    # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cropped = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)

    sift = cv.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img, None)
    kp2, des2 = sift.detectAndCompute(cropped, None)

    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv.drawMatches(img, kp1, cropped, kp2, matches[:FeatureMatch.LIMIT], None, flags=2)
    cv.imshow("", img3)
    cv.waitKey(0)

    # bf = cv.BFMatcher(cv.NORM_L2)
    # matches = bf.knnMatch(des1, des2, k=2)
    # # Apply ratio test
    # good = []
    # for m, n in matches:
    #     if m.distance < 0.3 * n.distance:
    #         good.append([m])
    # # cv2.drawMatchesKnn expects list of lists as matches.
    # print(len(good))
    # img3 = cv.drawMatchesKnn(img, kp1, cropped, kp2, good[:30], None, flags=2)
    # cv.imshow("", img3)
    # cv.waitKey(0)


def test_fm():
    img = "/tmp/tmp/wm/ia_100000137.jpg"
    img = "/tmp/ws/work/watermark/img_store/wm-0902/ia_100000017.jpg"
    cropped = save_rect_crop(img, "/tmp/")
    fm = FeatureMatch()
    fm.compose(img, cropped, "/tmp/saved.jpg")


def test_computer_delta_then_compose():
    fn = "ia_1100000203.jpg"
    fn = "ia_1400000279.jpg"
    fn = "ia_200000187.jpg"
    fn = "ia_1600000195.jpg"
    fn = "ia_2200000217.jpg"
    img = "/tmp/ws/work/watermark/img_store/wm-0902/" + fn
    cropped = save_rect_crop(img, "/tmp/")

    fm = FeatureMatch()
    dx, dy, l2 = fm.match(cv.imread(img, cv.IMREAD_GRAYSCALE), cv.imread(cropped, cv.IMREAD_GRAYSCALE))
    print(dx, dy, l2)
    fm.compose(img, "/tmp/" + fn, "/tmp/wm/" + fn, (dx, dy))

    cv.imshow("test", cv.imread("/tmp/wm/" + fn))
    cv.waitKey(0)


def test_computer_delta():
    fn = "ia_1600000195.jpg"
    fn = "ia_700000215.jpg"
    fn = "ia_2200000217.jpg"
    img = "/tmp/ws/work/watermark/img_store/wm-0902/" + fn

    cropped = save_rect_crop(img, "/tmp/")
    cropped = "/tmp/wm/cropped/" + fn
    fm = FeatureMatch()

    gray_org = cv.cvtColor(cv.imread(img), cv.COLOR_BGR2GRAY)  # todo
    gray_cropped = cv.cvtColor(cv.imread(cropped), cv.COLOR_BGR2GRAY)
    feat1 = fm.detect_feature(gray_org)
    feat2 = fm.detect_feature(gray_cropped)

    # feat1 = fm.detect_feature(cv.imread(img, cv.IMREAD_GRAYSCALE))
    # feat2 = fm.detect_feature(cv.imread(cropped, cv.IMREAD_GRAYSCALE))
    res = fm.computer_delta(feat1, feat2)
    print(res)



if __name__ == '__main__':
    #feature()
    #img = "/tmp/ws/work/watermark/img_store/wm-0902/ia_2000000265.jpg"
    #test_fm()
    #test_computer_delta_then_compose()
    test_computer_delta()
    show_matches()
