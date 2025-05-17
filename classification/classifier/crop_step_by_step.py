#coding=utf-8

import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import os
import shutil
from os import listdir, system
from os import mkdir
from shutil import copyfile
from os.path import isfile, join
import shutil
import _pickle as pickle

def show(desc, img):
    cv.imshow(desc, img)
    cv.waitKey()
    cv.destroyAllWindows()

def rect_crop_left_top(src, xb_offset, yb_offset):
    #src = cv.imread(img_path)
    h, w, c = src.shape
    #w_factor = random.randint(90, 95)
    #h_factor = random.randint(90, 95)
    #w_ = int(w * w_factor / 100.0)
    #h_ = int(h * h_factor / 100.0)
    x0 = 0
    y0 = 0
    x1 = w - xb_offset
    y1 = h - yb_offset
    cropped = src[y0: y1, x0: x1]
    return cropped

def extract(img_dir, extract_dir):
    path_exe = "/Users/macrosea/ws/git/work/demo-project/watermark/cmake-build-debug/watermark "
    os.system(path_exe + " extract " + img_dir + " " + extract_dir + "/")

def crop_left_top():
    saved_dir = "/tmp/crop_left_top"
    if os.path.exists(saved_dir):
        shutil.rmtree(saved_dir)
    os.makedirs(saved_dir)
    croped_dir = saved_dir + "/croped"
    os.makedirs(croped_dir)
    wm_dir = saved_dir + "/wm"
    os.makedirs(wm_dir)

    img_path = "/tmp/v_embed_wm/ia_100000046.bmp"
    src = cv.imread(img_path)
    h, w, c = src.shape
    range_h = int(h * 10 /100.0)
    range_w = int(w * 10 /100.0)
    for i in range(range_h):
        for j in range(range_w):
            res = rect_crop_left_top(src, j, i)
            cv.imwrite(croped_dir + "/x_offset_{}-y_offset{}_.bmp".format(j, i), res)

    extract(croped_dir, wm_dir)

def rect_crop_left_bottom(src, x_offset, y_offset):
    h, w, c = src.shape
    x0 = 0
    y0 = y_offset
    x1 = w - x_offset
    y1 = h
    cropped = src[y0: y1, x0: x1]
    return cropped
 
def crop_left_bottom():
    saved_dir = "/tmp/crop_left_bottom"
    if os.path.exists(saved_dir):
        shutil.rmtree(saved_dir)
    os.makedirs(saved_dir)
    croped_dir = saved_dir + "/croped"
    os.makedirs(croped_dir)
    wm_dir = saved_dir + "/wm"
    os.makedirs(wm_dir)

    img_path = "/tmp/v_embed_wm/ia_v_0710_100000307.bmp"
    src = cv.imread(img_path)
    h, w, c = src.shape
    range_h = int(h * 10 /100.0)
    range_w = int(w * 10 /100.0)

    for i in range(range_h):
        for j in range(range_w):
            res = rect_crop_left_bottom(src, j, i)
            cv.imwrite(croped_dir + "/x_offset_{}-y_offset{}_.bmp".format(j, i), res)

    extract(croped_dir, wm_dir)

if __name__ == "__main__":
    crop_left_bottom()
    pass