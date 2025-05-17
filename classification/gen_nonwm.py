#coding=utf-8

import cv2 as cv
import numpy as np
#from PIL import Image, ImageDraw, ImageFont
import random
import os
import shutil
from os import listdir, system
from os import mkdir
from shutil import copyfile
from os.path import isfile, join
import shutil

path_exe = "/Users/macrosea/ws/git/work/watermark_project/watermark-jni/cmake-build-debug/watermark "

def scan_and_extract(scan_dir):
    scan_dir = scan_dir if scan_dir.endswith('/') else scan_dir+'/'
    saved_dir = "/tmp/extract_non"
    # if os.path.exists(saved_dir):
    #     shutil.rmtree(saved_dir)
    # os.makedirs(saved_dir)
    os.system(path_exe + " extract " + scan_dir + " " + saved_dir)

def scan_and_expand_sample(scan_dir):
    saved_dir = "/tmp/sample_augmentation"
    if os.path.exists(saved_dir):
        shutil.rmtree(saved_dir)
    os.makedirs(saved_dir)

    scan_dir = scan_dir if scan_dir.endswith('/') else scan_dir+'/'
    files = [f for f in listdir(scan_dir) if ((f.endswith("jpeg") or f.endswith("jpg") or f.endswith("bmp")) and isfile(join(scan_dir, f)))]
    for idx, itm in enumerate(files):
        img_path = scan_dir + itm
        image = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        image_x = cv.flip(image, 0)
        image_y = cv.flip(image, 1)
        image_xy = cv.flip(image, -1)
        cv.imwrite(join(saved_dir, "x_"+itm), image_x)
        cv.imwrite(join(saved_dir, "y_"+itm), image_y)
        cv.imwrite(join(saved_dir, "xy_"+itm), image_xy)

if __name__ == "__main__":
    scan_and_expand_sample("/Users/macrosea/tmp/photo_0803")
    scan_and_extract("/tmp/sample_augmentation")
    pass
