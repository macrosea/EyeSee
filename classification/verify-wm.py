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
import _pickle as pickle

path_exe = "/Users/macrosea/ws/git/work/watermark_project/watermark-jni/cmake-build-debug/watermark "
#org_img_dir = "/tmp/test_img"
org_img_dir = "/Users/macrosea/ws/work/watermark/sets_verify"

def scan_and_statistic(img_dir, expect_v, desc):
    print(desc + "; expect: " + ("non" if expect_v == 0 else "wm"))
    os.system(path_exe + " stat " + img_dir + " " + str(expect_v))

def scan_and_embed(img_dir, outDir):
    os.system(path_exe + " embed " + img_dir + " " + outDir)

def no_wm():
    scan_and_statistic(org_img_dir, 0, "not embed org image")

def embed():
    saved_dir = "/tmp/v_embed_wm"
    if os.path.exists(saved_dir):
        shutil.rmtree(saved_dir)
    os.makedirs(saved_dir)
    scan_and_embed(org_img_dir, saved_dir)
    scan_and_statistic(saved_dir, 1, "wm_embed")


def _overlay(img_path, cover):
    src = cv.imread(img_path)
    y, x, c = src.shape
    h = int(y * 0.15)
    w = int(x * 0.15)
    pos_y = random.randint(0, y-h)
    pos_x = random.randint(0, x-w)

    resized = cv.resize(cover, (w, h))
    #cropped = src[5: 5+h, 5:5+w]
    #block = (255*np.random.rand(*cropped.shape)).astype(np.uint8) 
    #block = (np.random.rand(h,w,3)*255).astype(np.uint8)

    src[pos_y:h+pos_y, pos_x:w+pos_x] = resized
    return src

def overlay():
    embed_img_dir = "/tmp/v_embed_wm"
    package_dir = "/tmp/v_embed_overlay"
    if os.path.exists(package_dir):
        shutil.rmtree(package_dir)
    os.makedirs(package_dir)

    cover_imgs = list()
    cover_imgs.append(cv.imread("/Users/macrosea/ws/work/watermark/cover/douyin.jpeg"))
    cover_imgs.append(cv.imread("/Users/macrosea/ws/work/watermark/cover/sina.jpeg"))
    cover_imgs.append(cv.imread("/Users/macrosea/ws/work/watermark/cover/v_china.jpeg"))

    imgFiles = [f for f in listdir(embed_img_dir) if ((f.endswith("jpeg") or f.endswith("jpg") or f.endswith("bmp")) and isfile(join(embed_img_dir, f)))]
    for idx, itm in enumerate(imgFiles):
        for i in range(4):
            index = random.randint(0, 2)
            #print(index)
            c_img = cover_imgs[index]
            overlay_img = _overlay(join(embed_img_dir, itm), c_img)
            saved_path = join(package_dir, "overlay_" + str(i) + "_" + itm)
            cv.imwrite(saved_path, overlay_img)

    scan_and_statistic(package_dir, 1, "wm_overlay")
    if os.path.exists(package_dir):
        shutil.rmtree(package_dir)

def _cover_block(img_path):
    src = cv.imread(img_path)
    y, x, c = src.shape
    h = int(y * 0.15)
    w = int(x * 0.15)
    pos_y = random.randint(0, y-h)
    pos_x = random.randint(0, x-w)

    #cropped = src[5: 5+h, 5:5+w]
    #block = (255*np.random.rand(*cropped.shape)).astype(np.uint8) 
    block = (np.random.rand(h,w,3)*255).astype(np.uint8)
    src[pos_y:h+pos_y, pos_x:w+pos_x] = block
    return src

def cover_block():
    embed_img_dir = "/tmp/v_embed_wm"
    package_dir = "/tmp/v_embed_cover_block"
    if os.path.exists(package_dir):
        shutil.rmtree(package_dir)
    os.makedirs(package_dir)

    imgFiles = [f for f in listdir(embed_img_dir) if ((f.endswith("jpeg") or f.endswith("jpg") or f.endswith("bmp")) and isfile(join(embed_img_dir, f)))]
    for idx, itm in enumerate(imgFiles):
        for i in range(4):
            overlay_img = _cover_block(join(embed_img_dir, itm))
            saved_path = join(package_dir, "cover_block_" + str(i) + "_" + itm)
            cv.imwrite(saved_path, overlay_img)

    scan_and_statistic(package_dir, 1, "cover_block")
    if os.path.exists(package_dir):
        shutil.rmtree(package_dir)      


def _compress(img_path):  
    src = cv.imread(img_path)
    quality = random.randint(75, 80)
    img_param = [int(cv.IMWRITE_JPEG_QUALITY), quality]
    _, img_encode = cv.imencode('.jpg', src, img_param)
    ret = cv.imdecode(img_encode, cv.IMREAD_COLOR)
    return ret    

def compress():
    embed_img_dir = "/tmp/v_embed_wm"
    package_dir = "/tmp/v_embed_compress"
    if os.path.exists(package_dir):
        shutil.rmtree(package_dir)
    os.makedirs(package_dir)

    imgFiles = [f for f in listdir(embed_img_dir) if ((f.endswith("jpeg") or f.endswith("jpg") or f.endswith("bmp")) and isfile(join(embed_img_dir, f)))]
    for idx, itm in enumerate(imgFiles):
        for i in range(4):
            overlay_img = _compress(join(embed_img_dir, itm))
            saved_path = join(package_dir, "compress_" + str(i) + "_" + itm)
            cv.imwrite(saved_path, overlay_img)

    scan_and_statistic(package_dir, 1, "compress")
    if os.path.exists(package_dir):
        shutil.rmtree(package_dir)  

def v_contrast():
    alpha = random.randint(8, 12)/10.0
    beta = random.randint(0, 20)
    return alpha, beta

def _saturate(img_path):  # a [0, 0.3]; b [0, 100]
    src = cv.imread(img_path)
    alpha, beta = v_contrast()
    mat = (alpha * src + beta)
    res = np.uint8(np.clip(mat, 0, 255))
    return res    

def saturate():
    embed_img_dir = "/tmp/v_embed_wm"
    package_dir = "/tmp/v_embed_saturate"
    if os.path.exists(package_dir):
        shutil.rmtree(package_dir)
    os.makedirs(package_dir)

    imgFiles = [f for f in listdir(embed_img_dir) if ((f.endswith("jpeg") or f.endswith("jpg") or f.endswith("bmp")) and isfile(join(embed_img_dir, f)))]
    for idx, itm in enumerate(imgFiles):
        for i in range(4):
            overlay_img = _saturate(join(embed_img_dir, itm))
            saved_path = join(package_dir, "saturate_" + str(i) + "_" + itm)
            cv.imwrite(saved_path, overlay_img)

    scan_and_statistic(package_dir, 1, "saturate")
    if os.path.exists(package_dir):
        shutil.rmtree(package_dir)  



def rect_crop_left_top(img_path):
    src = cv.imread(img_path)
    h, w, c = src.shape
    w_factor = random.randint(88, 95)
    h_factor = random.randint(88, 95)
    w_ = int(w * w_factor / 100)
    h_ = int(h * h_factor / 100)
    x0 = 0
    y0 = 0
    x1 = w_
    y1 = h_
    cropped = src[y0: y1, x0: x1]
    return cropped

def crop_left_top():
    embed_img_dir = "/tmp/v_embed_wm"
    package_dir = "/tmp/v_embed_crop_left_top"
    if os.path.exists(package_dir):
        shutil.rmtree(package_dir)
    os.makedirs(package_dir)

    imgFiles = [f for f in listdir(embed_img_dir) if ((f.endswith("jpeg") or f.endswith("jpg") or f.endswith("bmp")) and isfile(join(embed_img_dir, f)))]
    for idx, itm in enumerate(imgFiles):
        for i in range(4):
            img = rect_crop_left_top(join(embed_img_dir, itm))
            saved_path = join(package_dir, "lt_" + str(i) + "_" + itm)
            cv.imwrite(saved_path, img)                       

    scan_and_statistic(package_dir, 1, "crop_left_top")
    if os.path.exists(package_dir):
        shutil.rmtree(package_dir)  


def rect_crop_left_bottom(img_path):
    src = cv.imread(img_path)
    h, w, c = src.shape
    w_factor = random.randint(88, 95)
    h_factor = random.randint(88, 95)
    w_ = int(w * w_factor / 100)
    h_ = int(h * h_factor / 100)
    x0 = 0
    y0 = h - h_
    x1 = w_
    y1 = h
    cropped = src[y0: y1, x0: x1]
    return cropped

def crop_left_bottom():
    embed_img_dir = "/tmp/v_embed_wm"
    package_dir = "/tmp/v_embed_crop_left_bottom"
    if os.path.exists(package_dir):
        shutil.rmtree(package_dir)
    os.makedirs(package_dir)

    imgFiles = [f for f in listdir(embed_img_dir) if ((f.endswith("jpeg") or f.endswith("jpg") or f.endswith("bmp")) and isfile(join(embed_img_dir, f)))]
    for idx, itm in enumerate(imgFiles):
        for i in range(4):
            img = rect_crop_left_bottom(join(embed_img_dir, itm))
            saved_path = join(package_dir, "lb_" + str(i) + "_" + itm)
            cv.imwrite(saved_path, img)                       
    scan_and_statistic(package_dir, 1, "crop_left_bottom")
    if os.path.exists(package_dir):
        shutil.rmtree(package_dir)  

def rect_crop_right_top(img_path):
    src = cv.imread(img_path)
    h, w, c = src.shape
    w_factor = random.randint(88, 95)
    h_factor = random.randint(88, 95)
    w_ = int(w * w_factor / 100)
    h_ = int(h * h_factor / 100)
    x0 = w - w_
    y0 = 0
    x1 = w
    y1 = h_
    cropped = src[y0: y1, x0: x1]
    return cropped

def crop_right_top():
    embed_img_dir = "/tmp/v_embed_wm"
    package_dir = "/tmp/v_embed_crop_right_top"
    if os.path.exists(package_dir):
        shutil.rmtree(package_dir)
    os.makedirs(package_dir)

    imgFiles = [f for f in listdir(embed_img_dir) if ((f.endswith("jpeg") or f.endswith("jpg") or f.endswith("bmp")) and isfile(join(embed_img_dir, f)))]
    for idx, itm in enumerate(imgFiles):
        for i in range(4):
            img = rect_crop_right_top(join(embed_img_dir, itm))
            saved_path = join(package_dir, "rt_" + str(i) + "_" + itm)
            cv.imwrite(saved_path, img)                     

    scan_and_statistic(package_dir, 1, "crop_right_top")
    if os.path.exists(package_dir):
        shutil.rmtree(package_dir)  


def rect_crop_right_bottom(img_path):
    src = cv.imread(img_path)
    h, w, c = src.shape
    w_factor = random.randint(88, 95)
    h_factor = random.randint(88, 95)
    w_ = int(w * w_factor / 100)
    h_ = int(h * h_factor / 100)
    x0 = w - w_
    y0 = h - h_
    x1 = w
    y1 = h
    cropped = src[y0: y1, x0: x1]
    return cropped

def crop_right_bottom():
    embed_img_dir = "/tmp/v_embed_wm"
    package_dir = "/tmp/v_embed_crop_right_bottom"
    if os.path.exists(package_dir):
        shutil.rmtree(package_dir)
    os.makedirs(package_dir)

    imgFiles = [f for f in listdir(embed_img_dir) if ((f.endswith("jpeg") or f.endswith("jpg") or f.endswith("bmp")) and isfile(join(embed_img_dir, f)))]
    for idx, itm in enumerate(imgFiles):
        for i in range(4):
            img = rect_crop_right_bottom(join(embed_img_dir, itm))
            saved_path = join(package_dir, "rb_" + str(i) + "_" + itm)
            cv.imwrite(saved_path, img)                        

    scan_and_statistic(package_dir, 1, "crop_right_bottom")
    if os.path.exists(package_dir):
        shutil.rmtree(package_dir)  



if __name__ == "__main__":
    mat = cv.imread("/tmp/test_img/ia_v_03912.jpg")
    cv.imwrite("/tmp/test_img/ia_v_03912.bmp", mat)
    no_wm()  
    embed()
    #overlay()
    #cover_block()
    compress()
    saturate()
    crop_left_top()
    crop_left_bottom()
    crop_right_top()
    crop_right_bottom()
    pass

