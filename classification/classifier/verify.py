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

def img2array(img_path):
    #print(img_path)
    src = cv.imread(img_path, 0)
    resized = cv.resize(src,(28,28))
    #print("src channels: %s"%(resized.shape, ))
    data = resized.flatten()
    return data

def predict(dirPath, result_dir, expect):
    with open('./model.pkl', 'rb') as f_mod:
        clf2 = pickle.load(f_mod)
        imgFiles = [f for f in listdir(dirPath) if ((f.endswith("png") or f.endswith("bmp")) and isfile(join(dirPath, f)))]
        count = 0
        for idx, itm in enumerate(imgFiles):
            img_path = join(dirPath, itm)
            rawdata = img2array(img_path)
            data = np.asarray([rawdata])/255
            res = clf2.predict(data)
            #print("res: %d, file: %s"%(res, img_path))
            if int(res[0]) != expect:
                count += 1
            shutil.copyfile(img_path, os.path.join(result_dir, str(res[0]) + "_" + itm))

        return len(imgFiles), count

def extract(img_dir, extract_dir):
    path_exe = "/tmp/watermark "
    os.system(path_exe + " extract " + img_dir + " " + extract_dir + "/")
   
def extract_predict(img_dir, expect, folder_name):
    package_dir = join("/tmp/", folder_name)
    if os.path.exists(package_dir):
        shutil.rmtree(package_dir)
    os.makedirs(package_dir)
    extract_dir = join(package_dir, "./extract")
    os.makedirs(extract_dir)
    result_dir = join(package_dir, "./result")
    os.makedirs(result_dir)
    extract(img_dir, extract_dir)
    total, error_count = predict(extract_dir, result_dir, expect)
    print("%s: total=%d, error_count=%d, accuracy=%.3f%%"%(folder_name, total, error_count, 100 *(total - error_count)/float(total)))
    

def no_wm():
    no_wm_img_dir = "/tmp/images"
    extract_predict(no_wm_img_dir, 0, "v_no_wm")

def _embed(img_dir, saved_dir):
    path_exe = "/tmp/watermark "
    os.system(path_exe + " embed " + img_dir + " " + saved_dir + "/")
    pass

def embed():
    img_dir = "/tmp/images"
    #img_dir = "/Users/macrosea/ws/work/watermark/image_from_zhao"
    saved_dir = "/tmp/v_embed_wm"
    if os.path.exists(saved_dir):
        shutil.rmtree(saved_dir)
    os.makedirs(saved_dir)
    _embed(img_dir, saved_dir)

    package_dir = saved_dir
    extract_predict(package_dir, 1, "wm_embed")


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

    imgFiles = [f for f in listdir(embed_img_dir) if ((f.endswith("png") or f.endswith("bmp")) and isfile(join(embed_img_dir, f)))]
    for idx, itm in enumerate(imgFiles):
        for i in range(4):
            index = random.randint(0, 2)
            #print(index)
            c_img = cover_imgs[index]
            overlay_img = _overlay(join(embed_img_dir, itm), c_img)
            saved_path = join(package_dir, "overlay_" + str(i) + "_" + itm)
            cv.imwrite(saved_path, overlay_img)

    extract_predict(package_dir, 1, "wm_overlay")

    

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

    imgFiles = [f for f in listdir(embed_img_dir) if ((f.endswith("png") or f.endswith("bmp")) and isfile(join(embed_img_dir, f)))]
    for idx, itm in enumerate(imgFiles):
        for i in range(4):
            overlay_img = _cover_block(join(embed_img_dir, itm))
            saved_path = join(package_dir, "cover_block_" + str(i) + "_" + itm)
            cv.imwrite(saved_path, overlay_img)

    extract_predict(package_dir, 1, "wm_cover_block")
    

def _compress(img_path):  
    src = cv.imread(img_path)
    quality = random.randint(78, 80)
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

    imgFiles = [f for f in listdir(embed_img_dir) if ((f.endswith("png") or f.endswith("bmp")) and isfile(join(embed_img_dir, f)))]
    for idx, itm in enumerate(imgFiles):
        for i in range(4):
            overlay_img = _compress(join(embed_img_dir, itm))
            saved_path = join(package_dir, "compress_" + str(i) + "_" + itm)
            cv.imwrite(saved_path, overlay_img)

    extract_predict(package_dir, 1, "wm_compress")


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

    imgFiles = [f for f in listdir(embed_img_dir) if ((f.endswith("png") or f.endswith("bmp")) and isfile(join(embed_img_dir, f)))]
    for idx, itm in enumerate(imgFiles):
        for i in range(4):
            overlay_img = _saturate(join(embed_img_dir, itm))
            saved_path = join(package_dir, "saturate_" + str(i) + "_" + itm)
            cv.imwrite(saved_path, overlay_img)

    extract_predict(package_dir, 1, "wm_saturate")


def rect_crop_left_top(img_path):
    src = cv.imread(img_path)
    h, w, c = src.shape
    w_factor = random.randint(89, 91)
    h_factor = random.randint(89, 91)
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

    imgFiles = [f for f in listdir(embed_img_dir) if ((f.endswith("png") or f.endswith("bmp")) and isfile(join(embed_img_dir, f)))]
    for idx, itm in enumerate(imgFiles):
        for i in range(4):
            img = rect_crop_left_top(join(embed_img_dir, itm))
            saved_path = join(package_dir, "lt_" + str(i) + "_" + itm)
            cv.imwrite(saved_path, img)                       

    extract_predict(package_dir, 1, "wm_crop_left_top")


def rect_crop_left_bottom(img_path):
    src = cv.imread(img_path)
    h, w, c = src.shape
    w_factor = random.randint(89, 91)
    h_factor = random.randint(89, 91)
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

    imgFiles = [f for f in listdir(embed_img_dir) if ((f.endswith("png") or f.endswith("bmp")) and isfile(join(embed_img_dir, f)))]
    for idx, itm in enumerate(imgFiles):
        for i in range(4):
            img = rect_crop_left_bottom(join(embed_img_dir, itm))
            saved_path = join(package_dir, "lb_" + str(i) + "_" + itm)
            cv.imwrite(saved_path, img)                       

    extract_predict(package_dir, 1, "wm_crop_left_bottom")

def rect_crop_right_top(img_path):
    src = cv.imread(img_path)
    h, w, c = src.shape
    w_factor = random.randint(89, 91)
    h_factor = random.randint(89, 91)
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

    imgFiles = [f for f in listdir(embed_img_dir) if ((f.endswith("png") or f.endswith("bmp")) and isfile(join(embed_img_dir, f)))]
    for idx, itm in enumerate(imgFiles):
        for i in range(4):
            img = rect_crop_right_top(join(embed_img_dir, itm))
            saved_path = join(package_dir, "rt_" + str(i) + "_" + itm)
            cv.imwrite(saved_path, img)                     

    extract_predict(package_dir, 1, "wm_crop_right_top")


def rect_crop_right_bottom(img_path):
    src = cv.imread(img_path)
    h, w, c = src.shape
    w_factor = random.randint(89, 91)
    h_factor = random.randint(89, 91)
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

    imgFiles = [f for f in listdir(embed_img_dir) if ((f.endswith("png") or f.endswith("bmp")) and isfile(join(embed_img_dir, f)))]
    for idx, itm in enumerate(imgFiles):
        for i in range(4):
            img = rect_crop_right_bottom(join(embed_img_dir, itm))
            saved_path = join(package_dir, "rb_" + str(i) + "_" + itm)
            cv.imwrite(saved_path, img)                        

    extract_predict(package_dir, 1, "wm_crop_right_bottom")

if __name__ == "__main__":
    no_wm()
    #exit(0)
    embed()
    overlay()
    cover_block()
    compress()
    saturate()
   
    crop_left_top()
    crop_left_bottom()
    crop_right_top()
    crop_right_bottom()
    pass