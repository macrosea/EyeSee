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

def show(desc, img):
    cv.imshow(desc, img)
    cv.waitKey()
    cv.destroyAllWindows()

def v_code():
    ret = ""
    for i in range(6):
        num = random.randint(0, 9)
        letter = chr(random.randint(97, 122))#小写字母
        Letter = chr(random.randint(65, 90))#大写字母
        s = str(random.choice([num,letter,Letter]))
        ret += s
    return ret

def v_pos(shape):
    y, x, _ = shape
    x_ = random.randint(10, x-200)
    y_ = random.randint(10, y-100)
    return x_, y_

def v_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return b, g, r

def v_contrast():
    alpha = random.randint(8, 12)/10.0
    beta = random.randint(0, 20)
    return alpha, beta



def putTxt(img_path):
    src = cv.imread(img_path)
    image = cv.putText(src, v_code(), v_pos(src.shape), cv.FONT_HERSHEY_COMPLEX, 3.0, v_color(), 6)
    #show("txt", image)
    return image

def put2Txt(img_path):
    src = cv.imread(img_path)
    image = cv.putText(src, v_code(), v_pos(src.shape), cv.FONT_HERSHEY_COMPLEX, 3.0, v_color(), 5)
    image = cv.putText(src, v_code(), v_pos(src.shape), cv.FONT_HERSHEY_COMPLEX, 3.0, v_color(), 5)
    #show("txt", image)
    return image

def saturate(img_path):  # a [0, 0.3]; b [0, 100]
    src = cv.imread(img_path)
    alpha, beta = v_contrast()
    mat = (alpha * src + beta)
    res = np.uint8(np.clip(mat, 0, 255))
    return res

def overlay(img_path):
    src = cv.imread(img_path)
    y, x, c = src.shape
    h = int(y/5)
    w = int(x/5)
    pos_y = random.randint(0, y-h)
    pos_x = random.randint(0, x-w)

    #cropped = src[5: 5+h, 5:5+w]
    #block = (255*np.random.rand(*cropped.shape)).astype(np.uint8) 
    block = (np.random.rand(h,w,3)*255).astype(np.uint8)

    src[pos_y:h+pos_y, pos_x:w+pos_x] = block
    return src
  
def overlay_log(dst_path, log_path):
    dst_img = cv.imread(dst_path)
    dst_y, dst_x, _ = dst_img.shape
    w = int(dst_x/6)
    h = int(dst_y/6)

    pos_y = random.randint(0, dst_y-h)
    pos_x = random.randint(0, dst_x-w)

    log_img = cv.imread(log_path)
    log_img = cv.resize(log_img, (w, h))
    rows, cols, channels = log_img.shape
   
    roi = dst_img[pos_y:h+pos_y, pos_x:w+pos_x] 

    log2gray = cv.cvtColor(log_img, cv.COLOR_BGR2GRAY)# 颜色空间的转换
    ret, mask = cv.threshold(log2gray, 30, 255, cv.THRESH_BINARY)# 掩码 黑色
    mask_inv = cv.bitwise_not(mask)# 掩码取反 白色
    # #取mask中像素不为的0的值，其余为0
    log_bg = cv.bitwise_and(log_img, log_img, mask=mask)
    log_fg = cv.bitwise_and(roi, roi, mask=mask_inv)
    res = cv.add(log_bg, log_fg)

    dst_img[pos_y:h+pos_y, pos_x:w+pos_x] = res
    #src[pos_y:h+pos_y, pos_x:w+pos_x] = block
    return dst_img

def compress(img_path):  
    src = cv.imread(img_path)
    quality = random.randint(80, 100)
    img_param = [int(cv.IMWRITE_JPEG_QUALITY), quality]
    _, img_encode = cv.imencode('.jpg', src, img_param)
    ret = cv.imdecode(img_encode, cv.IMREAD_COLOR)
    return ret

def rect_crop_left_top(img_path):
    src = cv.imread(img_path)
    h, w, c = src.shape
    w_factor = random.randint(90, 95)
    h_factor = random.randint(90, 95)
    w_ = int(w * w_factor / 100)
    h_ = int(h * h_factor / 100)
    x0 = 0
    y0 = 0
    x1 = w_
    y1 = h_
    cropped = src[y0: y1, x0: x1]
    return cropped

def rect_crop_left_bottom(img_path):
    src = cv.imread(img_path)
    h, w, c = src.shape
    w_factor = random.randint(90, 95)
    h_factor = random.randint(90, 95)
    w_ = int(w * w_factor / 100)
    h_ = int(h * h_factor / 100)
    x0 = 0
    y0 = h - h_
    x1 = w_
    y1 = h
    cropped = src[y0: y1, x0: x1]
    return cropped

def rect_crop_right_top(img_path):
    src = cv.imread(img_path)
    h, w, c = src.shape
    w_factor = random.randint(90, 95)
    h_factor = random.randint(90, 95)
    w_ = int(w * w_factor / 100)
    h_ = int(h * h_factor / 100)
    x0 = w - w_
    y0 = 0
    x1 = w
    y1 = h_
    cropped = src[y0: y1, x0: x1]
    return cropped

def rect_crop_right_bottom(img_path):
    src = cv.imread(img_path)
    h, w, c = src.shape
    w_factor = random.randint(90, 95)
    h_factor = random.randint(90, 95)
    w_ = int(w * w_factor / 100)
    h_ = int(h * h_factor / 100)
    x0 = w - w_
    y0 = h - h_
    x1 = w
    y1 = h
    cropped = src[y0: y1, x0: x1]
    return cropped


def scan2enlarge(varPath, destDir):

    if os.path.exists(destDir):
        shutil.rmtree(destDir)
    os.makedirs(destDir)  

    imgFiles = [f for f in listdir(varPath) if (f.endswith("bmp") and isfile(join(varPath, f)))]
    for idx, itm in enumerate(imgFiles):
        img_path = join(varPath, itm) 
        print(img_path)
        for i in range(1):
            modified = compress(img_path)
            fn = os.path.splitext(itm)[0]
            randm =  random.randint(10000, 99999)
            path_mod = destDir + fn + '_compress_' + str(randm) + ".bmp"
            cv.imwrite(path_mod, modified)

            modified = overlay_log(img_path, "/Users/macrosea/ws/work/watermark/ocv_log.png")
            fn = os.path.splitext(itm)[0]
            randm =  random.randint(10000, 99999)
            path_mod = destDir + fn + '_ocv_log_' + str(randm) + ".bmp"
            cv.imwrite(path_mod, modified)

            modified = put2Txt(img_path)
            fn = os.path.splitext(itm)[0]
            randm =  random.randint(10000, 99999)
            path_mod = destDir + fn + '_put2txt_' + str(randm) + ".bmp"
            cv.imwrite(path_mod, modified)

            modified = putTxt(img_path)
            fn = os.path.splitext(itm)[0]
            randm =  random.randint(10000, 99999)
            path_mod = destDir + fn + '_putTxt_' + str(randm) + ".bmp"
            cv.imwrite(path_mod, modified)

            modified = overlay(img_path)
            fn = os.path.splitext(itm)[0]
            randm =  random.randint(10000, 99999)
            path_mod = destDir + fn + '_overlay_' + str(randm) + ".bmp"
            cv.imwrite(path_mod, modified)

            modified = saturate(img_path)
            fn = os.path.splitext(itm)[0]
            randm =  random.randint(10000, 99999)
            path_mod = destDir + fn + '_saturate_' + str(randm) + ".bmp"
            cv.imwrite(path_mod, modified)
            

            modified = rect_crop_left_top(img_path)
            fn = os.path.splitext(itm)[0]
            randm =  random.randint(10000, 99999)
            path_mod = destDir + fn + '_lt_' + str(randm) + ".bmp"
            cv.imwrite(path_mod, modified)

            modified = rect_crop_left_bottom(img_path)
            fn = os.path.splitext(itm)[0]
            randm =  random.randint(10000, 99999)
            path_mod = destDir + fn + '_lb_' + str(randm) + ".bmp"
            cv.imwrite(path_mod, modified)

            modified = rect_crop_right_top(img_path)
            fn = os.path.splitext(itm)[0]
            randm =  random.randint(10000, 99999)
            path_mod = destDir + fn + '_rt_' + str(randm) + ".bmp"
            cv.imwrite(path_mod, modified)

            modified = rect_crop_right_bottom(img_path)
            fn = os.path.splitext(itm)[0]
            randm =  random.randint(10000, 99999)
            path_mod = destDir + fn + '_rb_' + str(randm) + ".bmp"
            cv.imwrite(path_mod, modified)
        
def copy_files(src_dir, dst_dir):
    imgFiles = [f for f in listdir(src_dir) if (f.endswith("bmp") and isfile(join(src_dir, f)))]
    for idx, itm in enumerate(imgFiles):
        src = join(src_dir, itm) 
        shutil.copyfile(src, os.path.join(dst_dir, itm))

def gen_wm_sets():
    dir_embed_train = "/tmp/embed_train/"
    if os.path.exists(dir_embed_train):
        shutil.rmtree(dir_embed_train)
    os.makedirs(dir_embed_train) 

    dir_embed_verify = "/tmp/embed_verify/"
    if os.path.exists(dir_embed_verify):
        shutil.rmtree(dir_embed_verify)
    os.makedirs(dir_embed_verify) 

    dir_wm_train = "/tmp/wm_train/"
    if os.path.exists(dir_wm_train):
        shutil.rmtree(dir_wm_train)
    os.makedirs(dir_wm_train)

    dir_wm_verify = "/tmp/wm_verify/"
    if os.path.exists(dir_wm_verify):
        shutil.rmtree(dir_wm_verify)
    os.makedirs(dir_wm_verify)    


    path_exe = "/Users/macrosea/ws/git/work/demo-project/watermark/cmake-build-debug/watermark "
    path_sets_train = "/Users/macrosea/ws/work/watermark/sets_train/ "
    path_sets_verify = "/Users/macrosea/ws/work/watermark/sets_verify/ "


    os.system(path_exe + " embed " + path_sets_train + dir_embed_train)
    os.system(path_exe + " embed " + path_sets_verify + dir_embed_verify)
    
    ## extract
    dir_enlarge_train = "/tmp/embed_enlarge_train/"
    
    scan2enlarge(dir_embed_train, dir_enlarge_train)
    copy_files(dir_embed_train, dir_enlarge_train)
    os.system(path_exe + " extract " + dir_enlarge_train + " " + dir_wm_train)

    dir_enlarge_verify = "/tmp/embed_enlarge_verify/"
    scan2enlarge(dir_embed_verify, dir_enlarge_verify)
    copy_files(dir_embed_verify, dir_enlarge_verify)
    os.system(path_exe + " extract " + dir_enlarge_verify + " " + dir_wm_verify)

def gen_non_sets():
    path_exe = "/Users/macrosea/ws/git/work/demo-project/watermark/cmake-build-debug/watermark "
    path_sets_train = "/Users/macrosea/ws/work/watermark/sets_train/ "
    path_sets_verify = "/Users/macrosea/ws/work/watermark/sets_verify/ "
    ## fake no wm
    dir_no_wm_train = "/tmp/no_wm_train/"
    if os.path.exists(dir_no_wm_train):
        shutil.rmtree(dir_no_wm_train)
    os.makedirs(dir_no_wm_train)  
    os.system(path_exe + " extract " + path_sets_train + dir_no_wm_train)

    dir_no_wm_verify = "/tmp/no_wm_verify/"
    if os.path.exists(dir_no_wm_verify):
        shutil.rmtree(dir_no_wm_verify)
    os.makedirs(dir_no_wm_verify)
    os.system(path_exe + " extract " + path_sets_verify + dir_no_wm_verify)



def flip_hv():
    saved_dir = "/tmp/flip_train"
    if os.path.exists(saved_dir):
        shutil.rmtree(saved_dir)
    os.makedirs(saved_dir)  

    sets_train = "/Users/macrosea/ws/work/watermark/sets_train/"
    sets = [f for f in listdir(sets_train) if (f.endswith("jpg") and isfile(join(sets_train, f)))]   
    for idx, itm in enumerate(sets):
        path_itm = join(sets_train, itm)
        image = cv.imread(path_itm)
        hv_flip = cv.flip(image, -1)
        cv.imwrite(join(saved_dir, "flip_" + itm), hv_flip)



if __name__ == "__main__":
    gen_non_sets()
    flip_hv()
    path_exe = "/Users/macrosea/ws/git/work/demo-project/watermark/cmake-build-debug/watermark "
    dir_no_wm_train = "/tmp/no_wm_train/ "
    os.system(path_exe + " extract " + "/tmp/flip_train/  " + dir_no_wm_train)

    
