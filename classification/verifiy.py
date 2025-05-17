import cv2
from cv2 import dnn
import numpy as np
import shutil
from os import listdir, system
from os.path import isfile, join

print(cv2.__version__)
class_name = ['non', 'wm']
net = dnn.readNetFromTensorflow('/tmp/frozen_models/frozen_graph.pb')

def predict(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    blob = cv2.dnn.blobFromImage(img,     #  这里对颜色进行反转，这是训练图片存储格式的问题，可以把数值打印出来看下一
                             scalefactor=1.0/225.,
                             size=(32, 32),
                             mean=(0, 0, 0),
                             swapRB=False,
                             crop=False)
    net.setInput(blob)
    out = net.forward()
    out = out.flatten()                         
    classId = np.argmax(out)
    return classId

def scan(scan_dir):
    scan_dir = scan_dir if scan_dir.endswith('/') else scan_dir+'/'
    files = [f for f in listdir(scan_dir) if ((f.endswith("jpeg") or f.endswith("jpg") or f.endswith("bmp")) and isfile(join(scan_dir, f)))]
    for idx, itm in enumerate(files):
        img_path = scan_dir + itm
        res = predict(img_path)
        #print("res: " + str(res))
        if res != 1:
            shutil.move(img_path, "/tmp/check_1/"+itm)


scan_dir = "/tmp/extract_wm/check_1"
scan(scan_dir)


# #img = cv2.imread('/Users/macrosea/ws/work/watermark/classifier/tf4wm/verify/wm/test_wm_03.png', cv2.IMREAD_GRAYSCALE)
# img = cv2.imread('/Users/macrosea/ws/work/watermark/classifier/tf4wm/verify/no/flip_ia_4800000019.bmp', cv2.IMREAD_GRAYSCALE)
# cv2.imshow("2", img)
# print(img.shape)
# # print(1-img_cv21/255.)
# blob = cv2.dnn.blobFromImage(img,     #  这里对颜色进行反转，这是训练图片存储格式的问题，可以把数值打印出来看下一
#                              scalefactor=1.0/225.,
#                              size=(32, 32),
#                              mean=(0, 0, 0),
#                              swapRB=False,
#                              crop=False)

# print("[INFO]img shape: ", blob.shape)

# net = dnn.readNetFromTensorflow('/tmp/frozen_models/frozen_graph.pb')
# print("success!!")
# net.setInput(blob)
# out = net.forward()
# out = out.flatten()


# print("classId",classId)
# print("预测结果为：",class_name[classId])
# cv2.waitKey(0)
