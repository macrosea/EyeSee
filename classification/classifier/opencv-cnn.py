import cv2
import numpy as np
from cv2 import dnn

print(cv2.__version__)


class_name = ['non', 'wm']
#img = cv2.imread('/Users/macrosea/ws/work/watermark/classifier/tf4wm/verify/wm/test_wm_03.png', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('/Users/macrosea/ws/work/watermark/classifier/tf4wm/verify/no/flip_ia_4800000019.bmp', cv2.IMREAD_GRAYSCALE)
cv2.imshow("2", img)
print(img.shape)
# print(1-img_cv21/255.)
blob = cv2.dnn.blobFromImage(img,
                             scalefactor=1.0/225.,
                             size=(32, 32),
                             mean=(0, 0, 0),
                             swapRB=False,
                             crop=False)

print("[INFO]img shape: ", blob.shape)

net = dnn.readNetFromTensorflow('/tmp/frozen_models/frozen_graph.pb')
print("success!!")
net.setInput(blob)
out = net.forward()
out = out.flatten()

classId = np.argmax(out)
print("classId",classId)
print("预测结果为：",class_name[classId])
cv2.waitKey(0)
