import numpy as np
from sklearn import svm
from sklearn.datasets  import load_digits
from sklearn.model_selection  import train_test_split
import _pickle as pickle
import cv2
import os 
import random
from os import listdir
from os.path import isfile, join
from sklearn.utils import shuffle
from sklearn.preprocessing import scale
from scipy.sparse import coo_matrix
import shutil



def gen_sample(saved_dir, count):
    for i in range(count):
        img=np.random.randint(0, 256, size=[64, 64], dtype=np.uint8)
        ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(join(saved_dir + "/rand_"+str(random.randint(100000, 999999))+".png"), thresh)

def img2array(img_path):
    #print(img_path)
    src = cv2.imread(img_path, 0)
    resized = cv2.resize(src,(28,28))
    #print("src channels: %s"%(resized.shape, ))
    data = resized.flatten()
    return data

def scan(dir_path):
    ret =list()
    g = os.walk(dir_path)
    for path, dir_list, file_list in g:  
        for file_name in file_list:
            if file_name.find("DS_Store") != -1:
                continue
            img_path = os.path.join(path, file_name)
            data = img2array(img_path)
            ret.append(data)
    return ret

def load_data_set(dir_path):
    wm_path = os.path.join(dir_path, "wm")
    no_path = os.path.join(dir_path, "no")
    wm_data = np.asarray(scan(wm_path))/255
    wm_target = np.ones(len(wm_data))
    no_data = np.asarray(scan(no_path))/255
    no_target = np.zeros(len(no_data), dtype = np.int)
    dataset = dict()
    dataset["data"] = np.concatenate((wm_data, no_data))
    dataset["target"] = np.concatenate((wm_target, no_target), axis = 0)
    return dataset


def train(dir_img_sets):
    #gen_sample(200)
    dataset = load_data_set(dir_img_sets)
    #train_x =  scale(dataset["data"])
    train_x = dataset["data"]
    x,test_x,y,test_y = train_test_split(train_x, dataset["target"], test_size=0.3, random_state=0)
    print(x.shape)
    print(y.shape)
    x_sparse = coo_matrix(x)
    x, x_sparse, y = shuffle(x, x_sparse, y, random_state=0)

    #model = svm.LinearSVC()
    model = svm.SVC(C=2.0, kernel='rbf') 
    model.fit(x, y)

    z = model.predict(test_x)

    print('result:', np.sum(z==test_y)/z.size)

    with open('./model.pkl', 'wb') as file:
        pickle.dump(model,file)

def inference(dirPath, result_dir):
    with open('./model.pkl', 'rb') as f_mod:
        clf2 = pickle.load(f_mod)
        imgFiles = [f for f in listdir(dirPath) if ((f.endswith("png") or f.endswith("bmp")) and isfile(join(dirPath, f)))]
        for idx, itm in enumerate(imgFiles):
            img_path = join(dirPath, itm)
            rawdata = img2array(img_path)
            data = np.asarray([rawdata])/255
            res = clf2.predict(data)
            #print("res: %d, file: %s"%(res, img_path))
            shutil.copyfile(img_path, os.path.join(result_dir, str(res[0]) + "_" + itm))

if __name__ == '__main__':
    train("/Users/macrosea/ws/work/watermark/classifier/images")
    result_dir = "/tmp/result_verify"
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir)  
    #inference("/tmp/no_wm_verify/", result_dir)
    inference("/Users/macrosea/ws/work/watermark/classifier/images/verify_no", result_dir)
    inference("/Users/macrosea/ws/work/watermark/classifier/images/verify_wm", result_dir)
    pass

