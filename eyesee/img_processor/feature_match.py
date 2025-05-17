# -*- coding: utf-8 -*-

# @Time    : 2019/10/05
# @Author  : macrosea
# @File    : feature_match.py


import cv2
from common.debug import debug_show


class FeatureMatch:
    LIMIT = 12
    def __init__(self, method="sift"):
        '''
         L1 and L2 norms are preferable choices for SIFT and SURF descriptors,
         NORM_HAMMING should be used with ORB, BRISK and BRIEF,
         NORM_HAMMING2 should be used with ORB when WTA_K==3 or 4
         '''
        members = {
            "sift": [cv2.xfeatures2d_SIFT.create, cv2.NORM_L2],
            "surf": [cv2.xfeatures2d_SURF.create, cv2.NORM_L2],
            "orb": [cv2.ORB_create, cv2.NORM_HAMMING]
        }
        if method not in members:
            self.__init_error_exit()

        detector = members.get(method)[0]
        self.detector = detector()
        self.normType = members.get(method)[1]

    @staticmethod
    def __init_error_exit():
        print("[FeatureMatch.__init__]:: unknown method")
        exit(-1)

    def detect_feature(self, gray_img):
        #img = cv2.imread(target)
        #gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
        keyPoint, descriptor = self.detector.detectAndCompute(gray_img, None)
        return keyPoint, descriptor

    def match(self, gray_img1, gray_img2):
        kp1, des1 = self.detect_feature(gray_img1)
        kp2, des2 = self.detect_feature(gray_img2)
        return self.computer_delta((kp1, des1), (kp2, des2))

    def computer_delta(self, feat1, feat2):
        kp1, des1 = feat1
        kp2, des2 = feat2
        bf = cv2.BFMatcher(self.normType, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        c1 = {"x": 0, "y": 0}
        c2 = {"x": 0, "y": 0}
        matches = matches[:] if len(matches) < FeatureMatch.LIMIT else matches[:FeatureMatch.LIMIT]
        for idx, m in enumerate(matches):
            pt1 = kp1[m.queryIdx].pt
            pt2 = kp2[m.trainIdx].pt
            # print(idx, pt1, pt2)
            c1["x"] += pt1[0]
            c1["y"] += pt1[1]
            c2["x"] += pt2[0]
            c2["y"] += pt2[1]
        dx = round((c1["x"] - c2["x"]) / float(FeatureMatch.LIMIT))
        dy = round((c1["y"] - c2["y"]) / float(FeatureMatch.LIMIT))
        l2 = 0
        for idx, m in enumerate(matches):
            pt1 = kp1[m.queryIdx].pt
            pt2 = kp2[m.trainIdx].pt
            l2 = (pt2[0] + dx - pt1[0]) ** 2 + (pt2[1] + dy - pt1[1]) ** 2
        return dx, dy, l2

    def compose(self, img_org_path, img_cropped_path, saved_path, delta=None):
        img_org = cv2.imread(img_org_path)
        img_cropped = cv2.imread(img_cropped_path)
        reduced = self.reduce(img_org, img_cropped, delta)
        cv2.imwrite(saved_path, reduced)

    def reduce(self, img_org, img_cropped, delta=None):
        if not delta:
            gray_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)  # todo try
            gray_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
            debug_show("", gray_cropped)
            dx, dy, l2 = self.match(gray_org, gray_cropped)
        else:
            dx, dy = delta
        h, w = img_cropped.shape[:2]
        h_org, w_org = img_org.shape[:2]
        if h + dy > h_org or w + dx > w_org:
            raise Exception("bad match")
        img_org_copy = img_org.copy()
        img_org_copy[dy:h + dy, dx:w + dx] = img_cropped
        return img_org_copy
