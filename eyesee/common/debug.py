# -*- coding: utf-8 -*-
import sys
sys.path.append("..")

import cv2 as cv
from common.config import DEBUG_SW


def debug_show(winname, img):
    if DEBUG_SW:
        cv.imshow(winname, img)
        cv.waitKey(0)