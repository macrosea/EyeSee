# -*- coding: utf-8 -*-
from os import path
import shutil

import numpy as np
import cv2 as cv2

import urllib3
from common.config import WM_SERVER


class WmClient:
    def __init__(self):
        self.http = urllib3.PoolManager()
        self.host = WM_SERVER

    def detect(self, img_path):
        req_path = '/watermark/detect'
        basename = path.basename(img_path)
        extname = img_path.split(".")[1]
        print("detect: " + img_path)
        with open(img_path, 'rb') as fp:
            file_data = fp.read()
            boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
            r = self.http.request(
                'POST',
                self.host + req_path,
                headers={},
                multipart_boundary=boundary,
                fields={
                    'pic': (basename, file_data, "image/"+extname)
                })
            ret = r.status, r.data
            r.release_conn()
            return ret

    def embed(self, img_path, saved_dir):
        req_path = '/watermark/embed'
        basename = path.basename(img_path)
        extname = img_path.split(".")[1]
        saved_path = path.join(saved_dir, basename)

        with open(img_path, 'rb') as fp, open(saved_path, 'wb') as out_file:
            file_data = fp.read()
            boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
            r = self.http.request(
                'POST',
                self.host + req_path,
                headers={},
                multipart_boundary=boundary,
                fields={
                    'pic': (basename, file_data, "image/"+extname)
                })
            if r.status == 200:
                #shutil.copyfileobj(r.data, out_file)
                out_file.write(r.data)
            r.release_conn()
        return saved_path if path.isfile(saved_path) else None


#
# def sendtoserver(frame):
#     import http.client as httplib
#     conn = httplib.HTTPConnection('127.0.0.1', timeout=5)
#     imencoded = cv2.imencode(".jpg", frame)[1]
#     headers = {"Content-type": "text/plain"}
#     try:
#         conn.request("POST", "/", imencoded.tostring(), headers)
#         response = conn.getresponse()
#     except conn.timeout as e:
#         print("timeout")
#
#     return response

