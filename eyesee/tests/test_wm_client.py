#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
sys.path.append("..")

from os import path, listdir
import threading
import random
from wm_client.wm_client import WmClient


# from img_processor.feature_match import FeatureMatch
# from tests.helper import save_rect_crop
# from search.search import Search


def test_wm(client, img_path):
    # res = client.detect(img_path)
    # print(res)
    res = client.embed(img_path, "/tmp/")
    print(res if res else "failed ... ")
    # res = client.detect(res)
    # print(res)


class TestThread (threading.Thread):
    def __init__(self, name, client, scan_dir):
        threading.Thread.__init__(self, name=name)
        self.client = client
        self.scan_dir = scan_dir

    def run(self):
        img_paths = [path.join(self.scan_dir, f) for f in listdir(self.scan_dir) if f.endswith('.jpg')]
        random.shuffle(img_paths)
        for idx, itm in enumerate(img_paths):
            test_wm(self.client, itm)

#
# def img_crop(client):
#     img_path = "/tmp/ws/work/watermark/sets_verify/vegan-pasta_KRX6BEOKGM.jpg"
#     res = client.embed(img_path, "/tmp/")
#     cropped = save_rect_crop(res, "/tmp/wm_test")
#     search = Search()
#     res = search.search(cropped)
#     fm = FeatureMatch()
#     saved_path = "/tmp/saved.jpg"
#     fm.compose(res, cropped, saved_path)
#     res = client.detect(saved_path)
#     print(res)


def test_stress(client, scan_dir):
    threads = []
    for i in range(20):
        thr = TestThread("job-{}".format(i), client, scan_dir)
        thr.start()
        print(thr.getName())
        threads.append(thr)

    for t in threads:
        t.join()
        print(t.getName())
    print("Exiting  Thread")

if __name__ == '__main__':
    cli = WmClient()
    #img_path = "/tmp/ws/work/watermark/sets_verify/woman-shop_E1KVAKFMUK.jpg"
    #test_wm(cli, img_path)
    test_stress(cli, "/tmp/ws/work/watermark/sets_train")
    pass
