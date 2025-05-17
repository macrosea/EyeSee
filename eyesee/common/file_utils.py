from os import path, makedirs
from os import listdir, system
from os import mkdir
import shutil


# def get_file_name(file_path):
#     path.splitext(file_path)[0]


def create_dir(dir_):
    if path.exists(dir_):
        shutil.rmtree(dir_)
    makedirs(dir_)

