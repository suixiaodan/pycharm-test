# coding=utf-8
import xlrd
import os, sys
import xlutils.copy
import numpy as np
import pyExcelerator
import shutil

def copyFiles():

    path = '/Users/victor/code4suixiaodan/Mask_RCNN-master/code4work/ImageSynthesis/4lesions/temp/total/'
    newpath = '/Users/victor/code4suixiaodan/Mask_RCNN-master/code4work/ImageSynthesis/4lesions/temp/'

    file_names = next(os.walk(path))[2]

    images_num = len(file_names)

    for i in range(32):
        shutil.copy(os.path.join(path, file_names[i]), os.path.join(newpath + '1HE', file_names[i]))

    for j in range(32,63):
        shutil.copy(os.path.join(path, file_names[j]), os.path.join(newpath + '2HE', file_names[j]))

    for k in range(63,73):
        shutil.copy(os.path.join(path, file_names[k]), os.path.join(newpath + '3HE', file_names[k]))

    for l in range(73,83):
        shutil.copy(os.path.join(path, file_names[l]), os.path.join(newpath + '4HE', file_names[l]))

    for m in range(83,93):
        shutil.copy(os.path.join(path, file_names[m]), os.path.join(newpath + '5HE', file_names[m]))

    for n in range(93,103):
        shutil.copy(os.path.join(path, file_names[n]), os.path.join(newpath + '6HE', file_names[n]))

copyFiles()