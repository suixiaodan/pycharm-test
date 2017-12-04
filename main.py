# coding=utf-8
import os
import skimage
import SimpleITK as sitk
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np


mnist = mpimg.imread('460_right.jpeg')

# plt.imshow(mnist) # 显示图片
# plt.axis('off') # 不显示坐标轴
# plt.show()

# def imread(fname):
#     sitk_img = sitk.ReadImage(fname)
#     return sitk.GetArrayFromImage(sitk_img)

def imsave(fnamed, arrz):
#    sitk_img = sitk.GetImageFromArray(arrz, isVector=True)
    sitk_img = sitk.GetImageFromArray(arrz)
    sitk.WriteImage(sitk_img, fnamed)

if __name__ == '__main__':
    imsave('460_right.hdr', mnist)
    print("OK!")