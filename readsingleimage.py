# coding=utf-8
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np


mnist = mpimg.imread('001.jpg')

plt.imshow(mnist) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.show()