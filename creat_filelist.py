# coding=utf-8

import os

root_img = os.getcwd()  # 获取当前路径
#root_img = '/data4DR'
data = '/Users/victor/Project4sxd/DRandDME/data/ROC_Data/ROC_Mask/train'
path = os.listdir(root_img + '/' + data)  # 显示该路径下所有文件
path.sort()
file = open('IDRID_MA_Flip_train_80.txt', 'w')

i = 0

for line in path:
    str = root_img + '/' + data + '/' + line
    for child in os.listdir(str):
        str1 = data + '/' + line + '/' + child;
        d = ' %s' % (i)
        t = str1 + d
        file.write(t + '\n')
    i = i + 1

file.close()