# coding=utf-8

import os

root_img = os.getcwd()  # 获取当前路径
#root_img = '/data4DR'
data = '/Users/victor/code4suixiaodan/Mask_RCNN-master/coco/DRDataset/images/'
path = os.listdir(root_img + '/' + data)  # 显示该路径下所有文件
path.sort()
file = open('filelist.txt', 'w')

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