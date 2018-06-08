# coding=utf-8
import xlrd
import os, sys
import xlutils.copy
import numpy as np
import pyExcelerator
import shutil

#data = xlrd.open_workbook('data4DR/trainLabels4sxd.xlsx')
data = xlrd.open_workbook('/Users/victor/Project4sxd/DRandDME/experiment_results/Segmentation_Results/Sub-Challenge3/IDEA-UNC_OD_localization2.xlsx')

table = data.sheets()[0]          #通过索引顺序获取
nrows = table.nrows
print nrows
nameImage = table.col_values(0)

def euclidean_dis():
    #dist = 0
    euc = 0
    for i in range(2, nrows + 1):
        x1 = table.cell(i - 1, 1).value
        y1 = table.cell(i - 1, 2).value
        x2 = table.cell(i - 1, 3).value
        y2 = table.cell(i - 1, 4).value
        print(x1,y1,x2,y2)
        dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        euc += dist
    print(euc/413)

euclidean_dis()



