# coding=utf-8
import xlrd
import os, sys
import xlutils.copy
import numpy as np
import pyExcelerator
import shutil

#data = xlrd.open_workbook('data4DR/trainLabels4sxd.xlsx')
data = xlrd.open_workbook('/Users/victor/Project4sxd/DRandDME/code/challenge/SubChallenge2-DRandDMEGrading/GROUNDTRUTH_LABELS/IDRiD_Training_Set.xlsx')
table = data.sheets()[0]          #通过索引顺序获取
nrows = table.nrows
print nrows
nameImage = table.col_values(0)

# def __init__(self):
#     self.path = '../../Diabetic_Retinopathy/data/dataset2kaggle/original/1/'
#     self.newpath = 'data4DR/'

def drimgclass():

    #path = '../../Diabetic_Retinopathy/data/dataset2kaggle/original/1/'
    #path = '/data4DR/test/'
    path = '/Users/victor/code4suixiaodan/Mask_RCNN-master/code4work/TrainingSetSub2-crop/'
    newpath = '/Users/victor/code4suixiaodan/Mask_RCNN-master/code4work/TrainingSetSub2-crop-cnn/'
    for i in range(1,nrows+1):
        print i
        nvalue = table.cell(i-1,1).value
        if nvalue == 0.0:
            imgName = table.cell(i-1,0).value
            try:
                  filedir = os.path.join(path, imgName+'.jpg')
                  shutil.copy(str(filedir), os.path.join(newpath+'0', imgName+'CNN.jpg'))
            except:
                  print ("error, calss 0, maybe the file " + str(imgName) + "CNN.jpg is not exited!")

        if nvalue == 1.0:
            imgName = table.cell(i-1,0).value
            try:
                  filedir = os.path.join(path, imgName+'.jpg')
                  shutil.copy(str(filedir), os.path.join(newpath+'1', imgName+'CNN.jpg'))
            except:
                  print ("error, calss 1, maybe the file " + str(imgName) + "CNN.jpg is not exited!")

        if nvalue == 2.0:
            imgName = table.cell(i-1,0).value
            try:
                  filedir = os.path.join(path, imgName+'.jpg')
                  shutil.copy(str(filedir), os.path.join(newpath+'2', imgName+'CNN.jpg'))
            except:
                  print ("error, calss 2, maybe the file " + str(imgName) + "CNN.jpg is not exited!")

        if nvalue == 3.0:
            imgName = table.cell(i-1,0).value
            try:
                  filedir = os.path.join(path, imgName+'.jpg')
                  shutil.copy(str(filedir), os.path.join(newpath+'3', imgName+'CNN.jpg'))
            except:
                  print ("error, calss 3, maybe the file " + str(imgName) + "CNN.jpg is not exitd!")

        if nvalue == 4.0:
            imgName = table.cell(i-1,0).value
            try:
                  filedir = os.path.join(path, imgName+'.jpg')
                  shutil.copy(str(filedir), os.path.join(newpath+'4', imgName+'CNN.jpg'))
            except:
                  print ("error, calss 4, maybe the file " + str(imgName) + "CNN.jpg is not exited!")



def main():

    drimgclass()

if __name__ == '__main__':
    main()
    #np.savetxt('nameImage', list(nameImage))
    print "OK!"