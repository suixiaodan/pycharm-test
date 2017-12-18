# coding=utf-8
import xlrd
import os, sys
import xlutils.copy
import numpy as np
import pyExcelerator
import shutil

#data = xlrd.open_workbook('data4DR/trainLabels4sxd.xlsx')
data = xlrd.open_workbook('/Volumes/Untitled/kaggle4DR/trainLabels5sdnu.xlsx')
table = data.sheets()[0]          #通过索引顺序获取
nrows = table.nrows

nameImage = table.col_values(0)

# def __init__(self):
#     self.path = '../../Diabetic_Retinopathy/data/dataset2kaggle/original/1/'
#     self.newpath = 'data4DR/'

def drimgclass():
    a = 0
    b = 0
    c = 0
    d = 0
    e = 0
    #path = '../../Diabetic_Retinopathy/data/dataset2kaggle/original/1/'
    path = '/Volumes/Untitled/kaggle4DR/train'
    newpath = '/Volumes/Untitled/kaggle4DR/val'
    for i in range(1,nrows+1):
        nvalue = table.cell(i-1,1).value

        if nvalue == 0.0:
            a = a + 1
            imgName = table.cell(i-1,0).value
            if a < 4338:
               try:
                  filedir = os.path.join(path, imgName)
                  shutil.copy(str(filedir), os.path.join(newpath+'0', imgName))
               except:
                  print ("error, calss 0, maybe the file " + str(imgName) + " is not exited!")
            elif a == 4338:
                try:
                    filedir = os.path.join(path, imgName)
                    shutil.copy(str(filedir), os.path.join(newpath + '0', imgName))
                    print ("class 0 = 4338 ")
                except:
                    print ("error, calss 0, maybe the file " + str(imgName) + " is not exited!")
            else:
                continue

        if nvalue == 1.0:
            b = b + 1
            imgName = table.cell(i-1,0).value
            if b < 2500:
               try:
                  filedir = os.path.join(path, imgName)
                  shutil.copy(str(filedir), os.path.join(newpath+'1', imgName))
               except:
                  print ("error, calss 1, maybe the file " + str(imgName) + " is not exited!")
            elif b == 2500:
               try:
                  filedir = os.path.join(path, imgName)
                  shutil.copy(str(filedir), os.path.join(newpath + '1', imgName))
                  print ("class 1 = 2500 ")
               except:
                  print ("error, calss 1, maybe the file " + str(imgName) + " is not exited!")
            else:
               continue

        if nvalue == 2.0:
            c = c + 1
            imgName = table.cell(i-1,0).value
            if c < 2500:
               try:
                  filedir = os.path.join(path, imgName)
                  shutil.copy(str(filedir), os.path.join(newpath+'2', imgName))
               except:
                  print ("error, calss 2, maybe the file " + str(imgName) + " is not exited!")
            elif c == 2500:
               try:
                  filedir = os.path.join(path, imgName)
                  shutil.copy(str(filedir), os.path.join(newpath + '2', imgName))
                  print ("class 2 = 2500 ")
               except:
                  print ("error, calss 2, maybe the file " + str(imgName) + " is not exited!")
            else:
               continue

        if nvalue == 3.0:
            d = d + 1
            imgName = table.cell(i-1,0).value
            if d < 449:
               try:
                  filedir = os.path.join(path, imgName)
                  shutil.copy(str(filedir), os.path.join(newpath+'3', imgName))
               except:
                  print ("error, calss 3, maybe the file " + str(imgName) + " is not exited!")
            elif d == 449:
               try:
                  filedir = os.path.join(path, imgName)
                  shutil.copy(str(filedir), os.path.join(newpath + '3', imgName))
                  print ("class 3 = 449 ")
               except:
                  print ("error, calss 3, maybe the file " + str(imgName) + " is not exited!")
               else:
                   continue

        if nvalue == 4.0:
            e = e + 1
            imgName = table.cell(i-1,0).value
            if e < 213:
               try:
                  filedir = os.path.join(path, imgName)
                  shutil.copy(str(filedir), os.path.join(newpath+'4', imgName))
               except:
                  print ("error, calss 4, maybe the file " + str(imgName) + " is not exited!")
            elif e == 213:
               try:
                  filedir = os.path.join(path, imgName)
                  shutil.copy(str(filedir), os.path.join(newpath + '4', imgName))
                  print ("class 4 = 213 ")
               except:
                  print ("error, calss 4, maybe the file " + str(imgName) + " is not exited!")
            else:
               continue

def removefile():
    for i in range(3, nrows+1):
        nvalue = table.cell(i-1,1).value
        #if nvalue == u'\x4E0Dx786Ex5B9A':
        if nvalue == 'no':
            nvalue = table.cell(i-1,0).value
            try:
                nameImage.remove(nvalue)
                print "Successfully delete " + str(nvalue) + " from excel!"
            except:
                print ("error, maybe the file " + str(nvalue) + " is deleted!")
            try:
                os.remove('data4DR/test/'+nvalue+'')
                print "Successfully delete " + str(nvalue) + " !"
            except:
                print("error, maybe the file " + str(nvalue) + " is not existed!")

            #table.Rows(i-1).Delete()
            # xlsxwrite(i-1,'')


def savetxtsxd():
    arr1 = np.array(nameImage)
     #print arr2[2]
     # np.savetxt('001.txt',arr2)
    file = open('001.txt','w')  #可以工作，但是数据不符合要求
    for m in range(0, arr1.size):
        file.write(str(arr1[m]))   #可以工作，但是数据不符合要求
        file.write('\n')
    file.close()

def xlsxwrite():
    wb = pyExcelerator.Workbook()
    ws = wb.add_sheet('nameImage')
    arr1 = np.array(nameImage)
    for m in range(0,arr1.size):
        ws.write(m,0,arr1[m])
    #保存该excel文件,有同名文件时直接覆盖
    wb.save('data4DR/mini.xls')
    print '创建excel文件完成！'
    str = nameImage

def main():
    # removefile()
    # xlsxwrite()
    # savetxtsxd()
    drimgclass()

if __name__ == '__main__':
    main()
    #np.savetxt('nameImage', list(nameImage))
    print "OK!"