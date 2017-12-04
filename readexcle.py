# coding=utf-8
import xlrd
import os

b = os.path.exists("data4DR")
if b:
    print "File data4DR Exist!"
else:
    mkdir('data4DR')

def mkfolder():  #creat 5 folders to store data of DR classification
   for i in range(0,5):
        b = os.path.exists("data4DR\\"+str(i)+"\\")
        if b:
            print "File "+str(i)+" Exist!"
        else:
            os.mkdir("data4DR\\"+str(i)+"\\")

data = xlrd.open_workbook('jiao.xlsx')
table = data.sheets()[0]          #通过索引顺序获取
nrows = table.nrows
nvalue = table.cell(0,1).value
print(nvalue)


if __name__ == '__main__':
   mkfolder()
   print('ok')

