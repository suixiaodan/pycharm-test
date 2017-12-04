# coding=utf-8
import xlrd
import os, sys
import xlutils.copy

reload(sys)
sys.setdefaultencoding( "utf-8" )

data = xlrd.open_workbook('ValDR500.xlsx')
table = data.sheets()[0]          #通过索引顺序获取
nrows = table.nrows
wb = xlutils.copy.copy(data)
nameImage = table.col_values(0)
#nvalue = table.cell(i-1,0).value
def del_files(path):
    print 3
    i = 1
    for root,dirs,files in os.walk(path):
        for name in files:
            if name in nameImage:
                os.remove(os.path.join(root, name))
                print("Delete File : " + os.path.join(root, name))
                i = i+1
    print i
# test
if __name__ == "__main__":
    path = '/shenlab/lab_stor4/sxd/data/DRclass4kaggle/'
    print 1
    del_files(path)
    print 2