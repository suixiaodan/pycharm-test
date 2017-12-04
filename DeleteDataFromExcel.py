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
#print nameImage

def removefile():
    for i in range(1, nrows+1):
        nvalue = table.cell(i-1,0).value
        try:
            os.remove('../../shenlab/lab_stor4/sxd/data/DRclass4kaggle/'+nvalue+'')
            print "Successfully delete " + str(nvalue) + " !"
        except:
            print("error, maybe the file " + str(nvalue) + " is not existed!")

def main():
    removefile()

if __name__ == '__main__':
    main()
    print "OK!"