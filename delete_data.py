# coding=utf-8
import xlrd
import os, sys
import xlutils.copy
import numpy as np
import pyExcelerator

reload(sys)
sys.setdefaultencoding( "utf-8" )
#from xlwt import Style
#import xlwt
# import win32com.client
# import pyExcelerator
#创建workbook和sheet对象

# reload(sys)
# sys.setdefaultencoding('utf8')

data = xlrd.open_workbook('data4DR/test.xlsx')
table = data.sheets()[0]          #通过索引顺序获取
nrows = table.nrows
wb = xlutils.copy.copy(data)
# xlApp = win32com.client.Dispatch('Excel.Application')  #打开EXCEL，这里不需改动
# xlBook = xlApp.Workbooks.Open('test.xls')             #将D:\\1.xls改为要处理的excel文件路径
# table = xlBook.Worksheets('sheet1')                    #要处理的excel页，默认第一页是‘sheet1’
nameImage = table.col_values(0)
nameImage.remove('')
nameImage.remove('')
#print nameImage

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
    removefile()
    xlsxwrite()
    savetxtsxd()

if __name__ == '__main__':
    main()
    #np.savetxt('nameImage', list(nameImage))
    print "OK!"