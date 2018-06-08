# coding=utf-8
"""
递归重新命名文件名，具体区别查看下面代码注释
By SXD
"""
import os
# 列出当前目录下所有的文件
#files = os.listdir(".")
files = '/Users/victor/code4suixiaodan/IDEA/code/pycharm-test/rename/'
print(files)
for filename in files:
    print(filename)
    portion = os.path.splitext(filename)
    print(portion[0])
    print(portion[1])
    # 如果后缀是.txt
    #rename = portion[1].strip('JPG')
    rename = portion[1] + '_EX'
    rename = rename + 'jpg'# manual1
    print(rename)
        # 重新组合文件名和后缀名
    newname = rename + portion[1]
    print(newname)
    os.rename(filename,newname)