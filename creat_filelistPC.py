# coding=utf-8
"""
递归生成文件夹中文件名，具体区别查看下面代码注释
By SXD
"""
import os

root_img = os.getcwd()  # 获取当前路径
#root_img = '/data4DR'
#data = 'data4DR/DR_Grade_500'  #相对路径
data = '/Users/victor/Project4sxd/DRandDME/data/e_ophtha/MA/Flips' #绝对路径
#path = os.listdir(root_img + '/' + data)  # 显示该路径下所有文件git
path = os.listdir(data)
path.sort()
file = open('MA_e_optha_MA_Flip.txt', 'w')

i = 0

for line in path:
    print(line)
    #str = root_img + '/' + data + '/' + line
    str = data + '/' + line
    str = str.replace('.DS_Store','')  #去除mac系统.DS_Store文件，windows系统注释掉即可
    #print(str)
    #print(os.listdir(data))
    for child in os.listdir(data):

        str1 = data + '/' + line + '/' + child;
        d = ' %s' % (i)
        t = str1 + d
        file.write(child + '\n')   #递归生成各个文件夹里的文件名，不含路径
        #file.write(t + '\n')    #递归生成各个文件夹里的文件名，含路径,含0，1，2等标号
        #file.write(str1+'\n')   #递归生成各个文件夹里的文件名，含路径,不含0，1，2等标号

    i = i + 1

file.close()