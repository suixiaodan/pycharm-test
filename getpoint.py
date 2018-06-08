# coding=utf-8
"""
Create by 2015-10-25

@author: zhouheng

In this function you can get the position of the element
that you want in the matrix.
"""

import numpy as np


def getPositon(a):


    (x, y) = a.shape  # get the matrix of a raw and column

    #_positon = np.argmax(a)  # get the index of max in the a
    # print _positon
    #num = 5
    num_index = np.argwhere(a == 5)
    return num_index
    # m, n = divmod(_positon, column)
    # print "The raw is ", m
    # print "The column is ", n
    # print "The max of the a is ", a[m, n]

a = np.array([[2, 5, 7, 8, 9, 89], [6, 7, 5, 4, 6, 4]])
y=getPositon(a)
print(y)
print(type(y))
print(y[1])
print(len(y))