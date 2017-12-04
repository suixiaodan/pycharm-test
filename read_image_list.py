# coding=utf-8
import os
import skimage.io as io
import numpy as np
# from skimage import data_dir
print os.getcwd()+'../IDEA/Data'
str='/*.hdr'
coll = io.ImageCollection(os.getcwd()+'\sample\*.hdr')
#arr = np.array(coll)
print(len(coll))
io.imshow(coll[5])
io.show()