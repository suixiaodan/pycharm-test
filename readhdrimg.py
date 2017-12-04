# coding=utf-8
from medpy.io import load,save

# import itkImagePython

image_data, image_header = load('23_right.jpeg')
print image_data.shape
print image_data.dtype
save(image_data,'data4DR/l1.hdr',image_header)