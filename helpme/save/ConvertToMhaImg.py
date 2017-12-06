# coding=utf-8

import SimpleITK as sitk

def imsave(fname, arr):
    sitk_img = sitk.GetImageFromArray(arr, isVector=False)
    sitk.WriteImage(sitk_img,fname)

if __name__ == '__main__':
    arrD = sitk.ReadImage('l1.hdr')
    image_array = sitk.GetArrayFromImage(arrD)

    print type(arrD)
    print image_array.shape
    print image_array.ndim,image_array

    imsave('ssss2.mha',image_array)
