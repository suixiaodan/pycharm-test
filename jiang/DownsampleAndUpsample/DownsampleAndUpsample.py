"""
Downsample and Upsample
Created on June 12 2018

@author: Xiaodan Sui
"""

import SimpleITK as sitk
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

import cv2

Path_orig='/Users/victor/code4suixiaodan/IDEA/code/pycharm-test/jiang/DownsampleAndUpsample'
Path_slice='/Users/victor/code4suixiaodan/IDEA/code/pycharm-test/jiang/DownsampleAndUpsample'

# def Downsample(mat, poolSize=3, poolStride=2):
#
#     # inputMap sizes
#     [sz1, sz2, sz3] = mat.shape
#
#     # outputMap sizes

def upsample(inputMap, upsampleSize=2, upsampleStride=2, mode='linearInterpolation'):

    in_row, in_col, in_vol = np.shape(inputMap)
    #in_row, in_col, in_vol = 180, 216, 180

    print in_row, in_col, in_vol, "ok2"

    out_row, out_col, out_vol = int(np.floor(in_row * upsampleStride)), int(np.floor(in_col * upsampleStride)), int(np.floor(in_vol * upsampleStride))
    # row_remainder, col_remainder, vol_remainder = np.mod(in_row, poolStride), np.mod(in_col, poolStride), np.mod(in_vol,
    #                                                                                                              poolStride)
    # if row_remainder != 0:
    #     out_row += 1
    # if col_remainder != 0:
    #     out_col += 1
    # if vol_remainder != 0:
    #     out_vol += 1
    print "out_row", out_row, "out_col", out_col, "out_vol", out_vol

    outputMap = np.zeros((out_row, out_col, out_vol))
    print outputMap.shape

    # upsample
    for c_idx in range(0, in_col):
        for r_idx in range(0, in_row):
            for v_idx in range(0, in_vol):
                startX = v_idx * upsampleStride
                startY = r_idx * upsampleStride
                startZ = c_idx * upsampleStride
                poolvalue = inputMap[r_idx, c_idx, v_idx]
                outputMap[startY:startY + upsampleSize, startZ:startZ + upsampleSize, startX:startX + upsampleSize] = poolvalue
                # poolField = inputMap[startY:startY + poolSize, startZ:startZ + poolSize, startX:startX + poolSize]
                # print poolField, v_idx, r_idx, c_idx,startX, startY
                # poolOut = np.max(poolField)
                # print poolOut
                # outputMap[r_idx, c_idx, v_idx] = poolOut

    # retrun outputMap
    return outputMap

def pooling(inputMap, poolSize=2, poolStride=2, mode='max'):
    """INPUTS:
              inputMap - input array of the pooling layer
              poolSize - X-size(equivalent to Y-size) of receptive field
              poolStride - the stride size between successive pooling squares

       OUTPUTS:
               outputMap - output array of the pooling layer
    """
    # inputMap sizes
    in_row, in_col, in_vol = np.shape(inputMap)
    #in_row, in_col, in_vol = 180, 216, 180

    print in_row, in_col, in_vol, "ok"


    # outputMap sizes
    out_row, out_col, out_vol = int(np.floor(in_row / poolStride)), int(np.floor(in_col / poolStride)), int(np.floor(in_vol / poolStride))
    row_remainder, col_remainder, vol_remainder = np.mod(in_row, poolStride), np.mod(in_col, poolStride), np.mod(in_vol, poolStride)
    if row_remainder != 0:
        out_row += 1
    if col_remainder != 0:
        out_col += 1
    if vol_remainder != 0:
        out_vol += 1
    print "out_row", out_row, "out_col", out_col, "out_vol", out_vol
    outputMap = np.zeros((out_row, out_col, out_vol))
    print outputMap.shape
    # # padding
    # temp_map = np.lib.pad(inputMap, ((0, poolSize - row_remainder), (0, poolSize - col_remainder), (0, poolSize - vol_remainder)), 'edge')
    # temp_map = np.zeros((in_row, in_col, in_vol))
    # print temp_map.shape, "over"

    # max pooling
    for c_idx in range(0, out_col):
        for r_idx in range(0, out_row):
            for v_idx in range(0, out_vol):
                startX = v_idx * poolStride
                startY = r_idx * poolStride
                startZ = c_idx * poolStride
                poolField = inputMap[startY:startY + poolSize, startZ:startZ + poolSize, startX:startX + poolSize]
                # print poolField, v_idx, r_idx, c_idx,startX, startY
                poolOut = np.max(poolField)
                # print poolOut
                outputMap[r_idx, c_idx, v_idx] = poolOut

    # retrun outputMap
    return outputMap

def imsave(fname, arr):
    sitk_img = sitk.GetImageFromArray(arr, isVector=False)
    sitk.WriteImage(sitk_img,fname)

def main():

    File_Image = 'c1.hdr'
    Up_File_Image = 'test5.mha'
    Filepath_Img = os.path.join(Path_orig, File_Image)
    Up_Filepath_Img = os.path.join(Path_orig, Up_File_Image)
    img = sitk.ReadImage(Filepath_Img, sitk.sitkFloat32)
    up_img = sitk.ReadImage(Up_Filepath_Img, sitk.sitkFloat32)
    mat = sitk.GetArrayFromImage(img)
    up_mat = sitk.GetArrayFromImage(up_img)
    [sz1, sz2, sz3] = mat.shape
    print(sz1, sz2, sz3)
    mat = mat[:180, :216, :180]
    #test_result = pooling(mat, 2, 2, 'max')
    #print(test_result.shape)

    up_test_result = upsample(up_mat, 2, 2, 'linearInterpolation')
    # img_slice = sitk.GetImageFromArray(test_result)
    outputfilename = 'test6.mha'
    dataOutputPath = os.path.join(Path_slice, outputfilename)

    #mat = mat[:,120, :]
    imsave(dataOutputPath, up_test_result)


if __name__ == '__main__':
    main()
