"""
3D to 2D
Created on Mar 12 2018

@author: Yanyun Jiang
"""

import SimpleITK as sitk
import os
import numpy as np
import cv2
from PIL import Image

Path_orig='D:\yanyun\experiment\python\ytest\datatem'
Path_slice='D:\yanyun\experiment\python\ytest\datatem'


def main():
    for id in range(2, 3):
        File_Image = 'B%02d_Warped.mha'% (id)
        Filepath_Img = os.path.join(Path_orig, File_Image)
        img = sitk.ReadImage(Filepath_Img, sitk.sitkFloat32)
        mat = sitk.GetArrayFromImage(img)
        [sz1, sz2, sz3] = mat.shape

        mat_slice_i = np.zeros([sz2, sz2], dtype=np.float64)
        mat_slice_j = np.zeros([sz1, sz3], dtype=np.float64)
        mat_slice_k = np.zeros([sz1, sz2], dtype=np.float64)
        mat_slice_i[:,:] = mat[90,:,:]
        mat_slice_j[:,:] = mat[:,102,:]
        mat_slice_k[:,:] = mat[:,:,100]


        mat_slice_i = mat_slice_i[::-1]
        mat_slice_j = mat_slice_j[::-1]
        mat_slice_k = mat_slice_k[::-1]
        mat_slice = np.concatenate((mat_slice_i,mat_slice_j,mat_slice_k),axis=0)

        img_slice_i = sitk.GetImageFromArray(mat_slice_i)
        img_slice_j = sitk.GetImageFromArray(mat_slice_j)
        img_slice_k = sitk.GetImageFromArray(mat_slice_k)

        # caster = sitk.CastImageFilter()
        # caster.SetOutputPixelType(sitk.sitkInt16)
        # outimg = caster.Execute(img_slice)

        outputfilename = 'slice_i_90%02d.jpg'% (id)
        dataOutputPath = os.path.join(Path_slice, outputfilename)
        cv2.imwrite(dataOutputPath, mat_slice_i)

        outputfilename = 'slice_j_102%02d.jpg'% (id)
        dataOutputPath = os.path.join(Path_slice, outputfilename)
        cv2.imwrite(dataOutputPath, mat_slice_j)

        outputfilename = 'slice_k_100%02d.jpg'% (id)
        dataOutputPath = os.path.join(Path_slice, outputfilename)
        cv2.imwrite(dataOutputPath, mat_slice_k)

        outputfilename = 'slice_Bwarped%02d.jpg'% (id)
        dataOutputPath = os.path.join(Path_slice, outputfilename)
        cv2.imwrite(dataOutputPath, mat_slice)



if __name__ == '__main__':
    main()