'''
Target: Transfer kinds of medical images, such as hdr, nii, mha, mhd, raw and so on, and store them as hdf5 files
Created on Dec. 2, 2017
Author: Yanyun Jiang
'''

import SimpleITK as sitk

from multiprocessing import Pool
import os
import h5py
import numpy as np

Path_Image = 'D:\jyy--IDEA\experiment\python\jyy\yytest'



d1 = 64
d2 = 64
d3 = 64
dim_Patch = [d1,d2,d3]


'''
This is useed to generate hdf5 database
'''


def Tohdf5(matImgA, matImgB, matdiffAB, id):

    print 'size of imgA:', matImgA.shape
    print 'size of imgB:', matImgB.shape
    print 'size of imgAB:', matdiffAB.shape

    trainImg = np.zeros([10000, 3, d1, d2, d3], dtype=np.float32)
    trainImgA = np.zeros([10000, 1, d1, d2, d3], dtype=np.float32)
    trainImgB = np.zeros([10000, 1, d1, d2, d3], dtype=np.float32)


    trainImg[0, 0, :, :, :] = matImgA
    trainImg[0, 1, :, :, :] = matImgB
    trainImg[0, 2, :, :, :] = matdiffAB

    trainImgA = matImgA
    trainImgB = matImgB

    print 'trainImg shape, ', trainImg.shape


    with h5py.File('Data/Subject_%02d_%d.h5' % (AID, BID), 'w') as f:
        f['dataSubjectImg'] = trainImg
        f['dataImgA'] = trainImgA
        f['dataImgB'] = trainImgB

    with open('Data/trainSet_list.txt', 'a') as f:
        f.write('Data/Subject_%02d_%d.h5\n' % (fileID, warpID))


def main():
    File_ImageA='orig01.mha'
    File_ImageB='orig02.mha'
    Filepath_ImgB=os.path.join(Path_Image,File_ImageB)
    img_B=sitk.ReadImage(Filepath_ImgB, sitk.sitkFloat32)
    mat_B = sitk.GetArrayFromImage(img_B)

    muB = np.mean(mat_B)
    maxVB = np.max(mat_B)
    minVB = np.min(mat_B)
    mat_B = mat_B / (maxVB - minVB)

    Filepath_ImgA=os.path.join(Path_Image, File_ImageA)
    img_A = sitk.ReadImage(Filepath_ImgA, sitk.sitkFloat32)
    mat_A = sitk.GetArrayFromImage(img_A)

    muA = np.mean(mat_A)
    maxVA = np.max(mat_A)
    minVA = np.min(mat_A)
    mat_A = mat_A / (maxVA - minVA)

    mat_diffAB = mat_A - mat_B

    Cnt = Tohdf5(mat_A, mat_B, mat_diffAB, id)
    print '# of patches is ', Cnt


if __name__ == '__main__':
    main()
