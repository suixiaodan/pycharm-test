
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

Path_Image = '/Users/victor/code4suixiaodan/IDEA/code/pycharm-test/helpme/patch/'



d1 = 64
d2 = 64
d3 = 64
dim_Patch = [d1,d2,d3]


'''
This is useed to generate hdf5 database
'''


def Tohdf5(matImgA, matImgB, matdiffAB, id, jth):

    print 'size of imgA:', matImgA.shape
    print 'size of imgB:', matImgB.shape
    print 'size of imgAB:', matdiffAB.shape

    trainImg = np.zeros([10000, 3, d1, d2, d3], dtype=np.float32)
    trainImg_A = np.zeros([10000, 1, d1, d2, d3], dtype=np.float32)
    trainImg_B = np.zeros([10000, 1, d1, d2, d3], dtype=np.float32)


    trainImg[0, 0, :, :, :] = matImgA
    trainImg[0, 1, :, :, :] = matImgB
    trainImg[0, 2, :, :, :] = matdiffAB

    trainImg_A = matImgA
    trainImg_B = matImgB

    print 'trainImg shape, ', trainImg.shape


    with h5py.File('Data/Subject_%02d_%d.h5' %(id, jth), 'w') as f:
        f['dataSubjectImg'] = trainImg
        f['dataImgA'] = trainImg_A
        f['dataImgB'] = trainImg_B

    with open('Data/trainSet_list.txt', 'a') as f:
        f.write('Data/Subject_%02d_%d.h5\n' %(id, jth))


def main():
    for id in range(2,3):
        for jth in range(1,7):
            File_ImageA = 'patch%02d.mha' % (id)
            File_ImageB = 'patch%02d.mha' % (jth)
            Filepath_ImgB = os.path.join(Path_Image, File_ImageB)
            img_B = sitk.ReadImage(Filepath_ImgB, sitk.sitkFloat32)
            mat_B = sitk.GetArrayFromImage(img_B)

            muB = np.mean(mat_B)
            maxVB = np.max(mat_B)
            minVB = np.min(mat_B)
            mat_B = mat_B / (maxVB - minVB)

            Filepath_ImgA = os.path.join(Path_Image, File_ImageA)
            img_A = sitk.ReadImage(Filepath_ImgA, sitk.sitkFloat32)
            mat_A = sitk.GetArrayFromImage(img_A)

            muA = np.mean(mat_A)
            maxVA = np.max(mat_A)
            minVA = np.min(mat_A)
            mat_A = mat_A / (maxVA - minVA)

            mat_diffAB = mat_A - mat_B

            Cnt = Tohdf5(mat_A, mat_B, mat_diffAB, id, jth)
            print '# of patches is ', Cnt


if __name__ == '__main__':
    main()

