'''

Created on Dec. 2, 2017
Author: Yanyun Jiang
'''


import SimpleITK as sitk

from multiprocessing import Pool
import os
import numpy as np

Path_origmha='/Users/victor/code4suixiaodan/IDEA/code/pycharm-test/helpme/origmha/'
Path_patch='/Users/victor/code4suixiaodan/IDEA/code/pycharm-test/helpme/patch/'

a=80
b=120
c=90
d=64


def main():
    for id in range(1,7):
        File_Image = 'trans%02d.mha'%(id)
        Filepath_Img = os.path.join(Path_origmha, File_Image)
        img_A = sitk.ReadImage(Filepath_Img, sitk.sitkFloat32)
        mat_A = sitk.GetArrayFromImage(img_A)
        mat_B = np.zeros([d, d, d], dtype=np.float64)

        mat_B[:, :, :] = mat_A[a:a + d, b:b + d, c:c + d]
        img_B = sitk.GetImageFromArray(mat_B)

        caster = sitk.CastImageFilter()
        caster.SetOutputPixelType(sitk.sitkInt16)
        outimg = caster.Execute(img_B)

        outputfilename = 'patch%02d.mha'%(id)
        dataOutputPath = os.path.join(Path_patch, outputfilename)
        sitk.WriteImage(outimg, dataOutputPath)

if __name__ == '__main__':
    main()