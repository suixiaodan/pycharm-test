# coding=utf-8

import SimpleITK as sitk
import os

Path_orig='D:\yanyun\experiment\Data\BrainReg\Atlases_MNIspace'
Path_mha='D:\yanyun\experiment\Data\BrainReg\Data'

def imsave(fname, arr):
    sitk_img = sitk.GetImageFromArray(arr, isVector=False)
    sitk.WriteImage(sitk_img,fname)


def main():
    for id in range(1,41):
        File_Subimage='l%0d.hdr'%(id)
        FilePath_Subimage = os.path.join(Path_orig, File_Subimage)
        arrD = sitk.ReadImage(FilePath_Subimage)
        image_array = sitk.GetArrayFromImage(arrD)

        print type(arrD)
        print image_array.shape
        print image_array.ndim, image_array

        outputfilename='na%02d_label.mha'%(id)
        dataOutputPath = os.path.join(Path_mha,outputfilename)

        imsave(dataOutputPath, image_array)

if __name__ == '__main__':
    main()
