# coding=utf-8

import SimpleITK as sitk

Path_orig='/Users/victor/code4suixiaodan/IDEA/code/pycharm-test/helpme/orig/'
Path_mha='/Users/victor/code4suixiaodan/IDEA/code/pycharm-test/helpme/origmha/'

def imsave(fname, arr):
    sitk_img = sitk.GetImageFromArray(arr, isVector=False)
    sitk.WriteImage(sitk_img,fname)


def main():
    for id in range(1,7):
        File_Subimage='orig%02d.hdr'%(id)
        FilePath_Subimage = os.path.join(Path_orig, File_Subimage)
        arrD = sitk.ReadImage(FilePath_Subimage)
        image_array = sitk.GetArrayFromImage(arrD)

        print type(arrD)
        print image_array.shape
        print image_array.ndim, image_array

        outputfilename='trans%02d.mha'%(id)
        dataOutputPath = os.path.join(Path_mha,outputfilename)

        imsave(dataOutputPath, image_array)

if __name__ == '__main__':
    main()
