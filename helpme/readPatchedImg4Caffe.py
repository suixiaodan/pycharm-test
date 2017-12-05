'''
Target: Transfer kinds of medical images, such as hdr, nii, mha, mhd, raw and so on, and store them as hdf5 files
Created on Feb. 20, 2017
Author: Jingfan Fan
'''

import SimpleITK as sitk

from multiprocessing import Pool
import os
import h5py
import numpy as np

Path_Image='/Users/victor/code4suixiaodan/IDEA/code/pycharm-test/helpme/Data/'
Path_Deform='/Users/victor/code4suixiaodan/IDEA/code/pycharm-test/helpme/Data/'
    

d1 = 64
d2 = 64
d3 = 64
dim_Patch = [d1,d2,d3]
dec = 14
dc1 = 36
dc2 = 36
dc3 = 36
step = [dc1,dc2,dc3]
 
dell = 8   
dl1 = 12
dl2 = 12
dl3 = 12

dem = 12
dm1 = 20
dm2 = 20
dm3 = 20
'''
This is useed to generate hdf5 database
'''
def Tohdf5(matSub, matSubG, matTem, matTemG, matDeform, fileID, warpID):
    [sz1,sz2,sz3]=matSub.shape
    print 'size of img:', matSub.shape
    print 'size of DF:', matDeform.shape

    trainImg=np.zeros([10000,4,d1,d2,d3],dtype=np.float32)
    trainDeform0X=np.zeros([10000,1,dc1,dc2,dc3],dtype=np.float32)    
    trainDeform0Y=np.zeros([10000,1,dc1,dc2,dc3],dtype=np.float32)    
    trainDeform0Z=np.zeros([10000,1,dc1,dc2,dc3],dtype=np.float32)    
        
    trainDeform1X=np.zeros([10000,1,dm1,dm2,dm3],dtype=np.float32)    
    trainDeform1Y=np.zeros([10000,1,dm1,dm2,dm3],dtype=np.float32)    
    trainDeform1Z=np.zeros([10000,1,dm1,dm2,dm3],dtype=np.float32)    
    
    trainDeform2X=np.zeros([10000,1,dl1,dl2,dl3],dtype=np.float32)    
    trainDeform2Y=np.zeros([10000,1,dl1,dl2,dl3],dtype=np.float32)    
    trainDeform2Z=np.zeros([10000,1,dl1,dl2,dl3],dtype=np.float32)    
    
   
    cubicCnt=0

    for i in range(0, sz1-d1+1, step[0]):
        for j in range(0, sz2-d2+1, step[1]):
            for k in range(0, sz3-d3+1, step[2]):  
                maxV=np.max(matSub[i:i+d1, j:j+d2, k:k+d3])
                minV=np.min(matSub[i:i+d1, j:j+d2, k:k+d3])
                #print 'maxV, minV:',maxV,minV
                if maxV - minV < 0.2:
                    continue
                trainImg[cubicCnt,0,:,:,:] = matSub[i:i+d1, j:j+d2, k:k+d3]
                trainImg[cubicCnt,1,:,:,:] = matTem[i:i+d1, j:j+d2, k:k+d3]  
                trainImg[cubicCnt,2,:,:,:] = matSubG[i:i+d1, j:j+d2, k:k+d3]    
                trainImg[cubicCnt,3,:,:,:] = matTemG[i:i+d1, j:j+d2, k:k+d3]    


                trainDeform0X[cubicCnt,0,:,:,:] = matDeform[i+dec:i+dec+dc1, j+dec:j+dec+dc2, k+dec:k+dec+dc3, 0]
                trainDeform0Y[cubicCnt,0,:,:,:] = matDeform[i+dec:i+dec+dc1, j+dec:j+dec+dc2, k+dec:k+dec+dc3, 1]
                trainDeform0Z[cubicCnt,0,:,:,:] = matDeform[i+dec:i+dec+dc1, j+dec:j+dec+dc2, k+dec:k+dec+dc3, 2]
                
                trainDeform1X[cubicCnt,0,:,:,:] = matDeform[i+dem:i+dem+dm1*2:2, j+dem:j+dem+dm2*2:2, k+dem:k+dem+dm3*2:2, 0]
                trainDeform1Y[cubicCnt,0,:,:,:] = matDeform[i+dem:i+dem+dm1*2:2, j+dem:j+dem+dm2*2:2, k+dem:k+dem+dm3*2:2, 1]
                trainDeform1Z[cubicCnt,0,:,:,:] = matDeform[i+dem:i+dem+dm1*2:2, j+dem:j+dem+dm2*2:2, k+dem:k+dem+dm3*2:2, 2]
                
                trainDeform2X[cubicCnt,0,:,:,:] = matDeform[i+dell:i+dell+dl1*4:4, j+dell:j+dell+dl2*4:4, k+dell:k+dell+dl3*4:4, 0]
                trainDeform2Y[cubicCnt,0,:,:,:] = matDeform[i+dell:i+dell+dl1*4:4, j+dell:j+dell+dl2*4:4, k+dell:k+dell+dl3*4:4, 1]
                trainDeform2Z[cubicCnt,0,:,:,:] = matDeform[i+dell:i+dell+dl1*4:4, j+dell:j+dell+dl2*4:4, k+dell:k+dell+dl3*4:4, 2]
                
                
                cubicCnt = cubicCnt + 1;
    
    trainImg = trainImg[0:cubicCnt,:,:,:,:]
    trainDeform0X = trainDeform0X[0:cubicCnt,:,:,:,:] 
    trainDeform0Y = trainDeform0Y[0:cubicCnt,:,:,:,:] 
    trainDeform0Z = trainDeform0Z[0:cubicCnt,:,:,:,:] 
    trainDeform1X = trainDeform1X[0:cubicCnt,:,:,:,:] 
    trainDeform1Y = trainDeform1Y[0:cubicCnt,:,:,:,:] 
    trainDeform1Z = trainDeform1Z[0:cubicCnt,:,:,:,:] 
    trainDeform2X = trainDeform2X[0:cubicCnt,:,:,:,:] 
    trainDeform2Y = trainDeform2Y[0:cubicCnt,:,:,:,:] 
    trainDeform2Z = trainDeform2Z[0:cubicCnt,:,:,:,:] 
    print 'trainImg shape, ',trainImg.shape
    print 'trainDeform0 shape, ',trainDeform0X.shape
    print 'trainDeform1 shape, ',trainDeform1X.shape
    print 'trainDeform2 shape, ',trainDeform2X.shape
    
    with h5py.File('Data/Subject_%02d_%d.h5'%(fileID,warpID),'w') as f:
        f['dataSubjectImg']=trainImg
        f['dataDF0X']=trainDeform0X
        f['dataDF0Y']=trainDeform0Y
        f['dataDF0Z']=trainDeform0Z
        f['dataDF1X']=trainDeform1X
        f['dataDF1Y']=trainDeform1Y
        f['dataDF1Z']=trainDeform1Z
        f['dataDF2X']=trainDeform2X
        f['dataDF2Y']=trainDeform2Y
        f['dataDF2Z']=trainDeform2Z

     
    with open('Data/trainSet_list.txt','a') as f:
        f.write('Data/Subject_%02d_%d.h5\n'%(fileID,warpID))
    return cubicCnt
    
    	
def main():
    File_TemplateImg='na01_DF.mha'
    FilePath_TemplateImg=os.path.join(Path_Image,File_TemplateImg)
    img_Template=sitk.ReadImage(FilePath_TemplateImg, sitk.sitkFloat32)
    mat_Template=sitk.GetArrayFromImage(img_Template)
    maxV=np.max(mat_Template)
    minV=np.min(mat_Template)
    mat_Template = mat_Template/(maxV-minV)
        
    gradientFilter = sitk.GradientMagnitudeImageFilter() 
    img_TemplateGrad = gradientFilter.Execute(img_Template)
    mat_TemplateGrad=sitk.GetArrayFromImage(img_TemplateGrad)
    maxV=np.max(mat_TemplateGrad)
    minV=np.min(mat_TemplateGrad)
    mat_TemplateGrad = mat_TemplateGrad/(maxV-minV)
               
    for id in range(1, 2):
        for jth in range(0,1):
            
            File_SubjectImg='na%02d_Warped_%d.mha'%(id,jth)
            FilePath_SubjectImg=os.path.join(Path_Image,File_SubjectImg)
            
            File_Deform='na%02d_DF_%d.mha'%(id,jth)
            FilePath_Deform=os.path.join(Path_Deform,File_Deform)
            
            img_Subject=sitk.ReadImage(FilePath_SubjectImg, sitk.sitkFloat32)
            mat_Subject=sitk.GetArrayFromImage(img_Subject)
            maxV=np.max(mat_Subject)
            minV=np.min(mat_Subject)
            print 'maxV, minV:',maxV,minV
            mat_Subject = mat_Subject/(maxV-minV)
    
            img_SubjectGrad = gradientFilter.Execute(img_Subject)
            mat_SubjectGrad=sitk.GetArrayFromImage(img_SubjectGrad)
            maxV=np.max(mat_SubjectGrad)
            minV=np.min(mat_SubjectGrad)
            mat_SubjectGrad = mat_SubjectGrad/(maxV-minV)
    
    
            img_Deform=sitk.ReadImage(FilePath_Deform)
            mat_Deform=sitk.GetArrayFromImage(img_Deform) 
            #you can do what you want here for for your label img
            
            
    
            Cnt = Tohdf5(mat_Subject, mat_Template, mat_SubjectGrad, mat_TemplateGrad, mat_Deform, id, jth)
        print '# of patches is ', Cnt
    
if __name__ == '__main__':     
    main()

