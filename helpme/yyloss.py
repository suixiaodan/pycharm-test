"""
Created on Mon Aug 14 21:14:32 2017

@author: yanyun
"""

caffe_root = '//usr/local/caffe3/'  # this is the path in GPU server
import sys
sys.path.insert(0, caffe_root + 'python')
print caffe_root + 'python'
import caffe

import numpy as np
class EucLossLayer(caffe.Layer):
    """

        """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 3:
            raise Exception("Need three inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float64)

        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        [sz1, sz2, sz3] = bottom[0].shape
        print 'size:', [sz1, sz2, sz3]
        matOutA = np.zeros([sz1, sz2, sz3, 3], dtype=np.float64)
        matOutB = np.zeros([sz1, sz2, sz3, 3], dtype=np.float64)

        matOutA[:, :, :, 0] = bottom[0].data[:, 0, :, :, :]
        matOutA[:, :, :, 1] = bottom[0].data[:, 1, :, :, :]
        matOutA[:, :, :, 2] = bottom[0].data[:, 2, :, :, :]
        matOutB[:, :, :, 0] = bottom[0].data[:, 3, :, :, :]
        matOutB[:, :, :, 1] = bottom[0].data[:, 4, :, :, :]
        matOutB[:, :, :, 2] = bottom[0].data[:, 5, :, :, :]

        img_DeformA = sitk.GetImageFromArray(matOutA)
        outTxA = sitk.DisplacementFieldTransform(img_DeformA)

        img_DeformB = sitk.GetImageFromArray(matOutB)
        outTxB = sitk.DisplacementFieldTransform(img_DeformB)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(img_Subject)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(100)

        resampler.SetTransform(outTxA)
        resampler.SetTransform(outTxB)

        ImgA=bottom[1].data
        ImgB=bottom[2].data

        outimgA = resampler.Execute(ImgA)
        outimgB = resampler.Execute(ImgB)

        self.diff=np.zeros_like(outimgA, dtpy=np.float64)

        self.diff[...]=outimgA-outimgB
        [bsz, chsz, sz1, sz2, sz3] = self.diff.shape

        top[0].data[...] = np.sum(self.diff ** 2) / bsz / sz1 / sz2 / sz3 / 2.


    def backward(self, top, propagate_down, bottom):
        [sz1, sz2, sz3] = bottom[0].shape
        print 'size:', [sz1, sz2, sz3]
        matOutA = np.zeros([sz1, sz2, sz3, 3], dtype=np.float64)
        matOutB = np.zeros([sz1, sz2, sz3, 3], dtype=np.float64)

        matOutA[:, :, :, 0] = bottom[0].data[:, 0, :, :, :]
        matOutA[:, :, :, 1] = bottom[0].data[:, 1, :, :, :]
        matOutA[:, :, :, 2] = bottom[0].data[:, 2, :, :, :]
        matOutB[:, :, :, 0] = bottom[0].data[:, 3, :, :, :]
        matOutB[:, :, :, 1] = bottom[0].data[:, 4, :, :, :]
        matOutB[:, :, :, 2] = bottom[0].data[:, 5, :, :, :]

        img_DeformA = sitk.GetImageFromArray(matOutA)
        outTxA = sitk.DisplacementFieldTransform(img_DeformA)

        img_DeformB = sitk.GetImageFromArray(matOutB)
        outTxB = sitk.DisplacementFieldTransform(img_DeformB)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(img_Subject)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(100)

        resampler.SetTransform(outTxA)
        resampler.SetTransform(outTxB)

        ImgA=bottom[1].data
        ImgB=bottom[2].data

        outimgA = resampler.Execute(ImgA)
        outimgB = resampler.Execute(ImgB)

        self.diff=np.zeros_like(outimgA, dtpy=np.float64)

        self.diff[...]=outimgA-outimgB

        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].num
        pass
