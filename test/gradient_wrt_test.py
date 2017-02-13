#!/usr/bin/env python
import os,sys
import pickle

# force run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from random import randint, uniform
import numpy as np
import tensorflow as tf

from test_utils import *
bilateral_filters = load_func_from_lib()

import cv2
def imshow(wname,img):
    #cv2.imshow(wname,normalizeimg(img))
    #cv2.waitKey(0)
    describe("\n   "+wname,img)
    cv2.imwrite(wname+'.png',np.round(normalizeimg(img)*255.0).astype(np.uint8))
#---------------------------------------------------------------------
# setup a test

def nchw2hwc(arr):
    assert len(arr.shape) == 4
    return arr[0,:,:,:].transpose((1,2,0))

thispath = '/mywork/tensorflow-tuts/ops_new_cpp/meanfield_iteration'
sys.path.insert(1, os.path.join(sys.path[0], os.path.join(thispath,'CPP_BoostPython_SuperSlowGrads')))
import pymytestgradslib

varslist = pickle.load(open('trained_segment_reshcc.pkl', 'rb'))
for ii in range(len(varslist)):
    varslist[ii] = np.float64(varslist[ii])
for ii in range(4):
    varslist[ii] = nchw2hwc(varslist[ii])

for tvar in varslist:
    describe("var", tvar)

xxx, wrt, outspace, outbilat, grad_xxx, grad_wrt, grad_outbi, crfprescale, stdv_spatial_space, stdv_bilater_space = varslist

imshow("xxx",xxx)
imshow("wrt",wrt)
imshow("outspace",outspace)
imshow("outbilat",outbilat)

gradsret = pymytestgradslib.MyTestGradients([xxx, wrt, outspace, outbilat, grad_outbi],
                                            stdv_spatial_space, stdv_bilater_space,
                                            -1)

describe("gradsret",gradsret)
xxx, wrt, outspace, outbilat, grad_outbi = gradsret

imshow("outspace_SLOCPP",outspace)
imshow("outbilat_SLOCPP",outbilat)
