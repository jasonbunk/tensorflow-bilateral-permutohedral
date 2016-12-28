#!/usr/bin/env python
import os,sys
import pickle

# force run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from random import randint, uniform
import numpy as np
import tensorflow as tf
sys.path.insert(1, os.path.join(sys.path[0], '/mywork/tensorflow-tuts/sd19reader'))
from myutils import describe

from test_utils import *
bilateral_filters = load_func_from_lib()
path2file = os.path.dirname(os.path.realpath(__file__))

import cv2
def imshow(wname,img):
    #cv2.imshow(wname,normalizeimg(img))
    #cv2.waitKey(0)
    describe("\n   "+wname,img)
    cv2.imwrite(wname+'.png',np.round(normalizeimg(img)*255.0).astype(np.uint8))
def imshow_simplenorm(wname,img):
    cv2.imwrite(wname+'.png',np.round(normalizeimg(img,simpleclip=True)*255.0).astype(np.uint8))
#---------------------------------------------------------------------
# setup a test

import sys
try:
    infilename = sys.argv[1]
except:
    print("usage:  {test-filename}")
    quit()

def nchw2hwc(arr):
    assert len(arr.shape) == 4
    return arr[0,:,:,:].transpose((1,2,0))

thispath = '/mywork/tensorflow-tuts/ops_new_cpp/meanfield_iteration'
sys.path.insert(1, os.path.join(sys.path[0], os.path.join(thispath,'CPP_BoostPython_SuperSlowGrads')))
import pymytestgradslib

#catfile = os.path.join(path2file,'tiny_target_rgb.png')
catim = cv2.imread(infilename,cv2.IMREAD_COLOR).astype(np.float64) / 255.0
describe("catim",catim)

npwbilateral = 0.25
stdv_spatial_space = 1.8
stdv_bilater_space = 1.8
grad_chan = 0

gradsret = pymytestgradslib.MyTestGradients([catim/npwbilateral, catim/npwbilateral, catim.copy(), catim.copy(), catim.copy()],
                                            stdv_spatial_space, stdv_bilater_space,
                                            grad_chan)

describe("gradsret",gradsret)
xxx, wrt, outspace, outbilat, grad_outbi = gradsret

if grad_chan >= 0:
    imshow("grad_"+str(grad_chan)+"_outspace_SLOCPP",outspace)
    imshow("grad_"+str(grad_chan)+"_outbilat_SLOCPP",outbilat)
else:
    imshow("outspace_SLOCPP",outspace)
    imshow("outbilat_SLOCPP",outbilat)
    imshow_simplenorm("outspace_SLOCPP_simpnorm",outspace*npwbilateral)
    imshow_simplenorm("outbilat_SLOCPP_simpnorm",outbilat*npwbilateral)
