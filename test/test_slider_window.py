#!/usr/bin/env python
import os, sys
path2file = os.path.dirname(os.path.realpath(__file__))
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from glob import glob
import tensorflow as tf
import cv2
import numpy as np
from test_utils import *

print("usage:  {optional:input-image-file(s)}")
if len(sys.argv) > 1:
    infiles = [fname for argv in sys.argv[1:] for fname in glob(argv)]
else:
    infiles = [os.path.join(path2file,'cat.jpg'),]

def loadimg(fname):
    assert os.path.isfile(fname), fname
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    assert im is not None and im.size > 1 and len(im.shape) == 3
    return np.expand_dims(im.astype(np.float32) / 255.0, 0)

# load image
testims = [(ff,loadimg(ff)) for ff in infiles]

#########################################
# bilateral filter
from test_utils import *
bilateral_filters = load_func_from_lib()

#########################################
# tensorflow sess init

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())

#########################################

savemes = {}
placeholders = {}

#define the function that's called every time one of the trackbars is moved
def updateWindow(xxx):
    numiters = max(1, cv2.getTrackbarPos('num iters','Img-0'))
    prespace = float(cv2.getTrackbarPos('PRE_space*10','Img-0')) / 10.
    stdspace = float(cv2.getTrackbarPos('std_space*10','Img-0')) / 10.
    stdcolor = float(cv2.getTrackbarPos('std_color*400','Img-0')) / 400.

    prespace = max(1e-3, prespace)
    stdspace = max(1e-3, stdspace)
    stdcolor = max(1e-3, stdcolor)

    for ii in range(len(testims)):
        thisfname  = str(testims[ii][0])
        thistestim = testims[ii][1].copy()
        if ii not in placeholders:
            placeholders[ii] = tf.placeholder(tf.float32, thistestim.shape, name="tf_placehold_"+str(ii))

        ret = NHWC_to_NCHW(placeholders[ii])
        ret = bilateral_filters(ret, ret, prespace, 500.)
        for jj in range(numiters):
            ret = bilateral_filters(ret, ret, stdspace, stdcolor)
        outbilat = NCHW_to_NHWC(ret)
        tfret = outbilat.eval({placeholders[ii]: thistestim})

        tfret[tfret<0.0] = 0.0
        tfret[tfret>1.0] = 1.0
        cv2.imshow("Img-"+str(ii), tfret[0,...])
        savemes[thisfname] = tfret[0,...]

cv2.namedWindow('Img-0')
cv2.createTrackbar('num iters','Img-0',1,50,updateWindow)
cv2.createTrackbar('PRE_space*10','Img-0',1,100,updateWindow)
cv2.createTrackbar('std_space*10','Img-0',1,100,updateWindow)
cv2.createTrackbar('std_color*400','Img-0',1,100,updateWindow)
updateWindow(0) #Creates the window for the first time
cv2.waitKey(0)

for key in savemes:
    cv2.imwrite(key+'_CRF.png', np.uint8(np.round(savemes[key]*255.)))
