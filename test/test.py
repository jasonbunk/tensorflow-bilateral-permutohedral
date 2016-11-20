#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import cv2
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '/mywork/tensorflow-tuts/sd19reader'))
from myutils import describe

#---------------------------------------------------------------------
# load library
path2file = os.path.dirname(os.path.realpath(__file__))
builtlibpath_dir = os.path.join(path2file, '../build/lib')
builtlibpath = os.path.join(builtlibpath_dir, 'libtfgaussiancrf.so')
libtfgaussiancrf = tf.load_op_library(builtlibpath)

print("-----------------------------------")
from inspect import getmembers, isfunction
functions_list = [o for o in getmembers(libtfgaussiancrf) if isfunction(o[1])]
print(str(functions_list))
print("-----------------------------------")

#---------------------------------------------------------------------
# setup a test

# cv2 image read shape: (height, width, channels)
# tensorflow conv2d image shape: (batch, height, width, channels)
catfile = os.path.join(path2file,'cat.jpg')
catim = cv2.imread(catfile,cv2.IMREAD_COLOR).astype(np.float32) / 255.0
catim = catim.reshape([1,]+list(catim.shape))

tfcatplaceholder = tf.placeholder(tf.float32, [None,None,None,None], name="tfcatplaceholder")

copycat  = libtfgaussiancrf.bilateral_filters(tfcatplaceholder)
copycat2 = libtfgaussiancrf.bilateral_filters(copycat)

#---------------------------------------------------------------------
# run the test

sess = tf.InteractiveSession()

tf.initialize_all_variables().run()

npcopycat = copycat.eval({tfcatplaceholder: catim})

describe("catim",catim)
describe("npcopycat",npcopycat)
cv2.imshow("catim",catim[0,...])
cv2.imshow("npcopycat",npcopycat[0,...])
cv2.waitKey(0)

