#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import cv2
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '/mywork/tensorflow-tuts/sd19reader'))
from myutils import describe

from test_utils import *
bilateral_filters = load_func_from_lib()

path2file = os.path.dirname(os.path.realpath(__file__))
#---------------------------------------------------------------------
# setup a test

# cv2 image read shape: (height, width, channels)
# tensorflow conv2d image shape: (batch, height, width, channels)
catfile = os.path.join(path2file,'cat.jpg')
catim = cv2.imread(catfile,cv2.IMREAD_COLOR).astype(np.float32) / 255.0
catim = catim.reshape([1,]+list(catim.shape))

tfcatplaceholder = tf.placeholder(tf.float32, catim.shape, name="tfcatplaceholder")

npwspatial = 0.0
npwbilateral = 0.5

totalscalenorm = (npwspatial + npwbilateral)

''' SEE: bilateral_op_and_grad.py
input,
featswrt,
stdv_spatial_space=1.0,
stdv_bilater_space=1.0,
'''

nchw_cat = NHWC_to_NCHW(tfcatplaceholder)
ret = bilateral_filters(nchw_cat,
                        nchw_cat,
                        8.0,
                        8.0)
outspace, outbilat = ret
copycat = NCHW_to_NHWC(outspace * npwspatial + outbilat * npwbilateral)

print("constructed the filter!!!!!!!!!!!!!!!!!!!!!!")
describe("catim",catim)
describe("tfcatplaceholder",tfcatplaceholder)
describe("outspace",outspace)
describe("outbilat",outbilat)
describe("PREINIT: copycat",copycat)
print("\n")

#---------------------------------------------------------------------
# run the test

sess = tf.InteractiveSession()

tf.initialize_all_variables().run()

describe("BEF: copycat",copycat)
print("BEF: copycat.get_shape(): "+str(copycat.get_shape()))

npcopycat = copycat.eval({tfcatplaceholder: (catim/totalscalenorm)})
npcopyca2 = copycat.eval({tfcatplaceholder: (npcopycat/totalscalenorm)})

describe("AFT: copycat",copycat)

describe("catim",catim)
describe("npcopycat",npcopycat)
cv2.imshow("catim",catim[0,...])
cv2.imshow("npcopycat",npcopycat[0,...])
cv2.imshow("npcopyca2",npcopyca2[0,...])
cv2.waitKey(0)
