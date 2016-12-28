import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
import cv2
import numpy as np
sys.path.insert(1, os.path.join(sys.path[0], '/mywork/tensorflow-tuts/sd19reader'))
from myutils import describe

# load image
path2file = os.path.dirname(os.path.realpath(__file__))
inimg = os.path.join(path2file,'Lenna_noise1.png')
testim = cv2.imread(inimg).astype(np.float32) / 255.0
testim = np.expand_dims(testim, 0)

featswrt = testim.copy()

#########################################
# bilateral filter
tf_placehold_img = tf.placeholder(tf.float32, testim.shape, name="tf_placehold_img")
tf_placehold_wrt = tf.placeholder(tf.float32, featswrt.shape, name="tf_placehold_wrt")

from test_utils import *
bilateral_filters = load_func_from_lib()

#########################################
# tensorflow sess init

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.InteractiveSession(config=config)
sess.run(tf.initialize_all_variables())

#########################################

#define the function that's called every time one of the trackbars is moved
def updateWindow(xxx):
    stdspace = float(cv2.getTrackbarPos('std_space*10','ImageWindow')) / 10.
    stdcolor = float(cv2.getTrackbarPos('std_color*50','ImageWindow')) / 50.

    stdspace = max(1e-3, stdspace)
    stdcolor = max(1e-3, stdcolor)

    ret = bilateral_filters(NHWC_to_NCHW(tf_placehold_img),
                            NHWC_to_NCHW(tf_placehold_wrt),
                            stdspace, stdcolor)
    outbilat = NCHW_to_NHWC(ret)

    tfret = outbilat.eval({tf_placehold_img: testim, tf_placehold_wrt: featswrt})

    describe("tfret00", tfret)
    tfret[tfret<0.0] = 0.0
    tfret[tfret>1.0] = 1.0
    describe("tfret11", tfret)
    cv2.imshow("ImageWindow", tfret[0,...])

cv2.namedWindow('ImageWindow')
cv2.createTrackbar('std_space*10','ImageWindow',1,100,updateWindow)
cv2.createTrackbar('std_color*50','ImageWindow',1,100,updateWindow)
updateWindow(0) #Creates the window for the first time
cv2.waitKey(0)
