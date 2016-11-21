#!/usr/bin/env python
import os,sys

# force run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import tensorflow as tf
import cv2
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

def conv1x1(input, chansin, chansout, activation_fn, scopename="", bias=True):
    chansin = int(chansin)
    chansout = int(chansout)
    with tf.variable_scope("conv1x1"+scopename) as scope:
        initstdv = 1.1368468 * np.float32(1.0*np.sqrt(1.0/float(chansin+chansout)))
        print("1x1conv: initstdv "+str(initstdv)+", chansin "+str(chansin)+", chansout "+str(chansout))
        weights = tf.get_variable(shape=(1,1,chansin,chansout), initializer=tf.truncated_normal_initializer(stddev=initstdv), name='weights')
        convresult = tf.nn.conv2d(input, weights, strides=[1,1,1,1], padding='VALID')
        if bias:
            biases = tf.get_variable(shape=(1,1,1,chansout), initializer=tf.constant_initializer(value=0), name='biases')
            convresult += biases
        if activation_fn is None:
            return convresult
        else:
            return activation_fn(convresult)

def NHWC_to_NCHW(arr):
    try:
        return tf.transpose(arr, perm=(0,3,1,2))
    except:
        return arr.transpose((0,3,1,2))
def NCHW_to_NHWC(arr):
    try:
        return tf.transpose(arr, perm=(0,2,3,1))
    except:
        return arr.transpose((0,2,3,1))

# cv2 image read shape: (height, width, channels)
# tensorflow conv2d image shape: (batch, height, width, channels)
imfile = os.path.join(path2file,'orange_target.png')
fullim4ch = cv2.imread(imfile,cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0

train_x = fullim4ch[:,:,:3]
trainy1 = fullim4ch[:,:,3]
train_x = train_x.reshape([1,]+list(train_x.shape))
trainy1 = trainy1.reshape([1,]+list(trainy1.shape)+[1,])
# make one-hot labels
train_y = np.zeros(list(trainy1.shape[:-1])+[2,], dtype=np.float32)
train_y[trainy1>0.5] = 1.0
train_y = train_y[:,:,:,::-1]
train_y[:,:,:,0] = 1.0 - train_y[:,:,:,1]

xmean = np.mean(train_x, axis=(1,2), keepdims=True)
train_x -= xmean

cv2.imshow("train_x",(train_x+xmean)[0,...])
cv2.imshow("train_y_0",train_y[0,:,:,0])
cv2.imshow("train_y_1",train_y[0,:,:,1])

describe("fullim4ch",fullim4ch)
describe("train_x",train_x)
describe("train_y",train_y)
#cv2.waitKey(0)
#quit()

tf_x_placehold = tf.placeholder(tf.float32, train_x.shape, name="tf_x_placehold")
tf_y_placehold = tf.placeholder(tf.float32, train_y.shape, name="tf_x_placehold")

#----------------------------------------
useCRF = False
LEARNRATE = 0.05
NUMITERS = 10000

#comp_conv1 = conv1x1(tf_x_placehold, 3, 3, tf.nn.elu, "comp_conv1")
#comp_class = conv1x1(comp_conv1,     3, 2, None,      "comp_class")
comp_class = conv1x1(tf_x_placehold,  3, 2, None,      "comp_class")
#----------------------------------------

crfprescale  = None
tfwspatial   = None
tfwbilateral = None
if not useCRF:
    finalpred_logits = comp_class
else:
    npwspatial = np.expand_dims(np.expand_dims(np.eye(2),0),0).astype(np.float32)
    npwbilateral = npwspatial.copy()

    crfprescale  = tf.get_variable('crfprescale', initializer=tf.constant(1.0))
    tfwspatial   = tf.get_variable('tfwspatial',  initializer=tf.constant(npwspatial))
    tfwbilateral = tf.get_variable('tfwbilateral',initializer=tf.constant(npwbilateral))

    reshcc = NHWC_to_NCHW(comp_class * crfprescale)
    almostpreds = libtfgaussiancrf.bilateral_filters(input=reshcc, #input
                                                    featswrt=tf_x_placehold, #featswrt
                                                    wspatial=tfwspatial,
                                                    wbilateral=tfwbilateral,
                                                    stdv_spatial_space=10.0,
                                                    stdv_bilater_space=10.0)
    finalpred_logits = NCHW_to_NHWC(almostpreds)

finalpred_softmax = tf.nn.softmax(finalpred_logits)
total_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(finalpred_logits, tf_y_placehold))

train_step = tf.train.AdamOptimizer(LEARNRATE).minimize(total_loss)

print("constructed the filter!!!!!!!!!!!!!!!!!!!!!!")
describe("tf_x_placehold",tf_x_placehold)
describe("tf_y_placehold",tf_y_placehold)
describe("finalpred_logits",finalpred_logits)
describe("tfwspatial",tfwspatial)
describe("tfwbilateral",tfwbilateral)
print("\n")

#---------------------------------------------------------------------
# run the test

sess = tf.InteractiveSession()
tf.initialize_all_variables().run()
feeddict_xy = {tf_x_placehold: train_x, tf_y_placehold: train_y}
feeddict_x  = {tf_x_placehold: train_x}

for ii in range(NUMITERS):
    thisloss = sess.run([total_loss, train_step], feeddict_xy)[0]
    print("iter "+str(ii)+" loss: "+str(thisloss))
    if ii % 2 == 0:
        these_preds = finalpred_softmax.eval(feeddict_x)
        cv2.imshow("preds",these_preds[0,:,:,1])
        cv2.waitKey(100)
