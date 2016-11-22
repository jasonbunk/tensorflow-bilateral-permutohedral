#!/usr/bin/env python
import os,sys

# force run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from random import randint, uniform
import numpy as np
import tensorflow as tf
sys.path.insert(1, os.path.join(sys.path[0], '/mywork/tensorflow-tuts/sd19reader'))
from myutils import describe

from test_utils import *
bilateral_filters = load_func_from_lib()

#---------------------------------------------------------------------
# setup a test

# NCHW
xshape      = [randint(2,6) for dim in range(4)]
wrtshp    = xshape
wrtshp[1] = randint(3,6) # num channels doesn't have to be same as xxx
yshape      = xshape
wshape = [1,] #[1, 1, xshape[1], xshape[1]]

myvars = {}
myvars['xxx'] = xshape
myvars['wrt'] = wrtshp
myvars['yyy'] = yshape
myvars['wsp'] = wshape
myvars['wbi'] = wshape

updates = []
for key in myvars:
    newvar = tf.get_variable(key, myvars[key], initializer=tf.truncated_normal_initializer(), trainable=False)
    # will set variables to new random values every iteration
    updates.append(tf.assign(newvar, tf.truncated_normal(myvars[key])))
    myvars[key] = (newvar, myvars[key])

stdv_spatial_space = np.exp(uniform(-2.0, 2.0))
stdv_bilater_space = np.exp(uniform(-2.0, 2.0))

ret = bilateral_filters(input=myvars['xxx'][0],
                            featswrt=myvars['wrt'][0],
                            stdv_spatial_space=stdv_spatial_space,
                            stdv_bilater_space=stdv_bilater_space)
xfilt = (ret[0] * myvars['wsp'][0] + ret[1] * myvars['wsp'][1])

errs = tf.square(xfilt - myvars['yyy'][0])
loss = tf.reduce_mean(errs)

sess = tf.InteractiveSession()
tf.initialize_all_variables().run()

# computes gradient dy/dx
# grad_err = tf.test.compute_gradient_error(x, x_shape, y, y_shape)

for ii in range(100):
    print("---- iter "+str(ii))
    for key in myvars:
        try:
            grads = tf.test.compute_gradient(myvars[key][0], myvars[key][1],
                                                loss, [1,])
        except AssertionError:
            grads = None

        if grads is None:
            print(key+" grads are None")
        else:
            if len(grads[0].shape) == len(grads[1].shape) and grads[0].shape == grads[1].shape:
                grads = [gg.astype(np.float64) for gg in grads]
                grad_diff = grads[0] - grads[1]
                grad_magn = np.fabs(grads[0]) + np.fabs(grads[1])
                grad_reld = np.divide(grad_diff, grad_magn+1e-15)
            else:
                print("error??")
                grad_diff = None

            describe(key+'_reldiff', grad_reld)
            #describe('    '+key+'_grads[0]',grads[0])
            #describe('    '+key+'_grads[1]',grads[1])

            if np.mean(np.fabs(grad_reld)) > 0.05 or np.std(grad_reld) > 0.05:
                print("\nWARNING: mean or stdv of relative gradient diff " \
                     +"was > 5 % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
