#!/usr/bin/env python
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import tensorflow as tf
import cv2
import pickle

from test_utils import *
bilateral_filters = load_func_from_lib()

path2file = os.path.dirname(os.path.realpath(__file__))
#---------------------------------------------------------------------
# setup a test
useCRF = True

varsdict = pickle.load(open("trained_segment_weights.pkl", "rb"))
#for key in varsdict:
#    print(str(key))
def tfconst(vname, vval):
    return tf.get_variable(vname, initializer=tf.constant(vval), trainable=False)

train_x, train_y = load_4channel_truth_img(varsdict['imagefile'])

tf_x_placehold = tfconst('tf_x_placehold', train_x)
tf_y_placehold = tfconst('tf_y_placehold', train_y)

comp_class = conv1x1(tf_x_placehold,  3, 2, None, "comp_class",
                        weights_init_value=varsdict['conv1x1comp_class/weights:0'],
                        bias_init_value=varsdict['conv1x1comp_class/biases:0'])

crfprescale  = None
tfwspatial   = None
tfwbilateral = None
if not useCRF:
    print("NOT using crf")
    finalpred_logits = comp_class
else:
    stdv_space = varsdict['stdv_space']
    stdv_color = varsdict['stdv_color']
    print("using CRF")
    #crfprescale  = tfconst('crfprescale', varsdict['crfprescale:0'])
    tfwspatial   = tfconst('tfwspatial',  varsdict['tfwspatial:0'])
    tfwbilateral = tfconst('tfwbilateral',varsdict['tfwbilateral:0'])

    reshcc = NHWC_to_NCHW(comp_class)# * crfprescale)
    reshfwrt = NHWC_to_NCHW(tf_x_placehold)
    outbilat = bilateral_filters(input=reshcc, #input
                            featswrt=reshfwrt, #featswrt
                            stdv_space=stdv_space,
                            stdv_color=stdv_color)
    finalpred_logits = NCHW_to_NHWC(outbilat * tfwbilateral)

finalpred_softmax = tf.nn.softmax(finalpred_logits)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(finalpred_logits, tf_y_placehold))

#train_step = tf.train.AdamOptimizer(LEARNRATE).minimize(loss)

def tshape(tfarr):
    return [int(dd) for dd in tfarr.get_shape()]

myvars = {}
myvars['xxx'] = (reshcc,   tshape(reshcc))
myvars['wrt'] = (reshfwrt, tshape(reshfwrt))
myvars['outbi'] = (outbilat, tshape(outbilat))

sess = tf.InteractiveSession()
tf.initialize_all_variables().run()

savecheckvars = []
savecheckvars.append(reshcc.eval())
savecheckvars.append(reshfwrt.eval())
savecheckvars.append(outbilat.eval())

# computes gradient dy/dx
# grad_err = tf.test.compute_gradient_error(x, x_shape, y, y_shape)

def nchw2hwc(arr):
    assert len(arr.shape) == 4
    return arr[0,:,:,:].transpose((1,2,0))
def prepareasmat(grad, gradshape):
    assert np.prod(gradshape) == grad.size
    return nchw2hwc(grad.reshape(gradshape))
def myimshow(wname,grad,gradshape):
    assert np.prod(gradshape) == grad.size
    amin = np.amin(grad)
    amax = np.amax(grad)
    gvis = (grad-amin)/(amax-amin+1e-12)
    gres = prepareasmat(gvis, gradshape)
    scf = 6
    newsize = tuple([scf*gg for gg in gradshape[2:4]])
    print("gradshape "+str(gradshape)+", newsize "+str(newsize))
    if gres.shape[-1] == 2:
        gres = np_2channel_to_3channel(gres, 2)
    gres = cv2.resize(gres, newsize, interpolation=cv2.INTER_NEAREST)
    #cv2.imshow(wname,gres)
    cv2.imwrite('test/out/'+wname+'_min_'+str(amin)+'_max_'+str(amax)+'.png', np.round(gres*255.0).astype(np.uint8))

for ii in range(100):
    print("---- iter "+str(ii))
    compgrads = {}
    for key in myvars:
        try:
            grads = tf.test.compute_gradient(myvars[key][0], myvars[key][1],
                                                loss, [1,], x_init_value=myvars[key][0].eval())
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

                compgrads[key] = grads[1]
            else:
                print("error??")
                grad_diff = None

            describe(key+'_reldiff', grad_reld)
            #describe('    '+key+'_grads[0]',grads[0])
            #describe('    '+key+'_grads[1]',grads[1])
            #print("myvars["+str(key)+"][1] == "+str(myvars[key][1]))
            if len(myvars[key][1]) == 4:
                myimshow(key+'_an',grads[0],myvars[key][1])
                myimshow(key+'_nu',grads[1],myvars[key][1])

            #if np.mean(np.fabs(grad_reld)) > 0.05 or np.std(grad_reld) > 0.05:
            #    print("\nWARNING: mean or stdv of relative gradient diff " \
            #         +"was > 5 % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    myimshow('in_wrt', myvars['wrt'][0].eval(), myvars['wrt'][1])
    myimshow('in_xxx', myvars['xxx'][0].eval(), myvars['xxx'][1])
    myimshow('in_xxx', myvars['xxx'][0].eval(), myvars['xxx'][1])
    myimshow('outbilat', outbilat.eval(), tshape(outbilat))
    cv2.waitKey(0)

    savecheckvars.append(prepareasmat(compgrads['xxx'],   myvars['xxx'][1]))
    savecheckvars.append(prepareasmat(compgrads['wrt'],   myvars['wrt'][1]))
    savecheckvars.append(prepareasmat(compgrads['outbi'], myvars['outbi'][1]))
    #savecheckvars.append(varsdict['crfprescale:0'])
    savecheckvars.append(stdv_space)
    savecheckvars.append(stdv_color)

    pickle.dump(savecheckvars, open("trained_segment_reshcc.pkl", "wb"))
    print("dumped savecheckvars to trained_segment_reshcc")

    quit()
