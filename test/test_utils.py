import os, sys
import tensorflow as tf
import numpy as np

def load_func_from_lib():
    path2file = os.path.dirname(os.path.realpath(__file__))
    builtlibpath_dir = os.path.join(path2file, '../build/lib')
    if False:
        builtlibpath = os.path.join(builtlibpath_dir, 'libtfgaussiancrf.so')
        libtfgaussiancrf = tf.load_op_library(builtlibpath)
        print("-----------------------------------")
        from inspect import getmembers, isfunction
        functions_list = [o for o in getmembers(libtfgaussiancrf) if isfunction(o[1])]
        print(str(functions_list))
        print("-----------------------------------")
        bilateral_filters = libtfgaussiancrf.bilateral_filters
    else:
        sys.path.insert(1, os.path.join(sys.path[0], builtlibpath_dir))
        import bilateral_op_and_grad
        bilateral_filters = bilateral_op_and_grad.bilateral_filters
    return bilateral_filters

def conv1x1(input, chansin, chansout, activation_fn, scopename="", bias=True):
    chansin = int(chansin)
    chansout = int(chansout)
    with tf.variable_scope("conv1x1"+scopename) as scope:
        initstdv = 1.1368468 * np.float32(1.0*np.sqrt(1.0/float(chansin+chansout)))
        print("1x1conv: initstdv "+str(initstdv) \
                +", chansin "+str(chansin)+", chansout "+str(chansout))
        weights = tf.get_variable(shape=(1,1,chansin,chansout),
                        initializer=tf.truncated_normal_initializer(stddev=initstdv),
                        name='weights')
        convresult = tf.nn.conv2d(input, weights, strides=[1,1,1,1], padding='VALID')
        if bias:
            biases = tf.get_variable(shape=(1,1,1,chansout),
                                initializer=tf.constant_initializer(value=0),
                                name='biases')
            convresult += biases
        if activation_fn is None:
            return convresult
        else:
            return activation_fn(convresult)

# supports numpy arrays and tensorflow tensors
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
