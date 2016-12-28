import os, sys
import tensorflow as tf
import numpy as np

def normalizeimg(im, simpleclip=False):
    img = np.array(im)
    if len(img.shape) == 3 and img.shape[2] == 2:
        img = np_2channel_to_3channel(img,2)
    goalsize = 200*200
    if simpleclip:
        im[im<0.0] = 0.0
        im[im>1.0] = 1.0
        theim = im
    else:
        if True:
            if True:
                centered = img - np.median(img,axis=(0,1),keepdims=True)
                istdv = np.std(centered,axis=(0,1),keepdims=True)
            else:
                centered = img - np.median(img)
                istdv = np.std(centered)
            theim = np.clip(0.5 + centered / (8.0*istdv), 0.0, 1.0)
        else:
            imin = np.amin(img)
            imax = np.amax(img)
            theim = (img-imin)/(imax-imin)
    #if im.size < goalsize:
    #    fsc = np.ceil(np.sqrt(float(goalsize) / float(im.size)))
    #    theim = cv2.resize(theim,(0,0),fx=fsc,fy=fsc,interpolation=cv2.INTER_NEAREST)
    return theim

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

def load_4channel_truth_img(filepath):
    import cv2
    # cv2 image read shape: (height, width, channels)
    # tensorflow conv2d image shape: (batch, height, width, channels)
    fullim4ch = cv2.imread(filepath,cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
    assert len(fullim4ch.shape) == 3 and fullim4ch.shape[2] == 4

    train_x = fullim4ch[:,:,:3]
    trainy1 = fullim4ch[:,:,3]
    train_x = train_x.reshape([1,]+list(train_x.shape)) # 4 channels: NHWC
    trainy1 = trainy1.reshape([1,]+list(trainy1.shape)) # 3 channels: NHW
    # make one-hot labels
    train_y = np.zeros(list(trainy1.shape)+[2,], dtype=np.float32)
    train_y[:,:,:,0] = 1.0 - trainy1
    train_y[:,:,:,1] = trainy1
    return train_x, train_y

def conv1x1(input, chansin, chansout, activation_fn, scopename="", bias=True,
            weights_init_value=None, bias_init_value=None, kernelwidth=1):
    chansin = int(chansin)
    chansout = int(chansout)
    with tf.variable_scope("conv1x1"+scopename) as scope:
        initstdv = 1.1368468 * np.float32(1.0*np.sqrt(1.0/float(chansin+chansout)))
        print("1x1conv: initstdv "+str(initstdv) \
                +", chansin "+str(chansin)+", chansout "+str(chansout))
        if weights_init_value is not None:
            wshape = None
            weightinit = tf.constant(weights_init_value)
        else:
            wshape = (kernelwidth,kernelwidth,chansin,chansout)
            weightinit = tf.truncated_normal_initializer(stddev=initstdv)
        weights = tf.get_variable(shape=wshape,
                        initializer=weightinit,
                        name='weights')
        convresult = tf.nn.conv2d(input, weights, strides=[1,1,1,1], padding='VALID')
        if bias:
            if bias_init_value is not None:
                bshape = None
                biasinit = tf.constant(bias_init_value)
            else:
                bshape = (1,1,1,chansout)
                biasinit = tf.constant_initializer(value=0)
            biases = tf.get_variable(shape=bshape,
                                initializer=biasinit,
                                name='biases')
            convresult += biases
        if activation_fn is None:
            return convresult
        else:
            return activation_fn(convresult)

def np_2channel_to_3channel(arr, chanidx):
    chanidx = int(chanidx)
    assert(arr.shape[chanidx] == 2)
    arrsh = [int(arr.shape[ii]) for ii in range(len(arr.shape))]
    arrsh[chanidx] = 1
    return np.concatenate((arr,np.zeros(arrsh,dtype=arr.dtype)),axis=chanidx)

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
