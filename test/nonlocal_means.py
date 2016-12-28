import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
import cv2
import numpy as np
sys.path.insert(1, os.path.join(sys.path[0], '/mywork/tensorflow-tuts/sd19reader'))
from batches2patches_tensorflow import GetFuncToPatches, GetFuncOverlapAdd
from myutils import describe
from vizutils import bw_grid_vis, color_grid_vis
from mypca import my_PCA_scikitlike as PCA

# load image
path2file = os.path.dirname(os.path.realpath(__file__))
inimg = os.path.join(path2file,'Lenna_noise1.png')
testim = cv2.imread(inimg).astype(np.float32) / 255.0
# will use "valid" conv, so pad 1 wide for 3x3 patches
padtest = np.pad(testim, [(1,1), (1,1), (0,0)], 'edge')

# get patching function for local windows
imshape = [int(ii) for ii in padtest.shape]
batchsize = 1
batchimshapefull = [batchsize,]+imshape
patchsize = 3
bordermode = 'valid'
pimshape = (imshape[0]-patchsize+1,imshape[1]-patchsize+1)
reconstrmode = 'full'
N_PCA_COMPS = 6

batchunpadtest = np.expand_dims(testim, 0)
batchtestims = padtest.reshape(batchimshapefull) # only one in batch, so resize the one
featswrtshape = [int(ii) for ii in batchunpadtest.shape]
featswrtshape[-1] = N_PCA_COMPS

patchtheanofunc = GetFuncToPatches(batchimshapefull, patchsize, border_mode=bordermode, filter_flip=False)
overlapaddfunc = GetFuncOverlapAdd(batchimshapefull, patchsize, pimshape, border_mode=reconstrmode, filter_flip=False)

#########################################
# bilateral filter
#tf_stdv_space = tf.get_variable('tf_stdv_space', initializer=tf.constant(1.0))
#tf_stdv_bilat = tf.get_variable('tf_stdv_bilat', initializer=tf.constant(1.0))
tf_placehold_img = tf.placeholder(tf.float32, batchunpadtest.shape, name="tf_placehold_img")
tf_placehold_wrt = tf.placeholder(tf.float32, featswrtshape,        name="tf_placehold_wrt")

from test_utils import *
bilateral_filters = load_func_from_lib()

#########################################
# tensorflow sess init

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
outfold = 'test_patch_pca'

#########################################
# compute patch PCA

patches = patchtheanofunc(batchtestims)
print(" ")
describe("patches",patches)

flatpatches = patches.reshape((patches.shape[0]*patches.shape[1]*patches.shape[2], np.prod(patches.shape[3:])))
describe("flatpatches",flatpatches)
pca = PCA(n_components=N_PCA_COMPS, doplot=False).fit(flatpatches)

transfpatches = pca.transform(flatpatches)
reshtransfpatch = transfpatches.reshape((patches.shape[0], patches.shape[1], patches.shape[2], N_PCA_COMPS))

print(" ")
describe("transfpatches", transfpatches)
describe("reshtransfpatch", reshtransfpatch)
print(" ")

procpatches = pca.inverse_transform(transfpatches).reshape(patches.shape)
tehpidx = -1
for tehpatchs in [patches, procpatches]:
    tehpidx += 1
    FLPTCHS = tehpatchs.reshape((tehpatchs.shape[0], tehpatchs.shape[1]*tehpatchs.shape[2], np.prod(tehpatchs.shape[3:])))
    #describe("FLPTCHS", FLPTCHS)
    for jj in range(batchsize):
        #describe("FLPTCHS[jj,...]", FLPTCHS[jj,...])
        color_grid_vis(FLPTCHS[jj,...], savename=os.path.join(outfold,'pcacnn_FLPTCHS_'+str(tehpidx)+'_'+str(jj)+'.png'), flipbgr=True)

#quit()

#########################################

#define the function that's called every time one of the trackbars is moved
def updateWindow(xxx):
    stdspace = float(cv2.getTrackbarPos('std_space*10','ImageWindow')) / 10.
    stdcolor = float(cv2.getTrackbarPos('std_color*10','ImageWindow')) / 10.

    stdspace = max(1e-3, stdspace)
    stdcolor = max(1e-3, stdcolor)

    #tf_stdv_space = tf.get_variable('tf_stdv_space', initializer=tf.constant(1.0))
    #tf_stdv_bilat = tf.get_variable('tf_stdv_bilat', initializer=tf.constant(1.0))
    #tf_placehold_img = tf.placeholder(tf.float32, batchimshapefull, name="tf_placehold_img")
    #tf_placehold_wrt = tf.placeholder(tf.float32, featswrtshape,    name="tf_placehold_wrt")

    ret = bilateral_filters(NHWC_to_NCHW(tf_placehold_img),
                            NHWC_to_NCHW(tf_placehold_wrt),
                            stdspace, stdcolor)
    _,outbilNCHW = ret
    outbilat = NCHW_to_NHWC(outbilNCHW)

    tfret = outbilat.eval({tf_placehold_img: batchunpadtest, tf_placehold_wrt: reshtransfpatch})

    describe("tfret00", tfret)
    tfret[tfret<0.0] = 0.0
    tfret[tfret>1.0] = 1.0
    describe("tfret11", tfret)
    cv2.imshow("ImageWindow", tfret[0,...])

cv2.namedWindow('ImageWindow')
cv2.createTrackbar('std_space*10','ImageWindow',1,200,updateWindow)
cv2.createTrackbar('std_color*10','ImageWindow',1,200,updateWindow)
updateWindow(0) #Creates the window for the first time
cv2.waitKey(0)
