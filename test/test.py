import os,sys
# force run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import tensorflow as tf
sys.path.insert(1, os.path.join(sys.path[0], '/mywork/tensorflow-tuts/sd19reader'))
from myutils import describe

from test_utils import *
bilateral_filters = load_func_from_lib()

path2file = os.path.dirname(os.path.realpath(__file__))
#---------------------------------------------------------------------
# setup a test

cv2available = False
try:
    import cv2
    cv2available = True
except:
    from PIL import Image
from test_utils import normalizeimg

if cv2available:
    def imshow(wname,img):
        #cv2.imshow(wname,normalizeimg(img))
        #cv2.waitKey(0)
        cv2.imwrite(wname+'.png',np.round(normalizeimg(img)*255.0).astype(np.uint8))
    def imread(imfile):
        return cv2.imread(imfile,cv2.IMREAD_COLOR)
else:
    def imshow(wname,img):
        Image.fromarray(np.round(normalizeimg(img)*255.0).astype(np.uint8)[:,:,::-1],'RGB').show()
    def imread(imfile):
        inimg = np.array(Image.open(catfile))
        if len(inimg.shape) == 3 and inimg.shape[2] == 4:
            inimg = inimg[:,:,:3]
        return inimg[:,:,::-1]


THE_IMAGE_FILE = 'tiny_target_rgb.png'
#THE_IMAGE_FILE = 'cat_square_content_aware_108x108.png'


# cv2 image read shape: (height, width, channels)
# tensorflow conv2d image shape: (batch, height, width, channels)
catfile = os.path.join(path2file,THE_IMAGE_FILE)
catim = imread(catfile).astype(np.float32) / 255.0
catim = catim.reshape([1,]+list(catim.shape))

tfcatplaceholder = tf.placeholder(tf.float32, catim.shape, name="tfcatplaceholder")

npwspatial = 1.0
npwbilateral = 0.25
stdv = 3.375 * float(catim.shape[1])/108.0

totalscalenorm = (npwspatial + npwbilateral)

''' SEE: bilateral_op_and_grad.py
input,
featswrt,
stdv_spatial_space=1.0,
stdv_bilater_space=1.0,
'''
msgstr = "@@@@@@@@@@@@@@@@@@@@@@@ "

nchw_cat = NHWC_to_NCHW(tfcatplaceholder)

nchwcat_space = nchw_cat / tf.constant(1e-9 + npwspatial)
nchwcat_bilat = nchw_cat / tf.constant(1e-9 + npwbilateral)

ret = bilateral_filters(tf.Print(nchwcat_space, [nchwcat_space], msgstr+"input"),
                        tf.Print(nchwcat_space, [nchwcat_space], msgstr+"featswrt"),
                        stdv, stdv)
outspace,_ = ret
outspace = tf.Print(outspace, [outspace], msgstr+"outspace")

ret = bilateral_filters(nchwcat_bilat,
                        nchwcat_bilat,
                        stdv, stdv)
_,outbilat = ret
outbilat = tf.Print(outbilat, [outbilat], msgstr+"outbilat")

SCALEDoutsp = (outspace * npwspatial)
SCALEDoutbi = (outbilat * npwbilateral)

SCALEDoutsp = tf.Print(SCALEDoutsp, [SCALEDoutsp], msgstr+"SCALEDoutsp")
SCALEDoutbi = tf.Print(SCALEDoutbi, [SCALEDoutbi], msgstr+"SCALEDoutbi")

catspace = NCHW_to_NHWC(SCALEDoutsp)
catbilat = NCHW_to_NHWC(SCALEDoutbi)

catspace = tf.Print(catspace, [catspace], msgstr+"catspace")
catbilat = tf.Print(catbilat, [catbilat], msgstr+"catbilat")

print("constructed the filter!!!!!!!!!!!!!!!!!!!!!!")
describe("catim",catim)
describe("tfcatplaceholder",tfcatplaceholder)
describe("outspace",outspace)
describe("outbilat",outbilat)
describe("PREINIT: catbilat",catbilat)
print("stdv: "+str(stdv))
print("\n")

#---------------------------------------------------------------------
# run the test

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.InteractiveSession(config=config)

tf.initialize_all_variables().run()

describe("BEF: catbilat",catbilat)
print("BEF: catbilat.get_shape(): "+str(catbilat.get_shape()))

npcatspace = catspace.eval({tfcatplaceholder: catim})
print("running a second time!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
npcatbilat = catbilat.eval({tfcatplaceholder: catim})

describe("AFT: catbilat",catbilat)

describe("catim",catim)
describe("npcatbilat",npcatbilat)
imshow("catim",catim[0,...])
imshow("grad_4_permut_space_gputest",npcatspace[0,...])
imshow("grad_4_permut_bilat_gputest",npcatbilat[0,...])
#cv2.waitKey(0)

for jj in range(10):
    print("ALLL GOOODDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
