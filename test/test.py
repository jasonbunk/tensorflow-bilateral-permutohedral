import os,sys
# force run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

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

def normalizeimg(im):
    img = np.array(im)
    imin = np.amin(img)
    imax = np.amax(img)
    goalsize = 200*200
    theim = (img-imin)/(imax-imin)
    #if im.size < goalsize:
    #    fsc = np.ceil(np.sqrt(float(goalsize) / float(im.size)))
    #    theim = cv2.resize(theim,(0,0),fx=fsc,fy=fsc,interpolation=cv2.INTER_NEAREST)
    return theim

if cv2available:
    def imshow(wname,img):
        cv2.imshow(wname,normalizeimg(img))
        cv2.waitKey(0)
    def imread(imfile):
        return cv2.imread(imfile,cv2.IMREAD_COLOR)
else:
    def imshow(wname,img):
        Image.fromarray(normalizeimg(img),'RGB').show()
    def imread(imfile):
        return np.array(Image.open(catfile))[:,:,::-1]

# cv2 image read shape: (height, width, channels)
# tensorflow conv2d image shape: (batch, height, width, channels)
catfile = os.path.join(path2file,'cat.jpg')
catim = imread(catfile).astype(np.float32) / 255.0
catim = catim.reshape([1,]+list(catim.shape))

tfcatplaceholder = tf.placeholder(tf.float32, catim.shape, name="tfcatplaceholder")

npwspatial = 0.5
npwbilateral = 0.5

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
                        8.0, 8.0)
outspace,_ = ret
outspace = tf.Print(outspace, [outspace], msgstr+"outspace")

ret = bilateral_filters(nchwcat_bilat,
                        nchwcat_bilat,
                        8.0, 8.0)
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
imshow("npcatspace",npcatspace[0,...])
imshow("npcatbilat",npcatbilat[0,...])
#cv2.waitKey(0)

for jj in range(10):
    print("ALLL GOOODDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
