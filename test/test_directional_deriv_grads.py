import os,sys
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import cv2
path2file = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(1, os.path.join(sys.path[0], '/mywork/tensorflow-tuts/sd19reader'))
from myutils import describe

from test_directions_permutohedral import get_permutohedral_basis
def simpdesc(name,arr):
    print(str(name))
    print(str(arr))
    print(" ")


#testimpath = '1_0__crop.jpg'
testimpath = 'cat.jpg'
#testimpath = 'redcorner.png'

def imshow(wname, im, imin=None, imax=None):
    if imin is None:
        imin = np.amin(im)
    if imax is None:
        imax = np.amax(im)
    goalsize = 200*200
    theim = (im-imin)/(imax-imin)
    #if im.size < goalsize:
    #    fsc = np.ceil(np.sqrt(float(goalsize) / float(im.size)))
    #    theim = cv2.resize(theim,(0,0),fx=fsc,fy=fsc,interpolation=cv2.INTER_NEAREST)
    cv2.imshow(str(wname),theim)

cvim = cv2.imread(path2file+'/'+testimpath,cv2.IMREAD_COLOR).astype(np.float64)
#cv2.imshow('cvim',cvim)
#cv2.waitKey(0)

def buildkern(angle, stdv, deriv=False):
    angle = float(angle)
    stdv = float(stdv)
    # at about 4 standard deviations, standard normal is very close to zero
    pixwide = int(np.ceil(2.*stdv*4.))
    if pixwide % 2 == 0:
        pixwide += 1 # make odd width
    kk = np.zeros((pixwide,pixwide),dtype=np.float64)
    pixst = float(pixwide)/2.
    xcen = np.linspace(-pixst, pixst, pixwide)
    # Gaussian and its derivative
    gaussfun   = lambda x,s: 1/(np.sqrt(2*np.pi)*s) * np.exp(-0.5*(x/s)**2)
    gaussderiv = lambda x,s: 1/(np.sqrt(2*np.pi)*s) * np.exp(-0.5*(x/s)**2) * (-x/(s*s))
    if deriv:
        fcen = gaussderiv(xcen, stdv)
    else:
        fcen = gaussfun(xcen, stdv)
    # set columns of the center row to the Gaussian function
    kk[(pixwide-1)//2,:] = fcen
    # rotate counterclockwise, angle in degrees
    rotationmatrix = cv2.getRotationMatrix2D((pixst,pixst), angle, 1.0)
    krot = cv2.warpAffine(kk, rotationmatrix, (pixwide,pixwide))
    return krot

if False:
    gkern10 = buildkern( 0., 8.)
    gkern01 = buildkern(90., 8.)

    dkern10 = buildkern( 0., 8., deriv=True)
    dkern01 = buildkern(90., 8., deriv=True)

    gf10 = cv2.filter2D(cvim, -1, gkern10)
    gf11 = cv2.filter2D(gf10, -1, gkern01)
    gf01 = cv2.filter2D(cvim, -1, gkern01)

    dd10 = cv2.filter2D(cvim, -1, dkern10)
    dd01 = cv2.filter2D(cvim, -1, dkern01)

    imshow("cvim",cvim)
    imshow("gf10",gf10)
    imshow("gf01",gf01)
    imshow("gf11",gf11)
    imshow("dd10",dd10)
    imshow("dd01",dd01)


thestdv = 5.
offs = 0.
#angles = [30.+offs, -30.+offs, 90.+offs]
angles = [0,0,0]
anglebasis = get_permutohedral_basis(2)
for ii in range(3):
    angles[ii] = np.arctan2(anglebasis[ii,1], anglebasis[ii,0]) * 180.0/np.pi
print("angles: "+str(angles))
filts = []

Amat = anglebasis

true_ggx = cv2.filter2D(cvim, -1, buildkern( 0., thestdv, deriv=True))
true_ggy = cv2.filter2D(cvim, -1, buildkern(90., thestdv, deriv=True))

for ang in angles:
    filts.append(cv2.filter2D(cvim, -1, buildkern(ang, thestdv, deriv=True)))
    imshow(str(ang),filts[-1])

simpdesc("Amat",Amat)
simpdesc("anglebasis",anglebasis)

cv2.waitKey(0)

farr = np.array(filts).reshape((3,-1))

Ano = np.dot( npl.pinv(np.dot(Amat.transpose(),Amat)), Amat.transpose() )

describe("farr",farr)
describe("Amat",Amat)
describe("Ano",Ano)

warpedfar = np.dot(Ano, farr)

describe("warpedfar", warpedfar)

reshwafa = np.reshape(warpedfar, [Ano.shape[0],]+list(cvim.shape))

describe("reshwafa",reshwafa)

print("Ano == ")
print(str(Ano))
print(" ")

allmin =  1e18
allmax = -1e18
allmin = min(allmin, np.amin(true_ggx))
allmax = max(allmax, np.amax(true_ggx))
allmin = min(allmin, np.amin(true_ggy))
allmax = max(allmax, np.amax(true_ggy))
allmin = min(allmin, np.amin(reshwafa))
allmax = max(allmax, np.amax(reshwafa))

imshow("gg_x", true_ggx, allmin, allmax)
imshow("gg_y", true_ggy, allmin, allmax)

for ii in range(reshwafa.shape[0]):
    imshow("newgrad "+str(ii), reshwafa[ii,...], allmin, allmax)

def normedfabs(a,b):
    return np.fabs(a-b)
    #return 2.0 * np.fabs(a-b) / (np.fabs(a)+np.fabs(b)+1e-18)

err_ggx = normedfabs(true_ggx, reshwafa[0,...])
err_ggy = normedfabs(true_ggy, reshwafa[1,...])
describe("err_ggx",err_ggx)
describe("err_ggy",err_ggy)
imshow("err_ggx", err_ggx, allmin, allmax)
imshow("err_ggy", err_ggy, allmin, allmax)

cv2.waitKey(0)


#
