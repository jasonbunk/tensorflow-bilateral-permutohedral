import os,sys
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
from test_utils import *
def simpdesc(name,arr):
    print(str(name))
    print(str(arr))
    print(" ")

def isometric_project(rowvecs):
    assert rowvecs.shape[0] == 3 # only for 3D --> 2D
    rotmat = np.zeros((3, 3),dtype=np.float64)
    rotmat[0,:] = np.array([np.sqrt(3.),       0,     -np.sqrt(3.)])
    rotmat[1,:] = np.array([    1.,            2.,         1.     ])
    rotmat[2,:] = np.array([np.sqrt(2.), -np.sqrt(2.), np.sqrt(2.)])
    simpdesc("rotmat",rotmat)
    rotmat /= np.sqrt(6.)
    isomat = np.eye(3).astype(np.float64)
    isomat[2,2] = 0.
    simpdesc("isomat",isomat)
    transfmat = np.dot(rotmat.transpose(), isomat.transpose())
    simpdesc("transfmat",transfmat)
    return np.dot(rowvecs, transfmat)

def perm_project_d(rowvecs, d_):
    assert rowvecs.shape[0] == d_+1
    # see page 5 of "Fast High-Dimensional Filtering Using the Permutohedral Lattice",
    #                Andrew Adams, Jongmin Baek, Myers Abraham Davis, Eurographics
    e1 = np.triu(np.ones((d_,d_),dtype=np.float64))
    e2 = np.eye(d_).astype(np.float64)
    for ii in range(d_):
        e1[ii,ii] = -1.*float(ii+1)
        e2[ii,ii] /= np.sqrt(float((ii+1)*(ii+2)))
    e1 = np.concatenate((np.ones((1,d_)),e1))
    Emat = np.dot(e1,e2)
    simpdesc("e1",e1)
    simpdesc("e2",e2)
    simpdesc("Emat",Emat)
    return np.dot(rowvecs, Emat)

def perm_project_dp1(rowvecs, d_):
    assert rowvecs.shape[0] == d_+1
    projvecT = np.ones((d_+1,1), dtype=np.float64)
    projvec = projvecT.transpose()
    #print(str(projvecT))
    #describe("projvecT",projvecT)
    allscale = np.sum(np.square(projvecT))
    projper = np.dot(rowvecs, projvecT) / allscale

    THEDOTS = np.dot(projper,projvec)
    projs = rowvecs - THEDOTS

    describe("rowvecs",rowvecs)
    describe("allscale",allscale)
    describe("projper",projper)
    simpdesc("THEDOTS",THEDOTS)
    describe("projs",projs)

    return projs

def get_permutohedral_basis(dim):
    basis_dp1 = np.eye(dim+1).astype(np.float64)
    rawprojbasis = perm_project_d(basis_dp1, dim)
    projnorms = np.reciprocal(np.sqrt(np.sum(np.square(rawprojbasis),axis=1,keepdims=True)))
    simpdesc("projnorms", projnorms)
    return rawprojbasis * projnorms

if __name__ == '__main__':
    import sys
    try:
        d_ = int(sys.argv[1])
    except:
        print("usage:   {d_}")
        quit()
    projbasis = get_permutohedral_basis(d_)

    simpdesc("projbasis",projbasis)
    projdotproj = np.dot(projbasis.transpose(), projbasis)
    simpdesc("projdotproj",projdotproj)

    Amat = projbasis
    Ano = np.dot( npl.pinv(np.dot(Amat.transpose(),Amat)), Amat.transpose() )
    simpdesc("Ano",Ano)

    Anoalt = Amat.transpose() * float(d_)/float(d_+1)
    simpdesc("Anoalt",Anoalt)

    diffbetwA = np.sum(np.fabs(Ano - Anoalt))
    print("Anoalt vs Ano: sum of differences: "+str(diffbetwA))

    quit()

    print(" ")
    simpdesc("projbasis",projbasis)
    quit()

    if projbasis.shape[1] == 2:
        assert projbasis.shape[0] == d_+1
        plt.figure()
        for ii in range(d_+1):
            plt.plot([0., projbasis[ii,0]], [0., projbasis[ii,1]])
    else:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for ii in range(d_+1):
            ax.plot([0., projbasis[ii,0]], [0., projbasis[ii,1]], [0., projbasis[ii,2]])

    plt.show()
