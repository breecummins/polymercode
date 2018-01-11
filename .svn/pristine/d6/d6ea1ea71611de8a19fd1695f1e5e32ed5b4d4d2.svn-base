import numpy as nm
import RegularizedStokeslets2D as RS2D
import main

if __name__ == '__main__':
    nm.set_printoptions(threshold=nm.nan,linewidth=78)
    N = 17; M = 11; gridspc = 0.025
    eps = gridspc/2; mu = 1
    blob = RS2D.CubicRegStokeslet(eps,mu)
    l = main.makeGrid(N,M,gridspc)
    X = l[:,:,0].flatten()
    Y = l[:,:,1].flatten()
    L=nm.column_stack([X,Y])
    f = nm.exp(-(X-0.2)**2 - (Y-0.1)**2)
    F = nm.column_stack([-f,f])/nm.sqrt(2)
    upt = nm.array([0.3,0.2])
    ###### Original method #########
    u = blob.linOp(upt,L,F)
    print u*gridspc**2
    ###### Convoluted method #######
    r0, r1 = blob.kernValsOriginal(upt,L,N*M)
    ux = main.posHelper(r0,F.flatten(),N,M,gridspc)
    uy = main.posHelper(r1,F.flatten(),N,M,gridspc)
    print [ux,uy]
    
    
    