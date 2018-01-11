#!/usr/bin/env python

import numpy as nm
from scipy.integrate import ode
from scipy.sparse.linalg import gmres
import RegularizedStokeslets2D as RS2D
import SpatialDerivs2D as SD2D
import sys

def makeGrid(N,M,gridspc):
    '''N is the number of points in the x direction, M is the number of points in y direction, gridspc is the uniform point separation in both x and y directions.'''
    l0 = nm.zeros((N,M,2))
    x = nm.zeros((N,1))
    x[:,0] = nm.linspace(0,(N-1)*gridspc,N)
    l0[:,:,0] = nm.tile(x,(1,M))
    y = nm.linspace(0,(M-1)*gridspc,M)
    l0[:,:,1] = nm.tile(y,(N,1))
    return l0
    
# def trapRule2D(V,gridspc):
#     '''V is a 2D array with function values associated with points from makeGrid. I.e., V[i,j] is the value at the point (l0[i,j,0], l0[i,j,1]).'''
#     area = gridspc**2*nm.sum(V[1:-1,1:-1]) +  (gridspc**2 / 2)*(nm.sum(V[0,1:-1]) + nm.sum(V[-1,1:-1]) + nm.sum(V[1:-1,0]) + nm.sum(V[1:-1,-1])) +  (gridspc**2 / 4)*(nm.sum(V[0,0]) + nm.sum(V[-1,0]) + nm.sum(V[-1,-1]) + nm.sum(V[0,-1]))
#     return area
    
def matInv2x2(M):
    det = M[0,0]*M[1,1] -M[0,1]*M[1,0]
    B = nm.asarray([ [M[1,1],-M[0,1]], [-M[1,0],M[0,0]] ])
    return B/float(det)

def useGMRES(A,RHS,rs=None):
    forces, info = gmres(A,RHS,restart=rs)
    if info == 0:
        pass
    elif info > 0:
        print('Maximum iterations reached: '+str(info))
    else:
        print('gmres encountered an error.')
    return forces
    
def calcForcesSimpleSpring(fpts,xr,K):
    '''current position 'fpts' and resting positions 'xr' should be length 2 vectors. 'K' is the scalar spring constant.'''
    f = -K*(fpts - xr)
    return f        
    
def calcForcesTwoHeadSpring(fpts,xr,K):
    '''current position 'fpts' should be a 2 x 2 matrix (one row vector containing x and y components for each bead). resting length 'xr' and spring constant 'K' are scalars.'''
    distvec = fpts[0,:] - fpts[1,:]
    dist = nm.sqrt(nm.sum(distvec**2))
    udv = distvec/dist
    g = -K*(dist - xr)*udv
    return nm.row_stack([g,-g])

def updatePosStokes(ls,blob,fpts,f):
    ''''ls' is the current location of a single point, 'blob' is the regularizer object, 'fpts' are the current positions of the forces 'f'. Returns the velocity at 'ls' due to 'f' at 'fpts'.'''
    return blob.linOp(ls,fpts,f)

def updatePosVisco(ls,blob,fpts,f,l,gridspc,N,M,P,beta):
    '''ls is the current location of a single point, blob is the regularizer object, fpts are the current positions of the forces f, grid is the current position of all points in the domain, gridspc is the spacing in the initial configuration, N is the numebr of points in the x direction, M is the number in the y direction, P is the extra stress tensor for all the grid, beta is a proportionality parameter for the extra stress.'''
    pd = beta*SD2D.tensorDiv(P,gridspc,N,M)
    pd = nm.reshape(pd,(N*M,2))
    lt = gridspc**2*blob.linOp(ls,l,pd) + blob.linOp(ls,fpts,f)    
    # #old convoluted method below
    # p = pd.flatten() #flatten will collapse the last dimension first, so that the x-value at a point is immediately followed by the y-value at a point 
    # r0, r1 = blob.kernValsOriginal(ls,l,N*M)
    # ufromp0 = posHelper(r0,p,N,M,gridspc)
    # lt[0] = lt[0] + ufromp0
    # ufromp1 = posHelper(r1,p,N,M,gridspc)
    # lt[1] = lt[1] + ufromp1
    return lt
    
# def posHelper(r,p,N,M,gridspc):
#     mmult = r*p
#     mm = mmult[0::2] + mmult[1::2]
#     integrand = nm.reshape(mm,(N,M))    
#     return trapRule2D(integrand,gridspc)
        
def updateStress(Ps,gls,glst,Wi):
    '''Ps is the current stress at a single point ls. The arguments gls and glst are the Lapracian gradient of ls and its velocity respectively. Wi is the Weissenberg number for the extra stress.'''
    igls = matInv2x2(gls)
    Pt = nm.dot(nm.dot(glst,igls),Ps) - (1./Wi)*(Ps - igls.transpose())
    return Pt
    
def stokesFlowUpdater(t,y,pdict):
    '''t = current time, y = [fpts.flatten(), l.flatten(), P.flatten()], pdict contains: K is spring constant, xr is resting position, blob is regularized Stokeslet object.'''
    Q=len(y)/2
    fpts = nm.reshape(y,(Q,2))
    f = pdict['myForces'](fpts,pdict['xr'],pdict['K'])
    yt = nm.zeros((Q,2))
    for k in range(Q):
        ls = RS2D.getPoint(k,fpts)
        yt[k,:] = updatePosStokes(ls,pdict['blob'],fpts,f)
    return yt.flatten()
            
def viscoElasticUpdater(t,y,pdict):
    N = pdict['N']
    M = pdict['M']
    Q = len(y)/2 - N*M - 2*N*M
    fpts = nm.reshape(y[:2*Q],(Q,2))
    l = y[range(2*Q,2*Q+2*N*M)]
    allpts = y[:2*Q+2*N*M]
    P = nm.reshape(y[(2*Q+2*N*M):],(N,M,2,2))
    if pdict['forceon'] == 'n':
        f = nm.zeros(fpts.shape)
        if nm.mod(t,5) < 1.e-4:
            print('Forces are zeroed out at t = %d' % t)
    else:
        f = pdict['myForces'](fpts,pdict['xr'],pdict['K'])
        if pdict['myForces'] == calcForcesSimpleSpring and (nm.sqrt(nm.sum((fpts - xr)**2)) < 1.e-3):
            pdict['forceon'] = 'n'
    gl = SD2D.vectorGrad(nm.reshape(l,(N,M,2)),pdict['gridspc'],N,M)
    lt = nm.zeros(allpts.shape)
    for k in range(len(allpts)/2):
        ls = allpts[2*k:2*k+2]
        lt[2*k:(2*k+2)] = updatePosVisco(ls,pdict['blob'],fpts,f,nm.reshape(l,(N*M,2)),pdict['gridspc'],N,M,P,pdict['beta'])
    glt = SD2D.vectorGrad(nm.reshape(lt[range(2*Q,2*Q+2*N*M)],(N,M,2)),pdict['gridspc'],N,M)
    Pt = nm.zeros((N,M,2,2))
    for j in range(N):
        for k in range(M):
            Ps = P[j,k,:,:]
            gls = gl[j,k,:,:]
            glst = glt[j,k,:,:]
            Pt[j,k,:,:] = updateStress(Ps,gls,glst,pdict['Wi'])
            S=nm.dot(Ps,gls.transpose())
            if max(abs(S[0,1]),abs(S[1,0])) > 1.e-10 and nm.abs(S[0,1] - S[1,0])/max(abs(S[0,1]),abs(S[1,0])) > 1.e-2:
                print("Warning: stress tensor is not symmetric. Off diagonal elements shown below.")
                print([S[0,1], S[1,0]]) #check to see if the stress tensor is not symmetric
    return nm.append(lt,Pt.flatten())

    
def mySolver(myodefunc,y0,t0,dt,totalTime,method,pdict):
    '''myodefunc is f in y' = f(y); y0, t0 are initial conditions; dt is time step, totalTime is the stopping time for the integration; pdict is a dictionary containing the parameters for the update function f.'''
    r = ode(myodefunc).set_integrator('vode',method=method).set_initial_value(y0,t0).set_f_params(pdict)
    Q = len(y0) - 2*N*M - 4*N*M
    fpts=[]; l = []; P = []; Ptrace = []; t=[]
    c=0
    while r.successful() and r.t < totalTime+dt/2.:
        if Q < 0 and nm.mod(c,1e2) == 0: #stokes flow
            t.append(r.t)
            fpts.append(r.y)
        elif Q > 0 and nm.mod(c,1e2) == 0: #in the viscoelastic case, record Lagrangian variables
            t.append(r.t)
            fpts.append(r.y[:Q])
            l.append(nm.reshape(r.y[range(Q,Q+2*N*M)],(N,M,2)))
            Ptemp = nm.reshape(r.y[(Q+2*N*M):],(N,M,2,2))
            tr = nm.zeros((N,M))
            for j in range(N):
                for k in range(M):
                    tr[j,k] = Ptemp[j,k,0,0] + Ptemp[j,k,1,1]
            P.append(Ptemp)
            Ptrace.append(tr)
        c+=1
        r.integrate(r.t+dt)
    return fpts, t, l, P, Ptrace
    
    
if __name__ == '__main__':
    '''Spec: I need 1D and 2D numpy arrays associated with each ij point in a 2D grid. The grid itself should be evenly spaced in both x and y directions (same spacing in both). I will need 3 and 4 dimensional arrays for vectors and tensors at these points. First index = i (x-coord), second index = j (y-coord), third index = row of tensor (or vector component), fourth index =  column of tensor. In particular, I need the vector l, which will have the (moving) x and y coords of the ij-th point in the third dimension; and I need the tensor P, which is a 2x2 matrix at each point.'''
            
    #get ready to save files
    import mat2py

    #dictionary of variables for both Stokes flow and viscoelastic flow
    pdictsave = dict( N = 35, M = 23, gridspc = 0.012, mu = 1, K =1, xr = nm.array([0.1]), beta = 1, Wi = 1, forceon = 'y')
    pdictsave['eps'] = 2*pdictsave['gridspc']
    pdict = pdictsave.copy()
    pdict['myForces'] = calcForcesTwoHeadSpring
    pdict['blob'] = RS2D.CubicRegStokeslet(pdict['eps'],pdict['mu'])
    N = pdict['N']
    M = pdict['M']
    gridspc = pdict['gridspc']
    
    #set time parameters
    t0 = 0; totalTime = 50; dt = 1.e-3; 
    
    #save file name
    fname = 'scratch/oldversion_twohead'
        
    #Stokes flow test, works
    print('Stokes flow...')
    #set up initial conditions
    margin = (N*gridspc-0.2)/2
    y0 = nm.array([margin,M*gridspc/2,N*gridspc - margin,M*gridspc/2])
    #run the ode solver
    fpts, t, l, P, Ptrace = mySolver(stokesFlowUpdater,y0,t0,dt,totalTime,'adams',pdict)
    #save the output
    mat2py.write(fname+'_stokes.mat', {'t':t,'fpts':fpts,'pdict':pdictsave,'dt':dt})
    
    #Viscoelastic run
    print('VE flow...')
    #use same initial conditions for spring as in Stokes flow, but add on Lagrangian points and stress
    l0 = makeGrid(N,M,gridspc)
    P0 = nm.zeros((N,M,2,2))
    P0[:,:,0,0] = 1
    P0[:,:,1,1] = 1
    y0 = nm.append(nm.append(y0, l0.flatten()),P0.flatten())
    #run the ode solver
    fpts, t, l, P, Ptrace = mySolver(viscoElasticUpdater,y0,t0,dt,totalTime,'bdf',pdict)
    #save the output
    mat2py.write(fname+'_visco.mat', {'t':nm.asarray(t),'fpts':nm.asarray(fpts),'l':nm.asarray(l),'P':nm.asarray(P),'Ptrace':nm.asarray(Ptrace),'pdict':pdictsave,'dt':dt})
    
    
    
    
    
    
    
    
    
