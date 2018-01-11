#!/usr/bin/env python

import numpy as np
from scipy.integrate import ode
from scipy.sparse.linalg import gmres
import RegularizedStokeslets2D as RS2D
import SpatialDerivs2D as SD2D

def makeGrid(N,M,gridspc,origin=(0,0)):
    '''N is the number of points in the x direction, M is the number of points in y direction, gridspc is the uniform point separation in both x and y directions. Optional argument origin gives the lower left corner of the domain.'''
    l0 = np.zeros((N,M,2))
    x = np.zeros((N,1))
    x[:,0] = np.linspace(origin[0]+gridspc/2,origin[0]+N*gridspc-gridspc/2,N)
    l0[:,:,0] = np.tile(x,(1,M))
    y = np.linspace(origin[1]+gridspc/2,origin[1]+M*gridspc-gridspc/2,M)
    l0[:,:,1] = np.tile(y,(N,1))
    return l0
    
def matInv2x2(M):
    det = M[0,0]*M[1,1] -M[0,1]*M[1,0]
    B = np.asarray([ [M[1,1],-M[0,1]], [-M[1,0],M[0,0]] ])
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
    
def calcForcesSimpleSpring(fpts,xr,K,**kwargs):
    '''
    current position 'fpts' and resting positions 'xr' should be length 2 vectors. 
    'K' is the scalar spring constant. **kwargs currently has no needed entries -- 
    it's here for consistent API.
    
    '''
    f = -K*(fpts - xr)
    return f        
    
def calcForcesTwoHeadSpring(fpts,xr,K,**kwargs):
    '''
    current position 'fpts' should be a 2 x 2 matrix (one row vector containing 
    x and y components for each bead). resting length 'xr' and spring constant 
    'K' are scalars. **kwargs currently has no needed entries -- it's here for
    consistent API.
    
    '''
    distvec = fpts[0,:] - fpts[1,:]
    dist = np.sqrt(np.sum(distvec**2))
    g = -K*(1 - xr/dist)*distvec
    return np.row_stack([g,-g])
    
def calcForcesGravity(fpts,xr,K,**kwargs):
    '''
    current position fpts should be an n x 2 matrix (one row vector containing 
    x and y components for each bead). xr is not used, K is the strength of the 
    gravity force, and kwargs['n'] is the number of the force points that are 
    actually carrying gravity (the rest are marker points).
    
    '''
    g = np.zeros(fpts.shape)
    g[:kwargs['n'],1] = -K
    return g
    
def calcForcesQuadraticSpring(fpts,xr,K,**kwargs):
    '''
    current position fpts should be a 2 x 2 matrix (one row vector containing 
    x and y components for each bead). resting length xr and spring constant 
    K are scalars. **kwargs currently has no needed entries -- it's here for
    consistent API.
    
    '''
    distvec = fpts[0,:] - fpts[1,:]
    dist = np.sqrt(np.sum(distvec**2))
    g = -K*(dist/xr - 1)*distvec
    return np.row_stack([g,-g])

def calcForcesSwimmer(fpts,xr,K,**kwargs):
    '''
    Calculate spring and curvature forces due to a moving swimmer:
    x = s; y = a*s*sin(lam*s - w*t) 
    Current position fpts should be an n x 2 matrix (one row vector containing 
    x and y components for each point on the swimmer). Resting length xr and 
    spring constant K are scalars describing springs between adjacent points.
    a, w, lam, Kcurv, t should all be in kwargs. a is the maximum
    amplitude of the sine wave at the tail; w is the angular frequency of the 
    sine wave; lam is the wave number; Kcurv is the curvature stiffness; and t 
    is the current time.
    
    '''

    dx = fpts[1:,0]-fpts[:-1,0]
    dy = fpts[1:,1]-fpts[:-1,1]

    #spring forces first
    sep = np.sqrt(dx**2 + dy**2)
#    fx = -K*(sep/xr - 1)*dx/xr
#    fy = -K*(sep/xr - 1)*dy/xr
    fx = -K*(sep/xr - 1)*dx/sep
    fy = -K*(sep/xr - 1)*dy/sep
    F1 = np.zeros(fpts.shape)
    F1[1:,0] = fx
    F1[1:,1] = fy
    F1[:-1,0] = F1[:-1,0] - fx
    F1[:-1,1] = F1[:-1,1] - fy
    
    #now curvature forces
    #first calculate desired curvature for yt = kwargs['a']*xt*np.sin(kwargs['lam']*xt-kwargs['w']*kwargs['t'])
    #a=kwargs['a']
    a = min([kwargs['a'],kwargs['t']*kwargs['a']]) #ramp up to full amplitude    
    Np = fpts.shape[0]
    xt = np.arange(xr,(Np-1)*xr,xr)
    curv = -a*kwargs['lam']**2*xt*np.sin(kwargs['lam']*xt-kwargs['w']*kwargs['t']) + 2*a*kwargs['lam']*np.cos(kwargs['lam']*xt-kwargs['w']*kwargs['t'])
    #now calculate approximate actual curvature
    numcurv = ( dx[1:]*dy[:-1] - dx[:-1]*dy[1:] )/xr**3
    coeff = kwargs['Kcurv']*(numcurv-curv)/xr**2
    F2 = np.zeros(fpts.shape)
    F2[2:,0] = coeff*dy[:-1]
    F2[2:,1] = -coeff*dx[:-1]
    F2[1:-1,0] = F2[1:-1,0] - coeff*(dy[:-1]+dy[1:])
    F2[1:-1,1] = F2[1:-1,1] + coeff*(dx[:-1]+dx[1:])
    F2[:-2,0] = F2[:-2,0]   + coeff*dy[1:]
    F2[:-2,1] = F2[:-2,1]   - coeff*dx[1:]
   
   
#    #for loop version like Ricardo's code
#    F2 = np.zeros(fpts.shape)
#    xt = np.arange(0,Np*xr,xr)
#    curv = -a*kwargs['lam']**2*xt*np.sin(kwargs['lam']*xt-kwargs['w']*kwargs['t']) + 2*a*kwargs['lam']*np.cos(kwargs['lam']*xt-kwargs['w']*kwargs['t'])
#    for k in range(1,Np-1):
#        dxk=fpts[k,0]-fpts[k-1,0]
#        dxkp=fpts[k+1,0]-fpts[k,0]
#        dyk=fpts[k,1]-fpts[k-1,1]
#        dykp=fpts[k+1,1]-fpts[k,1]
#        
#        numcurv=(dxkp*dyk-dxk*dykp)/xr**3
#        coeff = kwargs['Kcurv']*(numcurv-curv[k])/xr**2
#        F2[k+1,0] = F2[k+1,0]+coeff*dyk 
#        F2[k+1,1] = F2[k+1,1]-coeff*dxk 
#    
#        F2[k,0] = F2[k,0]-coeff*(dyk+dykp)        
#        F2[k,1] = F2[k,1]+coeff*(dxk+dxkp)
#    
#        F2[k-1,0] = F2[k-1,0]+coeff*dykp 
#        F2[k-1,1] = F2[k-1,1]-coeff*dxkp 
    
    
    return F1-F2
        
def stokesFlowUpdater(t,y,pdict):
    '''
    t = current time, y = [fpts.flatten(), l.flatten(), P.flatten()], pdict 
    contains: K is spring constant, xr is resting position, blob is regularized 
    Stokeslet object, myForces is a function handle, forcedict is a dictionary
    containing optional parameters for calculating forces.
    
    '''
    if 'forcedict' not in pdict.keys():
        pdict['forcedict'] = {}
    pdict['forcedict']['t'] = t
    Q=len(y)/2
    fpts = np.reshape(y,(Q,2))
    f = pdict['myForces'](fpts,pdict['xr'],pdict['K'],**pdict['forcedict'])
    yt = np.zeros((Q,2))
    for k in range(Q):
        ls = fpts[k,:]
        ls = ls[np.newaxis,:]
        yt[k,:] = pdict['blob'].linOp(ls,fpts,f)
    return yt.flatten()
            
def viscoElasticUpdaterKernelDeriv(t,y,pdict):
    #split up long vector into individual sections (force pts, Lagragian pts, stress components)
    if 'forcedict' not in pdict.keys():
        pdict['forcedict'] = {}
    pdict['forcedict']['t'] = t
    N = pdict['N']
    M = pdict['M']
    Q = len(y)/2 - N*M - 2*N*M
    fpts = np.reshape(y[:2*Q],(Q,2))
    l = y[range(2*Q,2*Q+2*N*M)]
    l2col = np.reshape(l,(N*M,2))
    l3D = np.reshape(l,(N,M,2))
    allpts = y[:2*Q+2*N*M] #both force points and Lagrangian points
    P = np.reshape(y[(2*Q+2*N*M):],(N,M,2,2))
    #calculate tensor derivative
    Pd = pdict['beta']*SD2D.tensorDiv(P,pdict['gridspc'],N,M)
    Pd = np.reshape(Pd,(N*M,2))
    #calculate spring forces
    f = pdict['myForces'](fpts,pdict['xr'],pdict['K'],**pdict['forcedict'])
    #calculate deformation matrix
    gl = SD2D.vectorGrad(l3D,pdict['gridspc'],N,M)
    #calculate new velocities at all points of interest (Lagrangian points and force points)
    lt = np.zeros(allpts.shape)
    for k in range(len(allpts)/2):
        ls = allpts[2*k:2*k+2]
        ls = ls[np.newaxis,:]
        lt[2*k:(2*k+2)] = pdict['gridspc']**2*pdict['blob'].linOp(ls,l2col,Pd) + pdict['blob'].linOp(ls,fpts,f)    
    #calculate new stress time derivatives
    Pt = np.zeros((N,M,2,2))
    for j in range(N):
        for k in range(M):
            #calculate inverse deformation matrix
            ls = l3D[j,k,:]
            ls = ls[np.newaxis,:]
            gls = gl[j,k,:,:]
            igls = matInv2x2(gls)
            gls = gls[np.newaxis,:,:]
            #calculate Stokeslet derivative term
            glst = pdict['gridspc']**2*pdict['blobderiv'].linOp(ls,l2col,Pd,gls) + pdict['blobderiv'].linOp(ls,fpts,f,gls)    
            #stress time derivative
            Ps = P[j,k,:,:]
            Pt[j,k,:,:] = np.dot(np.dot(glst,igls),Ps) - (1./pdict['Wi'])*(Ps - igls.transpose())
            #check to make sure Eulerian stress is symmetric
            # S=np.dot(Ps,gls.transpose())
            # if max(abs(S[0,1]),abs(S[1,0])) > 1.e-2 and np.abs(S[0,1] - S[1,0])/max(abs(S[0,1]),abs(S[1,0])) > 5.e-2:
            #     print("Warning: stress tensor is not symmetric. Off diagonal elements shown below.")
            #     print([S[0,1], S[1,0]]) #check to see if the stress tensor is not symmetric
    return np.append(lt,Pt.flatten())
    
def mySolver(myodefunc,y0,t0,dt,totalTime,method,pdict,numskip):
    '''
    myodefunc is f in y' = f(y); y0, t0 are initial conditions; 
    dt is time step, totalTime is the stopping time for the integration; 
    pdict is a dictionary containing the parameters for the update function f.
    
    '''
    r = ode(myodefunc).set_integrator('vode',method=method,rtol=1.e-3).set_initial_value(y0,t0).set_f_params(pdict)
    N = pdict['N']
    M = pdict['M']
    Q = len(y0) - 2*N*M - 4*N*M
    fpts=[]; l = []; S=[]; Strace = []; t=[]
    c=0
    while r.successful() and r.t < totalTime+dt/2.:
        if Q < 0 and np.mod(c,numskip) == 0: #stokes flow
            print(r.t)
            t.append(r.t)
            fpts.append(r.y)
        elif Q > 0 and np.mod(c,numskip) == 0: #in the viscoelastic case, record Lagrangian variables
            print(r.t)
            t.append(r.t)
            fpts.append(r.y[:Q])
            l3D = np.reshape(r.y[range(Q,Q+2*N*M)],(N,M,2))
            l.append(l3D)
            gl = SD2D.vectorGrad(l3D,pdict['gridspc'],N,M)            
            Ptemp = np.reshape(r.y[(Q+2*N*M):],(N,M,2,2))
            Stemp = np.zeros((N,M,2,2))
            tr = np.zeros((N,M))
            for j in range(N):
                for k in range(M):
                    gls = gl[j,k,:,:]
                    Ps = Ptemp[j,k,:,:]
                    Stemp[j,k,:,:]=np.dot(Ps,gls.transpose())
                    tr[j,k] = Stemp[j,k,0,0] + Stemp[j,k,1,1]
            S.append(Stemp)
            Strace.append(tr)
        c+=1
        r.integrate(r.t+dt)
    return fpts, t, l, S, Strace

def mySwimmer():
    #get ready to save files
    import mat2py
    import os

    #dictionary of variables for both Stokes flow and viscoelastic flow
    pdict = dict( N = 40, M = 16, gridspc = 1.5/40, origin = [0,0], mu = 1, Wi = 0.2)
    pdict['beta'] = 1./(2*pdict['Wi'])
   
    #set time parameters, save data every numskip time steps
    t0 = 0; totalTime = 1; dt = 1.e-4; numskip = 500;
    
    #save file name
    fname = os.path.expanduser('~/scratch/swimmer')
    
    #make swimmer
    #first assign params
    a =0.1
    w = 2*np.pi  
    lam = 8.4 #2*np.pi/0.75
    Np = 20
    L = 0.78
    h = L/(Np-1)
    pdict['K'] =100
    pdict['xr'] = np.array([h])
    pdict['forcedict'] = {'a':a, 'w':w,'t':0,'Kcurv':0.25,'lam':lam}
    pdict['eps'] = 0.75*h
    #Now setup initial conditions
    y0 = 0.3*np.ones((2*Np,))
    v = np.arange(0.5,0.5+L+h/2,h)
    y0[:-1:2] = v #initial conds [x0,y0,x1,y1,...,xn,yn]
    
    #add unsaveable entries
    pdict1 = pdict.copy()
    pdict1['myForces'] = calcForcesSwimmer
    pdict1['blob'] = RS2D.CubicRegStokeslet(pdict['eps'],pdict['mu'])
    pdict1['blobderiv'] = RS2D.CubicRegStokesletDeriv(pdict['eps'],pdict['mu'])
 
    #run the ode solver
    print('Stokes flow...')
    fpts, t, l, S, Strace = mySolver(stokesFlowUpdater,y0,t0,dt,totalTime,'bdf',pdict1,numskip)
    mat2py.write(fname+'_stokes_RicardoParams_08162011.mat', {'t':t,'fpts':fpts,'pdict':pdict,'dt':dt})

#    #make the grid
#    N=pdict['N']
#    M=pdict['M']
#    gridspc = pdict['gridspc']
#    l0 = makeGrid(N,M,gridspc,pdict['origin'])
#    P0 = np.zeros((N,M,2,2))
#    P0[:,:,0,0] = 1
#    P0[:,:,1,1] = 1
#    y0 = np.append(np.append(y0, l0.flatten()),P0.flatten())
#
#    #Initialize swimmer
#
#    #Viscoelastic run
#    print('VE flow...')
#    fpts, t, l, S, Strace = mySolver(viscoElasticUpdaterKernelDeriv,y0,t0,dt,totalTime,'bdf',pdict1,numskip)
#    #save the output
#    mat2py.write(fname+'_visco_40.mat', {'t':np.asarray(t),'fpts':np.asarray(fpts),'l':np.asarray(l),'S':np.asarray(S),'Strace':np.asarray(Strace),'pdict':pdict,'dt':dt})


def mySprings():
    #get ready to save files
    import mat2py, os

    #dictionary of variables for both Stokes flow and viscoelastic flow
    pdictsave = dict( N = 20, M = 20, gridspc = 1./20, origin = [0,0], mu = 1, K =5, xr = np.array([0.1]), Wi = 0.2)
    pdictsave['beta'] = 1./(2*pdictsave['Wi'])
    pdictsave['eps'] = 4*pdictsave['gridspc']
    pdict = pdictsave.copy()
    pdict['myForces'] = calcForcesQuadraticSpring
    pdict['blob'] = RS2D.CubicRegStokeslet(pdict['eps'],pdict['mu'])
    pdict['blobderiv'] = RS2D.CubicRegStokesletDeriv(pdict['eps'],pdict['mu'])
    gridspc = pdict['gridspc']
    
    #set time parameters, save data every numskip time steps
    t0 = 0; totalTime = 1; dt = 1.e-3; numskip = 100;
    
    #save file name
    fname = os.path.expanduser('~/scratch/springtest')
        
    #Stokes flow test, works
    print('Stokes flow...')
    #set up initial conditions
    # margin = (N*gridspc-0.2)/2
    # y0 = np.array([margin,M*gridspc/2,N*gridspc - margin,M*gridspc/2])
    y0 = np.array([0.4,0.5,0.6,0.5])
    #run the ode solver
    fpts, t, l, S, Strace = mySolver(stokesFlowUpdater,y0,t0,dt,totalTime,'bdf',pdict,numskip)
    #save the output
    mat2py.write(fname+'_stokes_noC.mat', {'t':t,'fpts':fpts,'pdict':pdictsave,'dt':dt})
    
    #Viscoelastic run
    print('VE flow...')
    #use same initial conditions for spring as in Stokes flow, but add on Lagrangian points and stress
    N=pdict['N']
    M=pdict['M']
    l0 = makeGrid(N,M,gridspc,pdict['origin'])
    P0 = np.zeros((N,M,2,2))
    P0[:,:,0,0] = 1
    P0[:,:,1,1] = 1
    y0 = np.append(np.append(y0, l0.flatten()),P0.flatten())
    #run the ode solver
    fpts, t, l, S, Strace = mySolver(viscoElasticUpdaterKernelDeriv,y0,t0,dt,totalTime,'bdf',pdict,numskip)
    #save the output
    mat2py.write(fname+'_visco_noC.mat', {'t':np.asarray(t),'fpts':np.asarray(fpts),'l':np.asarray(l),'S':np.asarray(S),'Strace':np.asarray(Strace),'pdict':pdictsave,'dt':dt})

def fallingStar():
    ########################################################
    #falling star for art show
    ########################################################
    #get ready to save files
    import mat2py
    #dictionary of variables 
    pdictsave = dict( N = 80, M = 80, gridspc = 1./100, origin = [0,0], mu = 1, K =0.05, xr = [], Wi = 1)
    pdictsave['beta'] = 1./(2*pdictsave['Wi'])
    pdictsave['eps'] = 4*pdictsave['gridspc']
    pdict = pdictsave.copy()
    pdict['myForces'] = calcForcesGravity
    pdict['blob'] = RS2D.CubicRegStokeslet(pdict['eps'],pdict['mu'])
    pdict['blobderiv'] = RS2D.CubicRegStokesletDeriv(pdict['eps'],pdict['mu'])
    N = pdict['N']
    M = pdict['M']
    gridspc = pdict['gridspc']
    
    #set time parameters
    t0 = 0; totalTime = 0.45; dt = 1.e-2; numskip = 1;
    
    #save file name
    fname = 'scratch/star_tree2'
    
    
    #make the star
    R = 0.05; d = 0.05; r = 0.03
    spc = 2*np.pi/16
    theta = np.arange(0,6*np.pi+spc,spc)
    x = (R-r)*np.cos(theta) + d*np.cos(theta*(R-r)/r) + 0.35
    y = (R-r)*np.sin(theta) - d*np.sin(theta*(R-r)/r) + 0.45
    fpts = np.column_stack([x,y])
    pdictsave['n']=fpts.shape[0]
    pdict['n']=fpts.shape[0]
    
    #make tree marker points
    lx=np.arange(0,0.8+gridspc/2,gridspc)
    llinex = 0.4-lx*np.cos(np.pi/3)
    lliney = 0.8-lx*np.sin(np.pi/3)
    bllinex = np.arange(0,0.35+gridspc/2,gridspc)
    blliney = 0.1*np.ones(bllinex.shape)
    mx = np.append(llinex,bllinex)
    my = np.append(lliney,blliney)
    ltrunky = np.arange(0.1,-gridspc/2,-gridspc)
    ltrunkx = 0.35*np.ones(ltrunky.shape)
    mx = np.append(mx,ltrunkx)
    my = np.append(my,ltrunky)
    btrunkx = np.arange(0.35,0.45+gridspc/2,gridspc)
    btrunky = np.zeros(btrunkx.shape)
    mx = np.append(mx,btrunkx)
    my = np.append(my,btrunky)
    rtrunky = np.arange(0,0.1+gridspc/2,gridspc)
    rtrunkx = 0.45*np.ones(rtrunky.shape)
    mx = np.append(mx,rtrunkx)
    my = np.append(my,rtrunky)    
    brlinex = np.arange(0.45,0.8+gridspc/2,gridspc)
    brliney = blliney
    mx = np.append(mx,brlinex)
    my = np.append(my,brliney)    
    rx = lx[range(len(lx)-1,-1,-1)]
    rlinex = 0.4+rx*np.cos(np.pi/3)
    rliney = 0.8-rx*np.sin(np.pi/3)
    mx = np.append(mx,rlinex)
    my = np.append(my,rliney)
    mpts = np.column_stack([mx,my])
    
    #make the grid and stress ICs
    l0 = makeGrid(N,M,gridspc,pdict['origin'])
    P0 = np.zeros((N,M,2,2))
    P0[:,:,0,0] = 1
    P0[:,:,1,1] = 1
    y0 = np.append(np.append(np.append(fpts.flatten(), mpts.flatten()), l0.flatten()),P0.flatten())
    #run the ode solver
    fpts, t, l, S, Strace = mySolver(viscoElasticUpdaterKernelDeriv,y0,t0,dt,totalTime,'bdf',pdict,numskip)
    #save the output
    mat2py.write(fname+'_visco.mat', {'t':np.asarray(t),'fpts':np.asarray(fpts),'l':np.asarray(l),'S':np.asarray(S),'Strace':np.asarray(Strace),'pdict':pdictsave,'dt':dt})

    
if __name__ == '__main__':
#    mySprings()    
    mySwimmer()
    
    
    
    
    
    
    
