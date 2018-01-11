#import python modules
import numpy as np
from scipy.integrate import ode
import os, sys
from cPickle import Pickler
#import my modules
import SpatialDerivs2D as SD2D
import Gridding as mygrids
#import swimmervisualizer as sv
try:
    import pythoncode.cext.CubicStokeslet2D as CM
except:
    print('Please compile the C extension CubicStokeslet2D.c and put the .so in the cext folder.')
    raise(SystemExit)

def isIntDivisible(x,y,p=1.e-8):
    m = x/y
    return abs(round(m)-m) < p
    
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

def calcForcesConstant(fpts,xr,K,**kwargs):
    '''
    current position fpts should be an N x 2 matrix (one row vector containing 
    x and y components for each bead). constant forces xr should be an N x 2 
    matrix. K and **kwargs not needed-- here for consistent API.
    
    '''
    return xr


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
    #note that this is linear force density
    sep = np.sqrt(dx**2 + dy**2)
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
    xt = np.arange(xr,(Np-1.5)*xr,xr)
    curv = -a*kwargs['lam']**2*xt*np.sin(kwargs['lam']*xt-kwargs['w']*kwargs['t']) + 2*a*kwargs['lam']*np.cos(kwargs['lam']*xt-kwargs['w']*kwargs['t'])
    #now calculate approximate actual curvature
    numcurv = -( dx[1:]*dy[:-1] - dx[:-1]*dy[1:] )/xr**3
    coeff = kwargs['Kcurv']*(numcurv-curv)/xr**2
    F2 = np.zeros(fpts.shape)
    F2[2:,0] = coeff*dy[:-1]
    F2[2:,1] = -coeff*dx[:-1]
    F2[1:-1,0] = F2[1:-1,0] - coeff*(dy[:-1]+dy[1:])
    F2[1:-1,1] = F2[1:-1,1] + coeff*(dx[:-1]+dx[1:])
    F2[:-2,0] = F2[:-2,0]   + coeff*dy[1:]
    F2[:-2,1] = F2[:-2,1]   - coeff*dx[1:]   
    return F1+F2

def calcForcesSwimmerTFS(fpts,xr,K,**kwargs):
    '''
    Calculate spring and curvature forces due to a moving swimmer
    with target curvature as in Teran, Fauci, and Shelley.
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
    #note that this is linear force density
    sep = np.sqrt(dx**2 + dy**2)
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
    xt = np.arange(xr,(Np-1.5)*xr,xr)
    curv = -a*kwargs['lam']**2*(xt-1)*np.sin(kwargs['lam']*xt-kwargs['w']*kwargs['t'])
    #now calculate approximate actual curvature
    numcurv = -( dx[1:]*dy[:-1] - dx[:-1]*dy[1:] )/xr**3
    coeff = kwargs['Kcurv']*(numcurv-curv)/xr**2
    F2 = np.zeros(fpts.shape)
    F2[2:,0] = coeff*dy[:-1]
    F2[2:,1] = -coeff*dx[:-1]
    F2[1:-1,0] = F2[1:-1,0] - coeff*(dy[:-1]+dy[1:])
    F2[1:-1,1] = F2[1:-1,1] + coeff*(dx[:-1]+dx[1:])
    F2[:-2,0] = F2[:-2,0]   + coeff*dy[1:]
    F2[:-2,1] = F2[:-2,1]   - coeff*dx[1:]   
    return F1+F2

def FourRollMill(pdict,l,l2col):
    u = np.zeros(l.shape)
    u[::2] = pdict['U']*np.sin(l[::2])*np.cos(l[1::2])
    u[1::2] = -pdict['U']*np.cos(l[::2])*np.sin(l[1::2])
    ugrad = np.zeros((l2col.shape[0],2,2))
    ugrad[:,0,0] = pdict['U']*np.cos(l2col[:,0])*np.cos(l2col[:,1])
    ugrad[:,0,1] = -pdict['U']*np.sin(l2col[:,0])*np.sin(l2col[:,1])
    ugrad[:,1,0] = pdict['U']*np.sin(l2col[:,0])*np.sin(l2col[:,1])
    ugrad[:,1,1] = -pdict['U']*np.cos(l2col[:,0])*np.cos(l2col[:,1])
    return u, ugrad

def Extension(pdict,l,l2col):
    u = np.zeros(l.shape)
    u[::2] = pdict['U']*l[::2]
    u[1::2] = -pdict['U']*l[1::2]
    ugrad = np.zeros((l2col.shape[0],2,2))
    ugrad[:,0,0] = pdict['U']
    ugrad[:,1,1] = -pdict['U']
    return u, ugrad

def ParabolicShear(pdict,l,l2col):
    u = np.zeros(l2col.shape)
    yvals = l2col[:,1].copy()
    inds = np.nonzero(yvals < 0)
    u[:,0] = pdict['U']*yvals*(1-yvals) # y >= 0
    u[inds,0] = pdict['U']*yvals[inds]*(1+yvals[inds]) # y < 0
    u = u.flatten()
    ugrad = np.zeros((l2col.shape[0],2,2))
    ugrad[:,0,1] = 1 - 2*yvals # y >= 0
    ugrad[inds,0,1] = 1 + 2*yvals[inds] # y >= 0
    return u, ugrad

def regDipole(pdict,l,l2col):
    ''' 
    dipolef is the strength and dipolex0 the location of the dipole.
    
    '''
    u = np.zeros(l2col.shape)
    dx = l2col[:,0] - pdict['dipolex0'][0] 
    dy = l2col[:,1] - pdict['dipolex0'][1] 
    eps = 5*pdict['eps']
    r2 = dx**2 + dy**2 + eps**2
    D1 = (2*eps**2 - r2) / r2**2
    D2 = 2/r2**2
    f1 = pdict['dipolef'][0] 
    f2 = pdict['dipolef'][1]
    fdotx = f1*dx + f2*dy
    u[:,0] = ( f1*D1 + fdotx*dx*D2 ) / (2*np.pi*pdict['mu'])
    u[:,1] = ( f2*D1 + fdotx*dy*D2 ) / (2*np.pi*pdict['mu'])
    u = u.flatten()
    ugrad = np.zeros((l2col.shape[0],2,2))
    denom = np.pi*pdict['mu']*r2**3
    ugrad[:,0,0] = ( f1*dx*(4*dy**2 -   r2) + f2*dy*(-4*dx**2 + r2) ) / denom
    ugrad[:,0,1] = ( f1*dy*(4*dy**2 - 3*r2) + f2*dx*(-4*dy**2 + r2) ) / denom
    ugrad[:,1,0] = ( f2*dx*(4*dx**2 - 3*r2) + f1*dy*(-4*dx**2 + r2) ) / denom
    ugrad[:,1,1] = ( f2*dy*(4*dx**2 -   r2) + f1*dx*(-4*dy**2 + r2) ) / denom
    return u, ugrad

def circleFlow(pdict,l,l2col):
    u = np.zeros(l.shape)
    r2 = (l2col**2).sum(1)
    x = l[::2]
    y = l[1::2]
    u[::2] = -pdict['U']*y*np.exp(-r2)
    u[1::2] = pdict['U']*x*np.exp(-r2)
    ugrad = np.zeros((l2col.shape[0],2,2))
    ugrad[:,0,0] = pdict['U']*np.exp(-r2)*2*x*y
    ugrad[:,0,1] = pdict['U']*np.exp(-r2)*(-1+2*y**2)
    ugrad[:,1,0] = pdict['U']*np.exp(-r2)*(1-2*x**2)
    ugrad[:,1,1] = pdict['U']*np.exp(-r2)*(-2*x*y)
    return u, ugrad
       
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
    yt = CM.matmult(pdict['eps'],pdict['mu'],fpts,fpts,f)
    return yt

def stokesFlowUpdaterWithMarkers(t,y,pdict):
    if 'forcedict' not in pdict.keys():
        pdict['forcedict'] = {}
    pdict['forcedict']['t'] = t
    N = pdict['N']
    M = pdict['M']
    Q = len(y)/2 - N*M
    fpts = np.reshape(y[:2*Q],(Q,2))
    ap = np.reshape(y,(Q+N*M,2)) 
    #calculate spring forces
    f = pdict['myForces'](fpts,pdict['xr'],pdict['K'],**pdict['forcedict'])
    #calculate new velocities at all points of interest (Lagrangian points and force points)
    lt = CM.matmult(pdict['eps'],pdict['mu'],ap,fpts,f)
    return lt

            
def viscoElasticUpdaterKernelDeriv(t,y,pdict):
    #interior function for force pts only
    #split up long vector into individual sections (force pts, Lagrangian pts, stress components)
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
    ap = np.reshape(allpts,(Q+N*M,2)) 
    P = np.reshape(y[(2*Q+2*N*M):],(N,M,2,2))
    Ps = np.reshape(P,(N*M,2,2))
    #calculate tensor derivative
    Pd = pdict['beta']*SD2D.tensorDiv(P,pdict['gridspc'],N,M)
    Pd = np.reshape(Pd,(N*M,2))
    #calculate spring forces
    f = pdict['myForces'](fpts,pdict['xr'],pdict['K'],**pdict['forcedict'])
    #calculate deformation matrix and its inverse
    gl = SD2D.vectorGrad(l3D,pdict['gridspc'],N,M)
    gls = np.reshape(gl,(N*M,2,2))
    igls = CM.matinv2x2(gls)
    #calculate new velocities at all points of interest (Lagrangian points and force points)
    lt = pdict['gridspc']**2*CM.matmult(pdict['eps'],pdict['mu'],ap,l2col,Pd) + CM.matmult(pdict['eps'],pdict['mu'],ap,fpts,f)
    #calculate new stress time derivatives
    glst = pdict['gridspc']**2*CM.derivop(pdict['eps'],pdict['mu'],l2col,l2col,Pd,gls) + CM.derivop(pdict['eps'],pdict['mu'],l2col,fpts,f,gls)   
    Pt = np.zeros((N*M,2,2))
    for j in range(N*M):
        Pt[j,:,:] = np.dot(np.dot(glst[j,:,:],igls[j,:,:]),Ps[j,:,:]) - (1./pdict['Wi'])*(Ps[j,:,:] - igls[j,:,:].transpose())        
    return np.append(lt,Pt.flatten())

def veExtensionUpdater(t,y,pdict):
    #interior function for incompressible background flow only
    #split up long vector into individual sections (force pts, Lagrangian pts, stress components)
    N = pdict['N']
    M = pdict['M']
    l = y[range(2*N*M)]
    l2col = np.reshape(l,(N*M,2))
    l3D = np.reshape(l,(N,M,2))
    P = np.reshape(y[(2*N*M):],(N,M,2,2))
    Ps = np.reshape(P,(N*M,2,2))
    #calculate tensor derivative
    Pd = pdict['beta']*SD2D.tensorDiv(P,pdict['gridspc'],N,M)
    Pd = np.reshape(Pd,(N*M,2))
    #calculate deformation matrix and its inverse
    F = SD2D.vectorGrad(l3D,pdict['gridspc'],N,M)
    F = np.reshape(F,(N*M,2,2))
    Finv = CM.matinv2x2(F)
    #calculate new velocities at all points of interest (Lagrangian points and force points)
    ub, gradub = pdict['myVelocity'](pdict,l,l2col)
    lt = ub + pdict['gridspc']**2*CM.matmult(pdict['eps'],pdict['mu'],l2col,l2col,Pd) 
    #calculate new stress time derivatives
    gradlt = pdict['gridspc']**2*CM.derivop(pdict['eps'],pdict['mu'],l2col,l2col,Pd,F) 
    Pt = np.zeros((N*M,2,2))
    for j in range(N*M):
        Pt[j,:,:] = np.dot(gradub[j,:,:],Ps[j,:,:]) + np.dot(np.dot(gradlt[j,:,:],Finv[j,:,:]),Ps[j,:,:]) - (1./pdict['Wi'])*(Ps[j,:,:] - Finv[j,:,:].transpose())        
    return np.append(lt,Pt.flatten())

def calcState(r):
    StateNow = {}
    regridflag = False
    edgecrit = 0.1 #difference in trace
    detcrit = 0.1 #difference in det
    pdict = r.f_params[0]
    N = pdict['N']
    M = pdict['M']
    Q = len(r.y) - 2*N*M - 4*N*M
    StateNow['t'] = r.t
    if Q < 0: #stokes flow
        try:
            l3D = np.reshape(r.y[range(Q+4*N*M,Q+4*N*M+2*N*M)],(N,M,2))
            StateNow['l']=l3D
            gl = SD2D.vectorGrad(l3D,pdict['gridspc'],N,M)            
            adarray=np.zeros((N,M))
            for j in range(N):
                for k in range(M):
                    gls = gl[j,k,:,:]
                    gdet = np.linalg.det(gls)
                    adarray[j,k] = gdet
            StateNow['alldets']=adarray
            StateNow['fpts']=r.y[:Q+4*N*M]
        except:
            StateNow['fpts']=r.y
    elif Q >= 0: #in the viscoelastic case, record Lagrangian variables
        StateNow['fpts']=r.y[:Q]
        l3D = np.reshape(r.y[range(Q,Q+2*N*M)],(N,M,2))
        StateNow['l']=l3D
        gl = SD2D.vectorGrad(l3D,pdict['gridspc'],N,M)            
        Ptemp = np.reshape(r.y[(Q+2*N*M):],(N,M,2,2))
        Stemp = np.zeros((N,M,2,2))
        tr = np.zeros((N,M))
        for j in range(N):
            for k in range(M):
                gls = gl[j,k,:,:]
                gdet = np.linalg.det(gls)
                if np.abs(gdet-1.0) > detcrit:
                    print('det distorted')
                    regridflag = True
                Ps = Ptemp[j,k,:,:]
                Stemp[j,k,:,:]=np.dot(Ps,gls.transpose())
                tr[j,k] = Stemp[j,k,0,0] + Stemp[j,k,1,1]
        if np.any(np.abs(tr[:5,:]-2) > edgecrit) or np.any(np.abs(tr[N-5:,:]-2) > edgecrit) or np.any(np.abs(tr[:,:5]-2) > edgecrit) or np.any(np.abs(tr[:,M-5:]-2) > edgecrit): 
            print('stress close to edge')
            regridflag = True
        StateNow['S']=Stemp
        StateNow['Strace']=tr
    return StateNow, regridflag

def calcStateFixedRegrid(r,dt,t0,timecrit):
    StateNow = {}
    regridflag = False
#    addpts = False
    edgecrit = 0.1
    pdict = r.f_params[0]
    N = pdict['N']
    M = pdict['M']
    Q = len(r.y) - 2*N*M - 4*N*M
    StateNow['t'] = r.t
    if Q < 0: #stokes flow
        try:
            l3D = np.reshape(r.y[range(Q+4*N*M,Q+4*N*M+2*N*M)],(N,M,2))
            StateNow['l']=l3D
            gl = SD2D.vectorGrad(l3D,pdict['gridspc'],N,M)            
            adarray=np.zeros((N,M))
            for j in range(N):
                for k in range(M):
                    gls = gl[j,k,:,:]
                    gdet = np.linalg.det(gls)
                    adarray[j,k] = gdet
            StateNow['alldets']=adarray
            StateNow['fpts']=r.y[:Q+4*N*M]
        except:
            StateNow['fpts']=r.y
    elif Q >= 0: #in the viscoelastic case, record Lagrangian variables
        StateNow['fpts']=r.y[:Q]
        l3D = np.reshape(r.y[range(Q,Q+2*N*M)],(N,M,2))
        StateNow['l']=l3D
        gl = SD2D.vectorGrad(l3D,pdict['gridspc'],N,M)            
        Ptemp = np.reshape(r.y[(Q+2*N*M):],(N,M,2,2))
        Stemp = np.zeros((N,M,2,2))
        tr = np.zeros((N,M))
        for j in range(N):
            for k in range(M):
                gls = gl[j,k,:,:]
                gdet = np.linalg.det(gls)
                Ps = Ptemp[j,k,:,:]
                Stemp[j,k,:,:]=np.dot(Ps,gls.transpose())
                tr[j,k] = Stemp[j,k,0,0] + Stemp[j,k,1,1]
        if r.t > t0 and isIntDivisible((r.t-t0),timecrit): 
            regridflag = True
            addpts = True
            if np.any(np.abs(tr[:8,:]-2) > edgecrit) or np.any(np.abs(tr[N-8:,:]-2) > edgecrit) or np.any(np.abs(tr[:,:8]-2) > edgecrit) or np.any(np.abs(tr[:,M-8:]-2) > edgecrit): 
                pass
#                addpts = True
        if np.any(np.abs(tr[:4,:]-2) > edgecrit) or np.any(np.abs(tr[N-4:,:]-2) > edgecrit) or np.any(np.abs(tr[:,:4]-2) > edgecrit) or np.any(np.abs(tr[:,M-4:]-2) > edgecrit): 
            print('stress close to edge')
#            regridflag = True
#            addpts = True
        StateNow['S']=Stemp
        StateNow['Strace']=tr
#    return StateNow, regridflag, addpts
    return StateNow, regridflag

def logState(StateNow,StateSave):
    StateSave['t'].append(StateNow['t'])
    StateSave['fpts'].append(StateNow['fpts'])
    if StateNow.has_key('l'):
        StateSave['l'].append(StateNow['l'])
    if StateNow.has_key('S'):
        StateSave['S'].append(StateNow['S'])
        StateSave['Strace'].append(StateNow['Strace'])
    if StateNow.has_key('alldets'):
        StateSave['alldets'].append(StateNow['alldets'])
    return StateSave

def mySolver(myodefunc,y0,t0,dt,totalTime,pdict,stressflag=0,regridding=0,timecrit=1.0,scalefactor=2,addpts=1,alldetflag=0,rtol=1.e-3,method='bdf'):
    '''
    myodefunc is f in y' = f(y); y0, t0 are initial conditions; 
    dt is time step for saving data (integrator time step is chosen automatically); 
    totalTime is the stopping time for the integration; 
    pdict is a dictionary containing the parameters for the update function f;
    method is a string specifying which ode solver to use.
    
    '''
    #initialize integrator
    r = ode(myodefunc).set_integrator('vode',method=method,rtol=rtol).set_initial_value(y0,t0).set_f_params(pdict) 
    # initialize list of saved variables
    StateSave={}
    StateSave['t']=[]
    StateSave['fpts']=[]
    if myodefunc != stokesFlowUpdater:
        StateSave['l']=[]
    if stressflag:
        StateSave['S']=[]
        StateSave['Strace']=[]
    if alldetflag:
        StateSave['alldets']=[]
    StateNow, regridflag = calcStateFixedRegrid(r,dt,t0,timecrit)
#    StateNow, regridflag = calcState(r)  #use this function for dynamically chosen regridding
    StateSave = logState(StateNow,StateSave)
    # integrate in time until regridding is required
    try:
        numskip = pdict['numskip']
    except:
        numskip = 0
    print(t0) #let the user know the simulation has begun
    c=1
    while r.successful() and r.t < totalTime:  
        r.integrate(t0+c*dt)
#        StateNow, regridflag = calcState(r) #use this function for dynamically chosen regridding
        if (regridding and isIntDivisible((r.t-t0),timecrit)) or numskip ==0 or np.mod(c,numskip) == 0:
            print(r.t)
            StateNow, regridflag = calcStateFixedRegrid(r,dt,t0,timecrit)
            StateSave = logState(StateNow,StateSave)
        c+=1
        # regrid when needed, then reset integrator
        if regridding and (regridflag or addpts) and r.t < totalTime:
            print('time to regrid...')
            lnew,Pnew,Nnew,Mnew = mygrids.interp2NewGrid(StateNow['l'],StateNow['S'],pdict['gridspc'],1,scalefactor,addpts) 
#            plotforRegrid(StateNow['l'],StateNow['S'],lnew,Pnew,r.t)
            y0 = np.append(np.append(StateNow['fpts'],lnew.flatten()),Pnew.flatten())
            t0 = r.t
            pdict['N'] = Nnew
            pdict['M'] = Mnew
            print(StateNow['l'].shape)
            print(lnew.shape)
            r = ode(myodefunc).set_integrator('vode',method=method,rtol=rtol).set_initial_value(y0,t0).set_f_params(pdict) 
            regridflag = False
            addpts = False
            StateNow={}
            c=1
    return StateSave

def myCircleFlow():
        #dictionary of variables for both Stokes flow and viscoelastic flow
    for N in [54]:#[20,40,80,160]:
        pdict = dict( N = N, M = N, gridspc = 4.0/N, origin = [-2.0,-2.0], mu = 1.0, Wi = 1.2)
        pdict['beta'] = 1./(2*pdict['Wi'])
       
        #set time parameters, save data every numskip time steps (default is every time step)
        #note that I do not control the time step in the solver!! 
        # my time step only determines the maximum time step allowed
        t0 = 0; totalTime = 3.0; dt = 5.e-2; 
        #save file name
        fname = os.path.expanduser('~/VEsims/CircleFlow/circ_regrid_dynaddpts')
#        fname = os.path.expanduser('/scratch03/bcummins/mydata/ve/circ_regrid_dynaddpts')

        #first assign params
        gridspc = pdict['gridspc']
        pdict['eps'] = 2*gridspc
        pdict['U'] = 1.0
        pdict['myVelocity'] = circleFlow
        #make the grid
        M=pdict['M']
        l0 = mygrids.makeGridCenter(N,M,gridspc,pdict['origin'])
        P0 = np.zeros((N,M,2,2))
        P0[:,:,0,0] = 1.0
        P0[:,:,1,1] = 1.0
        y0 = np.append(l0.flatten(),P0.flatten())
    
        #Viscoelastic run
        print('Circle flow, N = %02d' % N)
        StateSave = mySolver(veExtensionUpdater,y0,t0,dt,totalTime,pdict)
        
        #save the output
        pdict['myVelocity'] = 'circleFlow'
        StateSave['pdict']=pdict
        StateSave['dt']=dt
#        print(StateSave.keys())
        F = open( fname+'_N%03d_Wi%02d_Time%02d.pickle' % (N,int(round(pdict['Wi'])),int(round(totalTime))), 'w' )
        Pickler(F).dump(StateSave)
        F.close()


def myDipole():
    #dictionary of variables for both Stokes flow and viscoelastic flow
    Nlist = [50]
    Wi = 1.2
    dipolef  = [1.e-2, 1.e-2]
    dipolex0 = [0.0, 0.0]
    T = 5.0
    dt = 0.05#2.5e-6
    numskip = 1
    basedir = os.path.expanduser('~/VEsims/DipoleFlow/')
    fnamestart = 'dipole_regrid'
    fname = basedir + fnamestart
    for N in Nlist:
        pdict = dict( N = N, M = N, gridspc = 1.0/N, origin = [-0.5,-0.5], mu = 1.0, Wi = Wi)
        pdict['beta'] = 1. / (2*pdict['Wi'])       
        pdict['dipolef'] = dipolef  
        pdict['dipolex0'] = dipolex0  
        pdict['myVelocity'] = regDipole     
        #set time parameters, save data every numskip time steps (default is every time step)
        #note that I do not control the time step in the solver!! 
        # my time step only determines the maximum time step allowed
        t0 = 0; totalTime = T; pdict['numskip']=numskip 
        #choose regularization parameter (should not matter when U=(x,-y))
        pdict['eps'] = 2*pdict['gridspc']
        #make the grid
        M=pdict['M']
        l0 = mygrids.makeGridCenter(N,M,pdict['gridspc'],pdict['origin'])
        P0 = np.zeros((N,M,2,2))
        P0[:,:,0,0] = 1.0
        P0[:,:,1,1] = 1.0
        print(P0.shape)
        y0 = np.append(l0.flatten(),P0.flatten())    
        #Viscoelastic run
        print('Dipole flow, N = %02d' % N)
        StateSave = mySolver(veExtensionUpdater,y0,t0,dt,totalTime,pdict)        
        #save the output
        pdict['myVelocity'] = 'regDipole'
        StateSave['pdict']=pdict
        StateSave['dt']=dt
        F = open( fname+'_N%03d_Wi%02d_Time%02d.pickle' % (N,int(round(pdict['Wi'])),int(round(totalTime))), 'w' )
        Pickler(F).dump(StateSave)
        F.close()

def myExtension():
    #dictionary of variables for both Stokes flow and viscoelastic flow
    for N in [40]:#[20,40,80,160]:
        pdict = dict( N = N, M = N, gridspc = 0.0375, origin = [-0.75,-0.75], mu = 1.0, Wi = 1.2)
        pdict['beta'] = 1./(2*pdict['Wi'])
       
        #set time parameters, save data every numskip time steps (default is every time step)
        #note that I do not control the time step in the solver!! 
        # my time step only determines the maximum time step allowed
        t0 = 0; totalTime = 30; dt = 5.e-2; 
        #save file name
        fname = os.path.expanduser('~/scratch/extension_noregrid')
#        fname = os.path.expanduser('/scratch03/bcummins/mydata/ve/fourmill')

        #first assign params
        Np = 20
        L = 0.78
        h = L/(Np-1)
        pdict['eps'] = 2*(0.75*h)
        pdict['U'] = 0.1
        pdict['myVelocity'] = FourRollMill
        #make the grid
        M=pdict['M']
        gridspc = pdict['gridspc']
        l0 = mygrids.makeGridCenter(N,M,gridspc,pdict['origin'])
        P0 = np.zeros((N,M,2,2))
        P0[:,:,0,0] = 1.0
        P0[:,:,1,1] = 1.0
        y0 = np.append(l0.flatten(),P0.flatten())
    
        #Viscoelastic run
        print('Four roll mill flow, N = %02d' % N)
        StateSave = mySolver(veExtensionUpdater,y0,t0,dt,totalTime,pdict,1,0)
        
        #save the output
        pdict['myVelocity'] = 'FourRollMill'
        StateSave['pdict']=pdict
        StateSave['dt']=dt
#        print(StateSave.keys())
        F = open( fname+'_N%03d_Wi%02d_Time%02d.pickle' % (N,int(round(pdict['Wi'])),int(round(totalTime))), 'w' )
        Pickler(F).dump(StateSave)
        F.close()

def mySwimmer():
    #dictionary of variables for both Stokes flow and viscoelastic flow
    Wilist = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]#[1.6]#[0.1,0.08,0.06,0.04,0.02]
    for N in [54]:#[20,40,80,160]:
        for Wi in Wilist:
            pdict = dict( N = N, M = int(np.ceil(0.9*N/1.3)), gridspc = 1.3/N, origin = [0.0,0.05], mu = 1.0, Wi = Wi)
            pdict['beta'] = 1./(2*pdict['Wi'])
           
            #set time parameters, save data every time step
            #note that I do not control the time step in the solver!! 
            # my time step only tells me where I will save data
            t0 = 0; totalTime = 10.0; dt = 5.e-2; 
            initTime=1.0 #need this to be a full swimmer cycle -- see curvature forces in code
            #save file name
            
            #make swimmer
            #first assign params
            a = 0.16#0.3#0.1
            w = -2*np.pi#2*np.pi  
            lam = 2.5*np.pi#8.4 #2*np.pi/0.75
            Np = 26
            L = 0.6
            h = L/(Np-1)
            pdict['K'] =40.0
            pdict['xr'] = np.array([h])
#            pdict['forcedict'] = {'a':a, 'w':w,'t':0,'Kcurv':0.01,'lam':lam,'Np':Np,'h':h,'L':L}
            pdict['forcedict'] = {'a':a, 'w':w,'t':0,'Kcurv':0.1,'lam':lam,'Np':Np,'h':h,'L':L}
            eps = 4.0*h
            pdict['eps'] = eps
            #Now setup initial conditions
            y0 = 0.5*np.ones((2*Np,))
            v = np.arange(0.3,0.3+L+h/2,h)#np.arange(0.5,0.5+L+h/2,h)
            y0[:-1:2] = v #initial conds [x0,y0,x1,y1,...,xn,yn]        
            pdict['myForces'] = calcForcesSwimmerTFS#calcForcesSwimmer #ramp up amplitude of swimmer forces
         
            #Initialize in Stokes flow for viscoelastic flow...
            print('Initialize swimmer in Stokes flow...')
            StateSave = mySolver(stokesFlowUpdater,y0,t0,dt,initTime,pdict,0,0)
            y0 = np.asarray(StateSave['fpts'])[-1,:]                 
             
            #run the ode solver
            if Wi ==Wilist[0]:
                print('Swimmer in Stokes flow...')
                StateSave = mySolver(stokesFlowUpdater,y0,initTime,dt,totalTime+initTime,pdict,0,0)
                #save the output
                StateSave['pdict']=pdict
                StateSave['dt']=dt
                pdict['myForces'] = 'calcForcesSwimmerTFS' #ramp up amplitude of swimmer forces
                try:
                    fname = os.path.expanduser('~/VEsims/Swimmer/TFS_FinalParams_SmallerDomain/')    
                    F = open( fname+'stokes_eps%03d_N%03d_Time%02d.pickle' % (int(round(eps*1000)),N,int(totalTime+initTime)), 'w' )
                except:
                    fname = '/scratch03/bcummins/mydata/ve/TFS_FinalParams_SmallerDomain/'    
                    F = open( fname+'stokes_eps%03d_N%03d_Time%02d.pickle' % (int(round(eps*1000)),N,int(totalTime+initTime)), 'w' )
                Pickler(F).dump(StateSave)
                F.close()
                pdict['myForces'] = calcForcesSwimmerTFS #ramp up amplitude of swimmer forces

        
    #        #run Stokes flow with markers
    #        #make the grid
    #        N=pdict['N']
    #        M=pdict['M']
    #        gridspc = pdict['gridspc']
    #        l0 = mygrids.makeGridCenter(N,M,gridspc,pdict['origin'])
    #        y0 = np.append(y0, l0.flatten())
    #    
    #        #Stokes flow
    #        print('Stokes flow with markers...')
    #        fpts, t, l, S, Strace, alldets = mySolver(stokesFlowUpdaterWithMarkers,y0,t0,dt,totalTime+t0,pdict1,'bdf',1,0,1)
    #        #save the output
    #        mat2py.write(fname+'stokesmarkers_%03d.mat' % N, {'t':np.asarray(t),'fpts':np.asarray(fpts),'l':np.asarray(l),'S':np.asarray(S),'Strace':np.asarray(Strace),'pdict':pdict,'dt':dt,'alldets':np.asarray(alldets)})

            #make the grid
            N=pdict['N']
            M=pdict['M']
            gridspc = pdict['gridspc']
            l0 = mygrids.makeGridCenter(N,M,gridspc,pdict['origin'])
            P0 = np.zeros((N,M,2,2))
            P0[:,:,0,0] = 1.0
            P0[:,:,1,1] = 1.0
            y0 = np.append(np.append(y0, l0.flatten()),P0.flatten())
        
            #Viscoelastic run
            print('Swimmer in VE flow, N = %02d, Wi = %f' % (N,Wi))
            stressflag=1
            regridding=1
            timecrit=0.4
            scalefactor=2
            addpts = 0
            StateSave = mySolver(viscoElasticUpdaterKernelDeriv,y0,initTime,dt,totalTime+initTime,pdict,stressflag,regridding,timecrit,scalefactor,addpts)
            #save the output
            StateSave['pdict']=pdict
            StateSave['dt']=dt
            pdict['myForces'] = 'calcForcesSwimmerTFS' #ramp up amplitude of swimmer forces
            F = open( fname+'visco_fixedregrid004_addpts0_eps%03d_N%03d_Wi%04d_Time%02d.pickle' % (int(round(eps*1000)),N,int(round(pdict['Wi']*100)),int(totalTime+initTime)), 'w' )
            Pickler(F).dump(StateSave)
            F.close()

def mySpot():
    #get ready to save files
    import mat2py
    import os
    #dictionary of variables for both Stokes flow and viscoelastic flow
    for N in [20,40,80,160]:
        pdict = dict( N = N, M = (16*N)/40, gridspc = 1.5/N, origin = [0.0,0.0], mu = 1.0, Wi = 0.2)
        pdict['beta'] = 1./(2*pdict['Wi'])
       
        #set time parameters, save data every time step
        #note that I do not control the time step in the solver!! 
        # my time step only tells me where I will save data
        t0 = 0; totalTime = 1.0; dt = 5.e-2; 
        #save file name
        fname = os.path.expanduser('~/scratch/spotC')
        
        #make swimmer
        #first assign params
        Np = 20
        L = 0.78
        h = L/(Np-1)
        pdict['K'] =10.0
        pdict['xr'] = np.array([[1.173,1.173]])
        pdict['eps'] = 2*(0.75*h)
        #Now setup initial conditions
        y0 = np.array([0.8-h/2.,0.3+h/4.])
        
        #add unsaveable entries
        pdict1 = pdict.copy()
        pdict1['myForces'] = calcForcesConstant 
    #    #run the ode solver
    #    print('Stokes flow...')
    #    fpts, t, l, S, Strace = mySolver(stokesFlowUpdater,y0,t0,dt,totalTime,pdict1)
    #    mat2py.write(fname+'_stokesonly.mat', {'t':t,'fpts':fpts,'pdict':pdict,'dt':dt})
        
         
        initTime=0.0 #remove this when initialization in Stokes flow is desired
         
         
        #make the grid
        N=pdict['N']
        M=pdict['M']
        gridspc = pdict['gridspc']
        l0 = mygrids.makeGridCenter(N,M,gridspc,pdict['origin'])
        P0 = np.zeros((N,M,2,2))
    #    P0[:,:,0,0] = 1.0
    #    P0[:,:,1,1] = 1.0
        y0 = np.append(np.append(y0, l0.flatten()),P0.flatten())
    
        #Viscoelastic run
        print('VE flow...')
        fpts, t, l, S, Strace, alldets = mySolver(viscoElasticUpdaterKernelDeriv,y0,initTime,dt,totalTime+initTime,pdict1)
        #save the output
        mat2py.write(fname+'_visco_nove%03d_centergrid.mat' % N, {'t':np.asarray(t),'fpts':np.asarray(fpts),'l':np.asarray(l),'S':np.asarray(S),'Strace':np.asarray(Strace),'pdict':pdict,'dt':dt,'alldets':np.asarray(alldets)})


def mySprings():
    #get ready to save files
    import mat2py
    #dictionary of variables for both Stokes flow and viscoelastic flow
    pdictsave = dict( N = 20, M = 20, gridspc = 1./20, origin = [0.0,0.0], mu = 1.0, K =5.0, xr = np.array([0.1]))
    pdictsave['eps'] = 4*pdictsave['gridspc']
    pdict = pdictsave.copy()
    pdict['myForces'] = calcForcesTwoHeadSpring    
    #set time parameters, save data every numskip time steps
    t0 = 0; totalTime = 4; dt = 1.e-3; numskip = 50;     
    #save file name
    fname = os.path.expanduser('~/scratch/springtestC')        
    #Stokes flow test, works
    print('Stokes flow...')
    #set up initial conditions
    y0 = np.array([0.3,0.5,0.7,0.5])
    #run the ode solver
    fpts, t, l, S, Strace, alldets = mySolver(stokesFlowUpdater,y0,t0,dt,totalTime,pdict,'bdf',numskip)
    #save the output
    mat2py.write(fname+'_stokes.mat', {'t':t,'fpts':fpts,'pdict':pdictsave,'dt':dt})    
    #use same initial conditions for spring as in Stokes flow, but add on Lagrangian points and stress
    N=pdict['N']
    M=pdict['M']
    gridspc = pdict['gridspc']
    l0 = mygrids.makeGridCenter(N,M,gridspc,pdict['origin'])
    P0 = np.zeros((N,M,2,2))
    P0[:,:,0,0] = 1
    P0[:,:,1,1] = 1
    y0 = np.append(np.append(y0, l0.flatten()),P0.flatten())
    for Wi in [0.2,0.4,0.8,1.6,3.2,6.4,12.8]:
        print('Wi = %f' % Wi)
        pdictsave['Wi'] = Wi
        pdictsave['beta'] = 1./(2*Wi)
        pdict['Wi'] = Wi
        pdict['beta'] = 1./(2*Wi)
        totalTime = 8*np.ceil(Wi);
        
        #Viscoelastic run
        print('VE flow...')
        #run the ode solver
        fpts, t, l, S, Strace, alldets = mySolver(viscoElasticUpdaterKernelDeriv,y0,t0,dt,totalTime,pdict,'bdf',numskip)
        #save the output
        mat2py.write(fname+'_visco%03d.mat' % int(Wi*10), {'t':np.asarray(t),'fpts':np.asarray(fpts),'l':np.asarray(l),'S':np.asarray(S),'Strace':np.asarray(Strace),'pdict':pdictsave,'dt':dt})

def fallingStar():
    ########################################################
    #falling star for art show
    ########################################################
    #get ready to save files
    import mat2py
    #dictionary of variables 
    pdictsave = dict( N = 80, M = 80, gridspc = 1./100, origin = [0,0], mu = 1.0, K =0.05, xr = [], Wi = 1.0)
    pdictsave['beta'] = 1./(2*pdictsave['Wi'])
    pdictsave['eps'] = 4*pdictsave['gridspc']
    pdict = pdictsave.copy()
    pdict['myForces'] = calcForcesGravity
    N = pdict['N']
    M = pdict['M']
    gridspc = pdict['gridspc']
    
    #set time parameters
    t0 = 0; totalTime = 0.45; dt = 1.e-2; numskip = 1;
    
    #save file name
    fname = 'scratch/star_tree2C'
    
    
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
    l0 = mygrids.makeGridCenter(N,M,gridspc,pdict['origin'])
    P0 = np.zeros((N,M,2,2))
    P0[:,:,0,0] = 1
    P0[:,:,1,1] = 1
    y0 = np.append(np.append(np.append(fpts.flatten(), mpts.flatten()), l0.flatten()),P0.flatten())
    #run the ode solver
    fpts, t, l, S, Strace = mySolver(viscoElasticUpdaterKernelDeriv,y0,t0,dt,totalTime,'bdf',pdict,numskip)
    #save the output
    mat2py.write(fname+'_visco.mat', {'t':np.asarray(t),'fpts':np.asarray(fpts),'l':np.asarray(l),'S':np.asarray(S),'Strace':np.asarray(Strace),'pdict':pdictsave,'dt':dt})

def plotforRegrid(lold,Sold,lnew,Snew,time):
    xmin = np.min(lnew[:,:,0]); xmax = np.max(lnew[:,:,0])
    ymin = np.min(lnew[:,:,1]); ymax = np.max(lnew[:,:,1])
    xdif = xmax-xmin; ydif = ymax-ymin
    limits = [xmin-0.1*xdif,xmax+0.1*xdif,ymin-0.1*ydif,ymax+0.1*ydif]
    S11min = np.min([np.min(Sold[:,:,0,0]),np.min(Snew[:,:,0,0])])
    S11max = np.max([np.max(Sold[:,:,0,0]),np.max(Snew[:,:,0,0])])
    S22min = np.min([np.min(Sold[:,:,1,1]),np.min(Snew[:,:,1,1])])
    S22max = np.max([np.max(Sold[:,:,1,1]),np.max(Snew[:,:,1,1])])
    vlims = [0.9*S11min,1.1*S11max,0.9*S22min,1.1*S22max]
    lvls = [np.linspace(np.min(Sold[:,:,0,0]),np.max(Sold[:,:,0,0]),10),np.linspace(np.min(Sold[:,:,1,1]),np.max(Sold[:,:,1,1]),10)]
    sv.contourRegrid(lold,Sold,limits,time,1,lvls,vlims)
    sv.contourRegrid(lnew,Snew,limits,time,0,lvls,vlims)
    
if __name__ == '__main__':
#    myCircleFlow()
#    myDipole()
#    myExtension()
    mySwimmer()
#    mySprings()    
#    mySpot()
    
    
    
    
    
    
    
