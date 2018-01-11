#import python modules
import matplotlib.pyplot as plt
import numpy as np
import cPickle, os, re
#import home-rolled python modules 
import SpatialDerivs2D as SD2D
import Gridding as mygrids
import mainC
from utilities import loadPickle
try:
    import pythoncode.cext.CubicStokeslet2D as CM
except:
    print('Please compile the C extension CubicStokeslet2D.c and put the .so in the cext folder.')
    raise(SystemExit)

#FIXME remove all references to regrdding. Not appropriate for this problem.

def findRegridTimes(l):
    regridinds=[]
    for k in range(1,len(l)):
        if l[k].shape != l[k-1].shape:
            regridinds.append(k)
    return regridinds
 

def knownsolution(Wi, l, times, regridinds):
    '''Calculate exact parabolic shear flow.'''
    lcalc = []
    Pcalc = []
    Scalc = []
    for k in range(len(times)):
        t = times[k]
        if regridinds == [] or k < regridinds[0]:
            initgrid = l[0]
            tau = 0
        else:
            # shift the solution to account for the regridding
            klist = [j for j in regridinds if j<=k]
            initgrid = l[klist[-1]]
            tau = times[klist[-1]]
        indsg = np.nonzero(initgrid[:,:,1]>=0) 
        indsl = np.nonzero(initgrid[:,:,1]<0)
        # position solution 
        pos = initgrid.copy()
        pos0 = pos[:,:,0]
        init1 = initgrid[:,:,1]
        pos0[indsg] += init1[indsg]*(1-init1[indsg])*(t-tau) # because pos0 is a slice of pos, these values will be written into pos
        pos0[indsl] += init1[indsl]*(1+init1[indsl])*(t-tau) # I do this because of the output inds for nonzero. I can't specify pos[inds,0].
        lcalc.append(pos.copy())
        # make matrices of the correct size
        stressS = np.zeros((initgrid.shape[0],initgrid.shape[1],2,2))
        stressP = np.zeros((initgrid.shape[0],initgrid.shape[1],2,2))
        # Eulerian stress solution
        stressS[:,:,0,0] = (1-2*initgrid[:,:,1])**2 * Wi * ( Wi - (Wi + t)*np.exp(-t/Wi) ) + 1 #FIXME -- add indsg and indsl
        stressS[:,:,0,1] = (1-2*initgrid[:,:,1])*Wi*(1-np.exp(-t/Wi))
        stressS[:,:,1,0] = stressS[:,:,0,1]
        stressS[:,:,1,1] = 1
        Scalc.append(stressS.copy())
        # Lagrangian stress solution
        stressP[:,:,0,0] = stressS[:,:,0,0] - (1-2*initgrid[:,:,1])*(t-tau)*stressS[:,:,0,1]
        stressP[:,:,0,1] = stressS[:,:,0,1]
        stressP[:,:,1,0] = stressS[:,:,1,0] - (1-2*initgrid[:,:,1])*(t-tau)
        stressP[:,:,1,1] = 1
        Pcalc.append(stressP.copy())
    return lcalc, Pcalc, Scalc

def simresults(basename, basedir):
    '''Retrieve approximate solution from saved output'''
    mydict = loadPickle(basename, basedir)
    l = mydict['l']
    regridinds = findRegridTimes(l)
    S=mydict['S']
    P=[]
    for k in range(len(mydict['t'])):
        # N and M can change because of regridding
        N = l[k].shape[0]
        M = l[k].shape[1]
        Ft = SD2D.vectorGrad(l[k],mydict['pdict']['gridspc'],N,M)
        Ftemp = np.reshape(Ft,(N*M,2,2))
        Ftinv = CM.matinv2x2(Ftemp)
        Ftinv = np.reshape(Ftinv,(N,M,2,2))
        stressP = np.zeros((N,M,2,2))
        for j in range(N):
            for m in range(M):
                stressP[j,m,:,:] = S[k][j,m,:,:]*Ftinv[j,m,:,:].transpose()
        P.append(stressP.copy())
    return l, P, S, mydict, regridinds

def calcerrs(l, P, S, lcalc, Pcalc, Scalc, mydict):
    '''Calculate Linf and L2 errors between the exact solution and the numerical approximation'''
    errorsLinf = [[] for k in range(10)]
    errorsLtwo = [[] for k in range(10)]
    for k in range(len(mydict['t'])):
        errorsLinf[0].append( np.max(np.abs(l[k][:,:,0] - lcalc[k][:,:,0])) / (np.max(np.abs(lcalc[k][:,:,0]))) )
        errorsLinf[1].append( np.max(np.abs(l[k][:,:,1] - lcalc[k][:,:,1])) / (np.max(np.abs(lcalc[k][:,:,1]))) )
        errorsLtwo[0].append( np.sqrt( np.sum( (l[k][:,:,0] - lcalc[k][:,:,0])**2 ) ) / np.sqrt( np.sum( (lcalc[k][:,:,0])**2 ) ) )
        errorsLtwo[1].append( np.sqrt( np.sum( (l[k][:,:,1] - lcalc[k][:,:,1])**2 ) ) / np.sqrt( np.sum( (lcalc[k][:,:,1])**2 ) ) )
        errorsLinf[2].append( np.max(np.abs(P[k][:,:,0,0] - Pcalc[k][:,:,0,0])) / np.max(np.abs(Pcalc[k][:,:,0,0])) )
        errorsLinf[3].append( np.max(np.abs(P[k][:,:,0,1] - Pcalc[k][:,:,0,1]))) #approximately zero
        errorsLinf[4].append( np.max(np.abs(P[k][:,:,1,0] - Pcalc[k][:,:,1,0]))) / np.max(np.abs(Pcalc[k][:,:,1,0])) )
        errorsLinf[5].append( np.max(np.abs(P[k][:,:,1,1] - Pcalc[k][:,:,1,1])) / np.max(np.abs(Pcalc[k][:,:,1,1])) )
        errorsLtwo[2].append( np.sqrt( np.sum( (P[k][:,:,0,0] - Pcalc[k][:,:,0,0])**2 ) ) / np.sqrt( np.sum( (Pcalc[k][:,:,0,0])**2 ) ) )
        errorsLtwo[3].append( np.sqrt( np.sum( (P[k][:,:,0,1] - Pcalc[k][:,:,0,1])**2 ) )*mydict['pdict']['gridspc'] )
        errorsLtwo[4].append( np.sqrt( np.sum( (P[k][:,:,1,0] - Pcalc[k][:,:,1,0])**2 ) ) / np.sqrt( np.sum( (Pcalc[k][:,:,1,0])**2 ) ) )
        errorsLtwo[5].append( np.sqrt( np.sum( (P[k][:,:,1,1] - Pcalc[k][:,:,1,1])**2 ) ) / np.sqrt( np.sum( (Pcalc[k][:,:,1,1])**2 ) ) )
        errorsLinf[6].append( np.max(np.abs(S[k][:,:,0,0] - Scalc[k][:,:,0,0])) / np.max(np.abs(Scalc[k][:,:,0,0])) )
        errorsLinf[7].append( np.max(np.abs(S[k][:,:,0,1] - Scalc[k][:,:,0,1])) ) #approximately zero
        errorsLinf[8].append( np.max(np.abs(S[k][:,:,1,0] - Scalc[k][:,:,1,0])) ) / np.max(np.abs(Scalc[k][:,:,1,0])) )
        errorsLinf[9].append( np.max(np.abs(S[k][:,:,1,1] - Scalc[k][:,:,1,1])) / np.max(np.abs(Scalc[k][:,:,1,1])) )
        errorsLtwo[6].append( np.sqrt( np.sum( (S[k][:,:,0,0] - Scalc[k][:,:,0,0])**2 ) ) / np.sqrt( np.sum( (Scalc[k][:,:,0,0])**2 ) ) )
        errorsLtwo[7].append( np.sqrt( np.sum( (S[k][:,:,0,1] - Scalc[k][:,:,0,1])**2 ) ) ) #The Eulerian domain is changing in time, so the grid spacing is too. Need to figure out what it is.
        errorsLtwo[8].append( np.sqrt( np.sum( (S[k][:,:,1,0] - Scalc[k][:,:,1,0])**2 ) ) ) / np.sqrt( np.sum( (Scalc[k][:,:,1,0])**2 ) ) )
        errorsLtwo[9].append( np.sqrt( np.sum( (S[k][:,:,1,1] - Scalc[k][:,:,1,1])**2 ) ) / np.sqrt( np.sum( (Scalc[k][:,:,1,1])**2 ) ) )
    return errorsLinf,errorsLtwo

def ploterrsguts(errorsLinf, mydict, basename, basedir, fname, titlestr):
    fname=basedir + basename + '/' + fname
    plt.close()
    plt.figure()
    plt.plot(mydict['t'], errorsLinf[0],'k',linewidth=2.0)     
    plt.plot(mydict['t'], errorsLinf[1],'r',linewidth=2.0)  
    plt.title(titlestr + ' in position')
    plt.legend(('X','Y'), loc=2)
    plt.xlabel('time')
    plt.savefig(fname+'X')
    plt.figure()
    plt.plot(mydict['t'], errorsLinf[2],'k',linewidth=2.0)     
    plt.plot(mydict['t'], errorsLinf[3],'r',linewidth=2.0)  
    plt.plot(mydict['t'], errorsLinf[4],'b',linewidth=2.0)     
    plt.plot(mydict['t'], errorsLinf[5],'g',linewidth=2.0)  
    plt.title(titlestr + ' in P')
    plt.legend(('$P_{11}$','$P_{12}$','$P_{21}$','$P_{22}$'), loc=2)
#    plt.legend(('$P_{11}$','$P_{22}$'), loc=2)
    plt.xlabel('time')
    plt.savefig(fname+'P')
    plt.figure()
    plt.plot(mydict['t'], errorsLinf[6],'k',linewidth=2.0)     
    plt.plot(mydict['t'], errorsLinf[7],'r',linewidth=2.0)  
    plt.plot(mydict['t'], errorsLinf[8],'b',linewidth=2.0)     
    plt.plot(mydict['t'], errorsLinf[9],'g',linewidth=2.0)  
    plt.title(titlestr + ' in S')
    plt.legend(('$S_{11}$','$S_{12}$','$S_{21}$','$S_{22}$'), loc=2)
#    plt.legend(('$S_{11}$','$S_{22}$'), loc=2)
    plt.xlabel('time')
    plt.savefig(fname+'S')
    plt.close()

def myParabolicShear(Nlist,Wi,T,dt,eps,numskip,basedir,fnamestart,regridding,timecrit):
    #dictionary of variables for both Stokes flow and viscoelastic flow
    for N in Nlist:
        pdict = dict( N = N, M = N, gridspc = 0.5/N, origin = [0.75,0.75], mu = 1.0, Wi = Wi)
        pdict['beta'] = 0 #1. / (2*pdict['Wi'])       
        pdict['U'] = 1.0 #don't change U. U=1 fixed in soln.
        pdict['myVelocity'] = mainC.ParabolicShear     
        #set time parameters, save data every numskip time steps (default is every time step)
        #note that I do not control the time step in the solver!! 
        # my time step only determines the maximum time step allowed
        t0 = 0; totalTime = T; pdict['numskip']=numskip 
        #choose regularization parameter (should not matter when U=(x,-y))
        pdict['eps'] = eps
        #make the grid
        M=pdict['M']
        gridspc = pdict['gridspc']
        l0 = mygrids.makeGridCenter(N,M,gridspc,pdict['origin'])
        P0 = np.zeros((N,M,2,2))
        P0[:,:,0,0] = 1.0
        P0[:,:,1,1] = 1.0
        print(P0.shape)
        y0 = np.append(l0.flatten(),P0.flatten())    
        #Viscoelastic run
        print('Parabolic shear, N = %02d' % N)
        StateSave = mainC.mySolver(mainC.veExtensionUpdater,y0,t0,dt,totalTime,pdict,1,regridding,timecrit)        
        #save the output
        StateSave['pdict']=pdict
        StateSave['dt']=dt
        fname = basedir + fnamestart
        F = open( fname+'_N%03d_Wi%02d_Time%02d.pickle' % (N,int(round(pdict['Wi'])),int(round(totalTime))), 'w' )
        cPickle.Pickler(F).dump(StateSave)
        F.close()
        
def compareh(Nlist,Wi,T,basedir,fnamestart):
    errorsLinf=[]
#    errorsLtwo=[]
    for N in Nlist:
        basename= fnamestart + '_N%03d_Wi%02d_Time%02d' % (N,int(round(Wi)),int(round(T)))
        l, P, S, mydict, regridinds = simresults(basename, basedir)
        print('regrid indices = ', regridinds)
        stressComponents(P, S, mydict, basename, basedir)
        lcalc, Pcalc, Scalc = knownsolution(mydict['pdict']['Wi'], l, mydict['t'], regridinds)
        eLinf, eLtwo = calcerrs(l, P, S, lcalc, Pcalc, Scalc, mydict)
        errorsLinf.append(eLinf)
#        errorsLtwo.append(eLtwo)
        fname = 'relerrors_Linf'
        titlestr = '$L_\infty$ error'
        ploterrsguts(eLinf, mydict, basename, basedir, fname, titlestr)
    errsinf=np.asarray(errorsLinf)
    if len(Nlist) == 3:  
#        pass      
#        errstwo=np.asarray(errorsLtwo)
        diffinf=[np.abs(errsinf[0,:,:]-errsinf[1,:,:]),np.abs(errsinf[1,:,:]-errsinf[2,:,:]),np.abs(errsinf[0,:,:]-errsinf[2,:,:])]
    #    difftwo=[errstwo[0,:,:]-errstwo[1,:,:],errstwo[1,:,:]-errstwo[2,:,:]]
        basename = fnamestart + '_N%03d_Wi%02d_Time%02d' % (Nlist[1],int(round(Wi)),int(round(T)))
        fname = 'errordiff2040_Linf'
        titlestr = 'Difference in (N=%02d to %02d) $L_\infty$ error' %(Nlist[0],Nlist[1])
        ploterrsguts(diffinf[0], mydict, basename, basedir, fname, titlestr)
        basename = fnamestart + '_N%03d_Wi%02d_Time%02d' % (Nlist[2],int(round(Wi)),int(round(T)))
        fname = 'errordiff4080_Linf'
        titlestr = 'Difference in (N=%02d to %02d) $L_\infty$ error' %(Nlist[1],Nlist[2])
        ploterrsguts(diffinf[1], mydict, basename, basedir, fname, titlestr)
        basename = fnamestart + '_N%03d_Wi%02d_Time%02d' % (Nlist[2],int(round(Wi)),int(round(T)))
        fname = 'errordiff2080_Linf'
        titlestr = 'Difference in (N=%02d to %02d) $L_\infty$ error' %(Nlist[0],Nlist[2])
        ploterrsguts(diffinf[2], mydict, basename, basedir, fname, titlestr)
        return errsinf, np.asarray(diffinf)
    else:
        print('Nlist is not length 3')
        return errsinf, None

def stressComponents(P, S, mydict, basename,basedir,fname='NumSoln'):
    plt.close()
    S11 = np.zeros((len(mydict['t']),))
    S12 = np.zeros((len(mydict['t']),))
    S22 = np.zeros((len(mydict['t']),))
    P11 = np.zeros((len(mydict['t']),))
    P12 = np.zeros((len(mydict['t']),))
    P22 = np.zeros((len(mydict['t']),))
    for k in range(0,len(mydict['t'])):
        S11[k]=np.max(S[k][:,:,0,0])
        S12[k]=np.max(S[k][:,:,0,1])
        S22[k]=np.max(S[k][:,:,1,1])
        P11[k]=np.max(P[k][:,:,0,0])
        P12[k]=np.max(P[k][:,:,0,1])
        P22[k]=np.max(P[k][:,:,1,1])
#        ml = S[k].shape[0]/2;  ml2 = S[k].shape[1]/2 
#        S11[k]=S[k][ml,ml2,0,0]
#        S12[k]=S[k][ml,ml2,0,1]
#        S22[k]=S[k][ml,ml2,1,1]
#        P11[k]=P[k][ml,ml2,0,0]
#        P12[k]=P[k][ml,ml2,0,1]
#        P22[k]=P[k][ml,ml2,1,1]
    plt.figure()
    plt.plot(mydict['t'],S11,'k')
    plt.plot(mydict['t'],S22,'b')
    plt.plot(mydict['t'],S12,'r')
    plt.legend( ('S11', 'S22', 'S12, S21'), loc=6 )
    plt.title('Stress components over time')
    fnameS = basedir + basename + '/' + fname + '_S'
    plt.savefig(fnameS)
    plt.figure()
    plt.plot(mydict['t'],P11,'k')
    plt.plot(mydict['t'],P22,'b')
    plt.plot(mydict['t'],P12,'r')
    plt.legend( ('P11', 'P22',  'P12, P21'), loc=2 )
    plt.title('Stress components over time')
    fnameP = basedir + basename + '/' + fname + '_P'
    plt.savefig(fnameP)
    plt.close()
    
def diffEps():
    # changing epsilon
    Nlist = [40]#[20,40,80]
    Wi = 1.2
    U = 0.1
    T = 30
    dt = 1.e-5#2.5e-6
    numskip = int(1./(10*dt))
    basedir = os.path.expanduser('~/VEsims/ExactExtensionalFlow/')
    Np = 20
    L = 0.78
    h = L/(Np-1)
    eps = (0.75*h)
    fnamestart = 'extension_noregrid_halfeps'
# #    basedir = '/scratch03/bcummins/mydata/ve/'
#    myExtension(Nlist,Wi,U,T,dt,eps,numskip,basedir,fnamestart,0)
    errsinf, diffinf=compareh(Nlist,Wi,T,basedir,fnamestart)
    try:
        print(np.max(errsinf[:,:10,:errsinf.shape[2]/2]), np.max(errsinf[:,:10,:]),np.max(diffinf[:,:10,:diffinf.shape[1]/2]), np.max(diffinf[:,:10,:]))
    except:
        print(np.max(errsinf[:,:10,:errsinf.shape[2]/2]), np.max(errsinf[:,:10,:]))
    errsinf_half = errsinf.copy()

    eps = 4*(0.75*h)
    fnamestart = 'extension_noregrid_twiceeps'
# #    basedir = '/scratch03/bcummins/mydata/ve/'
#    myExtension(Nlist,Wi,U,T,dt,eps,numskip,basedir,fnamestart,0)
    errsinf, diffinf=compareh(Nlist,Wi,T,basedir,fnamestart)
    try:
        print(np.max(errsinf[:,:10,:errsinf.shape[2]/2]), np.max(errsinf[:,:10,:]),np.max(diffinf[:,:10,:diffinf.shape[1]/2]), np.max(diffinf[:,:10,:]))
    except:
        print(np.max(errsinf[:,:10,:errsinf.shape[2]/2]), np.max(errsinf[:,:10,:]))
    errsinf_twice = errsinf.copy()
    
    fnamestart = 'extension_noregrid'
    errsinf, diffinf=compareh(Nlist,Wi,T,basedir,fnamestart)
    
    diffinf = [np.abs(errsinf[0,:,:]-errsinf_half[0,:,:]),np.abs(errsinf_twice[0,:,:]-errsinf[0,:,:]),np.abs(errsinf_twice[0,:,:]-errsinf_half[0,:,:])]
    diffinf = np.asarray(diffinf)
    print(np.max(errsinf[:,:10,:errsinf.shape[2]/2]), np.max(errsinf[:,:10,:]),np.max(diffinf[:,:10,:diffinf.shape[1]/2]), np.max(diffinf[:,:10,:]))
    
def diffN():
    Nlist = [20,40,80]
    Wi = 1.2
    U = 0.1
    T = 30
    dt = 1.e-5#2.5e-6
    numskip = int(1./(10*dt))
    basedir = os.path.expanduser('~/VEsims/ExactExtensionalFlow/')
    Np = 20
    L = 0.78
    h = L/(Np-1)
    eps = 2*(0.75*h)
    fnamestart = 'extension_noregrid_lowerrtol'
 #    basedir = '/scratch03/bcummins/mydata/ve/'
    myExtension(Nlist,Wi,U,T,dt,eps,numskip,basedir,fnamestart,0,0)
    errsinf, diffinf=compareh(Nlist,Wi,T,basedir,fnamestart)
    try:
        print(np.max(errsinf[:,:10,:errsinf.shape[2]/2]), np.max(errsinf[:,:10,:]), np.max(diffinf[:,:10,:diffinf.shape[1]/2]), np.max(diffinf[:,:10,:]))
    except:
        print( np.max(errsinf[:,:10,:errsinf.shape[2]/2]), np.max(errsinf[:,:10,:]))
        
def plotknown():
    Nlist = [20]
    Wi = 1.2
    U = 0.1
    T = 6.0
    timecrit = 5.0
    fnamestart = 'extension_regrid%02d' % int(round(timecrit))
    basedir = os.path.expanduser('~/VEsims/ExactExtensionalFlow/')
    basename= fnamestart + '_N%03d_Wi%02d_Time%02d' % (Nlist[0],int(round(Wi)),int(round(T)))
    l, P, S, F, Finv, mydict, regridinds = simresults(basename, basedir)
    print('regrid indices = ', regridinds)
    lcalc, Pcalc, Scalc, Fcalc, Finvcalc = knownsolution(U, Wi, l, mydict['t'], regridinds)
    stressComponents(Pcalc, Scalc, mydict, basename, basedir,'ExactSoln')
    return Scalc


if __name__ == '__main__':
    diffN()
#    plotknown()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        