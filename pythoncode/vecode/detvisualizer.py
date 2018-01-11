import matplotlib.pyplot as plt
import SpatialDerivs2D as SD2D
import numpy as np
import mat2py
import os
if not os.path.exists(os.path.expanduser('~/polymercode/trunk/pythoncode/cext/CubicStokeslet2D.so')):
    os.chdir(os.path.expanduser('~/polymercode/trunk/pythoncode/cext'))
    from cext.tests import buildnload
    cm = buildnload('CubicStokeslet2D')
    os.chdir(os.path.expanduser('~/polymercode/trunk/pythoncode/'))
else:
    os.chdir(os.path.expanduser('~/polymercode/trunk/pythoncode/'))
    import cext.CubicStokeslet2D as cm



def plotcomponents(fend='_nodegrid',fstart='swimmerC',tbegin=3):
    if fend == '' or fend == '_centergrid':
        fname1 = 'center_'+fstart+'_compratL2_'
        fname2 = 'center_'+fstart+'_compL2_'
    elif fend == '_nodegrid':
        fname1 = 'node_'+fstart+'_compratL2_'
        fname2 = 'node_'+fstart+'_compL2_'        
    elif fend == '_centergrid_lowerfish':
        fname1 = 'lower_'+fstart+'_compratL2_'
        fname2 = 'lower_'+fstart+'_compL2_'        

    for comp in [(0,0), (0,1), (1,0), (1,1)]:
        comps=[]
        h=[]
        N=[]
        M=[]
        for k,j in enumerate(2**np.arange(4)):
            mydict = mat2py.read(os.path.expanduser('~/scratch/'+fstart+'_visco_nove%03d' % (j*20) + fend + '.mat'))
            h.append(mydict['pdict']['gridspc'][0,0])
            N.append(mydict['pdict']['N'][0,0])
            M.append(mydict['pdict']['M'][0,0])
            print((h[k],N[k],M[k]))
            times = mydict['t'].flatten()
            ctemp = []
            for t in range(len(times)):
                gl = SD2D.vectorGrad(mydict['l'][t,:,:,:],h[k],N[k],M[k]) 
                x = ( (gl[:,:,comp[0],comp[1]])**2 ).sum()
                l2measure = h[k] * np.sqrt(x)
                ctemp.append(l2measure) 
            comps.append(ctemp)
        comps = np.asarray(comps).transpose().squeeze()
        Rats = np.zeros((comps.shape[0]-tbegin,2))
        plt.clf()
        for i in [1]:
            Rats[:,i] = ( comps[tbegin:,i] - comps[tbegin:,i+1] ) / ( comps[tbegin:,i+1] - comps[tbegin:,i+2] )
            plt.plot(mydict['t'].flatten()[tbegin:],Rats[:,i])
        print(np.max(Rats))
        plt.title('Ratios for component %d%d ($L_2$ measure)' % comp)
        plt.xlabel('Time')
        plt.savefig(os.path.expanduser('~/scratch/' + fname1 + '%d%d' % comp ))
        plt.clf()
        for i, k in enumerate(['$N=20$','$N=40$','$N=80$','$N=160$']):
            plt.plot(mydict['t'].flatten(),comps[:,i],label=k)
        plt.title('Component %d%d ($L_2$ measure)' % comp)
        plt.xlabel('Time')
        plt.legend(loc=0)
        plt.savefig(os.path.expanduser('~/scratch/' + fname2 + '%d%d' % comp ))
    print('done')


def plotdets(fend='_nodegrid',fstart='swimmerC',tbegin=3):
    if fend == '' or fend == '_centergrid':
        fname1 = 'center_'+fstart+'_detratL2'
        fname2 = 'center_'+fstart+'_detL2'
        fname3 = 'center_'+fstart+'_mindet'
        fname4 = 'center_'+fstart+'_maxdet'
    elif fend == '_nodegrid':
        fname1 = 'node_'+fstart+'_detratL2'
        fname2 = 'node_'+fstart+'_detL2'        
        fname3 = 'node_'+fstart+'_mindet'
        fname4 = 'node_'+fstart+'_maxdet'
    elif fend == '_centergrid_lowerfish':
        fname1 = 'lower_'+fstart+'_detratL2'
        fname2 = 'lower_'+fstart+'_detL2'        
        fname3 = 'lower_'+fstart+'_mindet'
        fname4 = 'lower_'+fstart+'_maxdet'
    dets=[]
    mins=[]
    maxs=[]
    h=[]
    N=[]
    M=[]
    for k,j in enumerate(2**np.arange(4)):
        mydict = mat2py.read(os.path.expanduser('~/scratch/'+fstart+'_visco_nove%03d' % (j*20) + fend + '.mat'))
        h.append(mydict['pdict']['gridspc'][0,0])
        N.append(mydict['pdict']['N'])
        M.append(mydict['pdict']['M'])
        times = mydict['t'].flatten()
        ctemp = []
        mtemp=[]
        Mtemp=[]
        for t in range(len(times)):
            x = ( (mydict['alldets'][t,:,:] - 0)**2 ).sum()
            l2measure = h[k] * np.sqrt(x)
            ctemp.append(l2measure) 
            mtemp.append(np.min(mydict['alldets'][t,:,:]))
            Mtemp.append(np.max(mydict['alldets'][t,:,:]))
        dets.append(ctemp)
        mins.append(mtemp)
        maxs.append(Mtemp)
    dets = np.asarray(dets).squeeze().transpose()
    mins = np.asarray(mins).squeeze().transpose()
    maxs = np.asarray(maxs).squeeze().transpose()
    Rats = np.zeros((dets.shape[0]-tbegin,2))
    plt.clf()
    for i in [1]:
        Rats[:,i] = ( dets[tbegin:,i] - dets[tbegin:,i+1] ) / ( dets[tbegin:,i+1] - dets[tbegin:,i+2] )
        plt.plot(mydict['t'].flatten()[tbegin:],Rats[:,i])
#    print(np.max(Rats))
    plt.title('Ratios for det ($L_2$ measure)')
    plt.xlabel('Time')
    plt.savefig(os.path.expanduser('~/scratch/' + fname1)  )
    plt.clf()
    for i, k in enumerate(['$N=20$','$N=40$','$N=80$','$N=160$']):
        plt.plot(mydict['t'].flatten(),dets[:,i],label=k)
    plt.title('Det ($L_2$ measure)')
    plt.xlabel('Time')
    plt.legend(loc=0)
    plt.savefig(os.path.expanduser('~/scratch/' + fname2) )
    plt.clf()
    for i, k in enumerate(['$N=20$','$N=40$','$N=80$','$N=160$']):
        plt.plot(mydict['t'].flatten(),mins[:,i],label=k)
    plt.title('Min det over domain')
    plt.xlabel('Time')
    plt.legend(loc=0)
    plt.savefig(os.path.expanduser('~/scratch/' + fname3) )
    plt.clf()
    for i, k in enumerate(['$N=20$','$N=40$','$N=80$','$N=160$']):
        plt.plot(mydict['t'].flatten(),maxs[:,i],label=k)
    plt.title('Max det over domain')
    plt.xlabel('Time')
    plt.legend(loc=0)
    plt.savefig(os.path.expanduser('~/scratch/' + fname4) )
    print('done')



def plotdetl1error(fend='_nodegrid',fstart='swimmerC',tbegin=3):
    if fend == '' or fend == '_centergrid':
        fname1 = 'center_'+fstart+'_deterrratL1' 
        fname2 = 'center_'+fstart+'_deterrL1'      
    elif fend == '_nodegrid':
        fname1 = 'node_'+fstart+'_deterrratL1'
        fname2 = 'node_'+fstart+'_deterrL1'      
    elif fend == '_centergrid_lowerfish':
        fname1 = 'lower_'+fstart+'_deterrratL1'
        fname2 = 'lower_'+fstart+'_deterrL1'      
    dets=[]
    h=[]
    N=[]
    M=[]
    for k,j in enumerate(2**np.arange(4)):
        mydict = mat2py.read(os.path.expanduser('~/scratch/'+fstart+'_visco_nove%03d' % (j*20) + fend + '.mat'))
        h.append(mydict['pdict']['gridspc'][0,0])
        N.append(mydict['pdict']['N'])
        M.append(mydict['pdict']['M'])
        times = mydict['t'].flatten()
        ctemp = []
        for t in range(len(times)):
            x = ( np.abs(mydict['alldets'][t,:,:] - 1) ).sum()
            l1measure = h[k]**2 * x
            ctemp.append(l1measure) 
        dets.append(ctemp)
    dets = np.asarray(dets).squeeze().transpose()
    Rats = np.zeros((dets.shape[0]-tbegin,2))
    plt.clf()
    for i in [1]:
        Rats[:,i] = ( dets[tbegin:,i] - dets[tbegin:,i+1] ) / ( dets[tbegin:,i+1] - dets[tbegin:,i+2] )
        plt.plot(mydict['t'].flatten()[tbegin:],Rats[:,i])
    print(np.max(Rats))
    plt.title('Ratios for det $L_1$ error')
    plt.xlabel('Time')
    plt.savefig( os.path.expanduser('~/scratch/' + fname1)  )
    plt.clf()
    for i, k in enumerate(['$N=20$','$N=40$','$N=80$','$N=160$']):
        plt.plot(mydict['t'].flatten(),dets[:,i],label=k)
    plt.title('Det $L_1$ error')
    plt.xlabel('Time')
    plt.legend(loc=0)
    plt.savefig(os.path.expanduser('~/scratch/' + fname2) )
    print('done')


def plotdetl2error(fend='_nodegrid',fstart='swimmerC',tbegin=3):
    if fend == '' or fend == '_centergrid':
        fname1 = 'center_'+fstart+'_deterrratL2' 
        fname2 = 'center_'+fstart+'_deterrL2'      
    elif fend == '_nodegrid':
        fname1 = 'node_'+fstart+'_deterrratL2'
        fname2 = 'node_'+fstart+'_deterrL2'      
    elif fend == '_centergrid_lowerfish':
        fname1 = 'lower_'+fstart+'_deterrratL2' 
        fname2 = 'lower_'+fstart+'_deterrL2'  
            
    dets=[]
    h=[]
    N=[]
    M=[]
    for k,j in enumerate(2**np.arange(4)):
        mydict = mat2py.read(os.path.expanduser('~/scratch/'+fstart+'_visco_nove%03d' % (j*20) + fend + '.mat'))
        h.append(mydict['pdict']['gridspc'][0,0])
        N.append(mydict['pdict']['N'])
        M.append(mydict['pdict']['M'])
        times = mydict['t'].flatten()
        ctemp = []
        for t in range(len(times)):
            x = ( (mydict['alldets'][t,:,:] - 1)**2 ).sum()
            l2measure = h[k] * np.sqrt(x)
            ctemp.append(l2measure) 
        dets.append(ctemp)
    dets = np.asarray(dets).squeeze().transpose()
    Rats = np.zeros((dets.shape[0]-tbegin,2))
    plt.clf()
    for i in [1]:
        Rats[:,i] = ( dets[tbegin:,i] - dets[tbegin:,i+1] ) / ( dets[tbegin:,i+1] - dets[tbegin:,i+2] )
        plt.plot(mydict['t'].flatten()[tbegin:],Rats[:,i])
    print(np.max(Rats))
    plt.title('Ratios for det $L_2$ error')
    plt.xlabel('Time')
    plt.savefig( os.path.expanduser('~/scratch/' + fname1)  )
    plt.clf()
    for i, k in enumerate(['$N=20$','$N=40$','$N=80$','$N=160$']):
        plt.plot(mydict['t'].flatten(),dets[:,i],label=k)
    plt.title('Det $L_2$ error')
    plt.xlabel('Time')
    plt.legend(loc=0)
    plt.savefig(os.path.expanduser('~/scratch/' + fname2) )
    print('done')


def pcolordet(fend='_nodegrid',fstart='swimmerC'):
    plt.close()
    fig = plt.figure()
    for k,j in enumerate(2**np.arange(4)):
        mydict = mat2py.read(os.path.expanduser('~/scratch/'+fstart+'_visco_nove%03d' % (j*20) + fend + '.mat'))
        fname='swimmerv_dets%03d_' % (j*20) + fstart + '_' + fend +'_'
        vmin=np.min(mydict['alldets'])
        vmax=np.max(mydict['alldets'])
        x = mydict['l'][:,:,:,0]
        y = mydict['l'][:,:,:,1]
        xmin = np.min(x)
        xmax = np.max(x)
        ymin = np.min(y)
        ymax = np.max(y)
        sx = mydict['fpts'][:,:-1:2]
        sy = mydict['fpts'][:,1::2]

        c=0
        for t in range(len(mydict['t'].flatten())):
    #        vmin=np.min(mydict['alldets'][t,:,:])
    #        vmax=np.max(mydict['alldets'][t,:,:])
            plt.pcolor(x[t,:,:],y[t,:,:],mydict['alldets'][t,:,:],vmin=vmin,vmax=vmax,cmap='jet')
            plt.colorbar()
            if len(sx[t,:])>1:
                plt.plot(sx[t,:],sy[t,:],'k',linewidth=4.0)
            else:
                plt.plot(sx[t,:],sy[t,:],'k.',markersize=16.0)
            plt.title('Time = '+str(mydict['t'][0,t]))
            plt.savefig(os.path.expanduser('~/scratch/' + fname + '%03d' % c))
            c+=1
            plt.clf()
    plt.close()
    
def plotDiv(fend='_nodegrid',fstart='swimmerC',tbegin=6):
    if fend == '' or fend == '_centergrid':
        fname1 = 'center_'+fstart+'_divratL2' 
        fname2 = 'center_'+fstart+'_divL2'      
        fname3 = 'center_'+fstart+'_divratL1' 
        fname4 = 'center_'+fstart+'_divL1'      
    elif fend == '_nodegrid':
        fname1 = 'node_'+fstart+'_divratL2' 
        fname2 = 'node_'+fstart+'_divL2'       
        fname3 = 'node_'+fstart+'_divratL1' 
        fname4 = 'node_'+fstart+'_divL1'       
    elif fend == '_centergrid_lowerfish':
        fname1 = 'lower_'+fstart+'_divratL2'  
        fname2 = 'lower_'+fstart+'_divL2'    
        fname3 = 'lower_'+fstart+'_divratL1'  
        fname4 = 'lower_'+fstart+'_divL1'    
    divl2=[]
    divl1=[]
    N=[]
    M=[]
    for k,j in enumerate(2**np.arange(4)):
        mydict = mat2py.read(os.path.expanduser('~/scratch/'+fstart+'_visco_nove%03d' % (j*20) + fend + '.mat'))
        N.append(mydict['pdict']['N'])
        M.append(mydict['pdict']['M'])
        times = mydict['t'].flatten()
        x = mydict['l'][:,:,:,0]
        y = mydict['l'][:,:,:,1]
        dt = mydict['dt'].flatten()
        h = mydict['pdict']['gridspc'].flatten()
        ctemp = []
        ctemp1 = []
        Mv = np.zeros(mydict['l'][0,:,:,:].shape)
        for t in range(1,len(times)-1):            
            u = (x[t+1,:,:]-x[t-1,:,:])/(2*dt) #center difference approximation to velocity
            v = (y[t+1,:,:]-y[t-1,:,:])/(2*dt)
            Mv[:,:,0] = u
            Mv[:,:,1] = v
            gM = SD2D.vectorGrad(Mv,h,N[k],M[k]) # du/da (Lagrangian derivative of velocity)
            gMs = np.reshape(gM,(N[k]*M[k],2,2))
            gl = SD2D.vectorGrad(mydict['l'][t,:,:,:],h,N[k],M[k]) # F (Jacobian matrix)
            gls = np.reshape(gl,(N[k]*M[k],2,2))
            igls = cm.matinv2x2(gls) #inverse Jacobian matrix
            div=[]
            for i in range(N[k]*M[k]):
                gradu = np.dot(gMs[i,:,:],igls[i,:,:]) # du/dx = (du/da)F^{-1} (Eularian derivative of vel is transform of Lagrangian)
                div.append(gradu[0,0]+gradu[1,1]) # div(u) = trace(du/dx)
            l2measure = h * np.sqrt( (np.asarray(div)**2).sum() )
            l1measure = h**2 * ( np.abs(np.asarray(div)) ).sum()
            ctemp.append(l2measure) 
            ctemp1.append(l1measure) 
        divl2.append(ctemp)
        divl1.append(ctemp1)
    divl2 = np.asarray(divl2).squeeze().transpose()
    Rats = np.zeros((divl2.shape[0]-tbegin+1,2))
    plt.clf()
    for i in [0]:
        Rats[:,i] = ( divl2[(tbegin-1):,i] - divl2[(tbegin-1):,i+1] ) / ( divl2[(tbegin-1):,i+1] - divl2[(tbegin-1):,i+2] )
        plt.plot(times[tbegin:-1],Rats[:,i])
    print(np.max(Rats))
    plt.title('Ratios for divergence $L_2$')
    plt.xlabel('Time')
    plt.savefig( os.path.expanduser('~/scratch/' + fname1)  )
    plt.clf()
    for i, k in enumerate(['$N=20$','$N=40$','$N=80$','$N=160$']):
        plt.plot(times[1:-1],divl2[:,i],label=k)
    plt.title('Div $L_2$')
    plt.xlabel('Time')
    plt.legend(loc=0)
    plt.savefig(os.path.expanduser('~/scratch/' + fname2) )

    divl1 = np.asarray(divl1).squeeze().transpose()
    Rats1 = np.zeros((divl1.shape[0]-tbegin+1,2))
    plt.clf()
    for i in [0]:
        Rats[:,i] = ( divl1[(tbegin-1):,i] - divl1[(tbegin-1):,i+1] ) / ( divl1[(tbegin-1):,i+1] - divl1[(tbegin-1):,i+2] )
        plt.plot(times[tbegin:-1],Rats[:,i])
    print(np.max(Rats))
    plt.title('Ratios for divergence $L_1$')
    plt.xlabel('Time')
    plt.savefig( os.path.expanduser('~/scratch/' + fname3)  )
    plt.clf()
    for i, k in enumerate(['$N=20$','$N=40$','$N=80$','$N=160$']):
        plt.plot(times[1:-1],divl1[:,i],label=k)
    plt.title('Div $L_1$')
    plt.xlabel('Time')
    plt.legend(loc=0)
    plt.savefig(os.path.expanduser('~/scratch/' + fname4) )
    print('done')

def plotMaxMinDiv(fend='_nodegrid',fstart='swimmerC',tbegin=6):
    if fend == '' or fend == '_centergrid':
        fname1 = 'center_'+fstart+'_maxdiv' 
        fname2 = 'center_'+fstart+'_mindiv'      
    elif fend == '_nodegrid':
        fname1 = 'node_'+fstart+'_maxdiv' 
        fname2 = 'node_'+fstart+'_mindiv'    
    elif fend == '_centergrid_lowerfish':
        fname1 = 'lower_'+fstart+'_maxdiv'   
        fname2 = 'lower_'+fstart+'_mindiv'  
    maxdiv=[]
    mindiv=[]
    N=[]
    M=[]
    for k,j in enumerate(2**np.arange(4)):
        mydict = mat2py.read(os.path.expanduser('~/scratch/'+fstart+'_visco_nove%03d' % (j*20) + fend + '.mat'))
        N.append(mydict['pdict']['N'])
        M.append(mydict['pdict']['M'])
        times = mydict['t'].flatten()
        x = mydict['l'][:,:,:,0]
        y = mydict['l'][:,:,:,1]
        dt = mydict['dt'].flatten()
        h = mydict['pdict']['gridspc'].flatten()
        ctemp = []
        ctemp1 = []
        Mv = np.zeros(mydict['l'][0,:,:,:].shape)
        for t in range(1,len(times)-1):            
            u = (x[t+1,:,:]-x[t-1,:,:])/(2*dt) #center difference approximation to velocity
            v = (y[t+1,:,:]-y[t-1,:,:])/(2*dt)
            Mv[:,:,0] = u
            Mv[:,:,1] = v
            gM = SD2D.vectorGrad(Mv,h,N[k],M[k]) # du/da (Lagrangian derivative of velocity)
            gMs = np.reshape(gM,(N[k]*M[k],2,2))
            gl = SD2D.vectorGrad(mydict['l'][t,:,:,:],h,N[k],M[k]) # F (Jacobian matrix)
            gls = np.reshape(gl,(N[k]*M[k],2,2))
            igls = cm.matinv2x2(gls) #inverse Jacobian matrix
            div=[]
            for i in range(N[k]*M[k]):
                gradu = np.dot(gMs[i,:,:],igls[i,:,:]) # du/dx = (du/da)F^{-1} (Eularian derivative of vel is transform of Lagrangian)
                div.append(gradu[0,0]+gradu[1,1]) # div(u) = trace(du/dx)
            ctemp.append(np.max(div)) 
            ctemp1.append(np.min(div)) 
        maxdiv.append(ctemp)
        mindiv.append(ctemp1)
    maxdiv = np.asarray(maxdiv).squeeze().transpose()
    mindiv = np.asarray(mindiv).squeeze().transpose()
    plt.clf()
    for i, k in enumerate(['$N=20$','$N=40$','$N=80$','$N=160$']):
        plt.plot(times[1:-1],maxdiv[:,i],label=k)
    plt.title('Max Div')
    plt.xlabel('Time')
    plt.legend(loc=0)
    plt.savefig(os.path.expanduser('~/scratch/' + fname1) )
    plt.clf()
    for i, k in enumerate(['$N=20$','$N=40$','$N=80$','$N=160$']):
        plt.plot(times[1:-1],mindiv[:,i],label=k)
    plt.title('Min Div')
    plt.xlabel('Time')
    plt.legend(loc=0)
    plt.savefig(os.path.expanduser('~/scratch/' + fname2) )
    print('done')

        
if __name__ == '__main__':
    for fend in ['_centergrid_lowerfish']:
        fstart='swimmerC'
#        plotcomponents(fend,fstart)       
#        plotdets(fend,fstart)
#        plotdetl1error(fend,fstart)
#        plotdetl2error(fend,fstart)
#        pcolordet(fend,fstart)
#        plotDiv(fend,fstart)
        plotMaxMinDiv(fend,fstart)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
