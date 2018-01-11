import matplotlib.pyplot as plt
import numpy as np
import mat2py
import os

def plotRelaxTimes():
    rextime=[]
    Wi = np.array([2,4,8,16,32,64,128])
    for w in Wi:
        mydict = mat2py.read(os.path.expanduser('~/rsyncfolder/data/VEflow/TwoHeadSpring/springtestC_visco%03d.mat' % w))
        vals = np.zeros(mydict['S'].shape)
        vals[:,:,:,0,0] = 1
        vals[:,:,:,1,1] = 1
        difs = np.abs(mydict['S'] - vals)
        print(len(mydict['t'].flatten()))
        for k in range(40,len(mydict['t'].flatten())):
            if np.all(difs[k,:,:,:,:]<0.06):
                rextime.append(mydict['t'].flatten()[k])
                print((w,k))
                break
            if k == len(mydict['t'])-1:
                print('no convergence for Wi = %f' % w)
        mydict={}
    print(rextime)
    plt.close()
    plt.plot(Wi/10.,np.asarray(rextime),'k',linewidth=4.0)
    plt.tick_params(labelsize=24)
    plt.title('Relaxation time vs Wi', fontsize=24)
    plt.xlabel('Wi', fontsize=24)
    plt.ylabel('Stress within 6% of I', fontsize=24)
    plt.savefig(os.path.expanduser('~/rsyncfolder/data/VEflow/TwoHeadSpring/relaxtimes.pdf'))
