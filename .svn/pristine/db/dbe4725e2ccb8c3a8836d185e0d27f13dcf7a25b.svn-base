#from pylab import *

def f(t):
    return exp(-t) * cos(2*pi*t)
    
def plotinipython(): 
    t1 = np.arange(0.0, 5.0, 0.1)
    t2 = np.arange(0.0, 5.0, 0.02)
    
    figure(1)
    subplot(211)
    plot(t1, f(t1), 'bo', t2, f(t2), 'k')
    
    subplot(212)
    plot(t2, np.cos(2*np.pi*t2), 'r--')
    
def plotinshell(): 
    t1 = np.arange(0.0, 5.0, 0.1)
    t2 = np.arange(0.0, 5.0, 0.02)
    
    subplot(211)
    plot(t1, f(t1), 'bo', t2, f(t2), 'k')
    
    subplot(212)
    plot(t2, np.cos(2*np.pi*t2), 'r--')
    savefig('/Users/bcummins/scratch/alldone', dpi=400) #always call savefig before show(), since show() will exit program after fig close
    show()
    
def threeDplot():
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter, NullLocator
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig = plt.figure()
    ax = fig.gca(projection='3d',azim=0,elev=90)
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.hot,
            linewidth=0, antialiased=False)
    ax.set_zlim3d(-1.01, 1.01)
    
    #ax.w_zaxis.set_major_locator(NullLocator())
    ax.w_zaxis.set_major_locator(LinearLocator(3))
    ax.w_zaxis.set_major_formatter(FormatStrFormatter(''))
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    #for tick in ax.w_zaxis.get_major_ticks():
        #tick.tick1On = False
        #tick.tick2On = False
        #print(tick.__dict__)
    #ax.w_zaxis.set_ticklabels('')
    #ax.w_zaxis.set_major_locator()
    plt.show()
    
def colorplot():
    from matplotlib import plt

    
if __name__ == '__main__':
    threeDplot()