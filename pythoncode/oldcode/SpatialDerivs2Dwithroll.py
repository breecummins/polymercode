import numpy as np
import sys

def vectorGrad(vec,gridspc,N,M):
    '''vec is a 3D numpy array with first index = i (x-coord), second index = j (y-coord), third index = vector component. gridspc is the grid spacing in both x and y. N is the number of points in the x direction, M is the number in the y direction. The output is a 4D numpy array (shape = (N,M,2,2)) containing a discrete center difference approximation to the vector gradient, with one-sided derivatives at the boundaries.'''
    if N != vec.shape[0] or M != vec.shape[1] or vec.shape[2] != 2:
        print('Shape mismatch. Aborting')
    out = np.zeros((N,M,2,2))
    #domain center
    out[:,:,:,0] = np.roll(vec,-1,axis=0) - np.roll(vec,1,axis=0)
    out[:,:,:,1] = np.roll(vec,-1,axis=1) - np.roll(vec,1,axis=1)
    out = out/(2*gridspc)
    #fix domain edges
    dxem1 = vec[[0,N-2],:,:]
    dxe = vec[[1,N-1],:,:]
    dyem1 = vec[:,[0,M-2],:]
    dye = vec[:,[1,M-1],:]
    out[[0,N-1],:,0,0] = (dxe[:,:,0] - dxem1[:,:,0])/gridspc
    out[[0,N-1],:,1,0] = (dxe[:,:,1] - dxem1[:,:,1])/gridspc
    out[:,[0,M-1],0,1] = (dye[:,:,0] - dyem1[:,:,0])/gridspc
    out[:,[0,M-1],1,1] = (dye[:,:,1] - dyem1[:,:,1])/gridspc
    return out
    
def tensorDiv(tensor,gridspc,N,M):
    '''tensor is a 4D numpy array with first index = i (x-coord), second index = j (y-coord), third index = tensor row component, fourth index = tensor column component. gridspc is the grid spacing in both x and y. N is the number of points in the x direction, M is the number in the y direction. The output is a 3D numpy array containing a discrete center difference approximation to the divergence, with one-sided derivatives at the boundaries.'''
    if N != tensor.shape[0] or M != tensor.shape[1] or tensor.shape[2] != 2 or tensor.shape[3] != 2:
        print('Shape mismatch. Aborting')
    out = np.zeros((N,M,2))
    #domain center
    dxm1 = tensor[:N-2,:,:,:] 
    dxp1 = tensor[2:,:,:,:]
    dym1 = tensor[:,:M-2,:,:] 
    dyp1 = tensor[:,2:,:,:]
    out[1:N-1,:,0] = dxp1[:,:,0,0] - dxm1[:,:,0,0] 
    out[:,1:M-1,0] = out[:,1:M-1,0] + dyp1[:,:,0,1] - dym1[:,:,0,1]
    out[1:N-1,:,1] = dxp1[:,:,1,0] - dxm1[:,:,1,0] 
    out[:,1:M-1,1] = out[:,1:M-1,1] + dyp1[:,:,1,1] - dym1[:,:,1,1]
    out = out/(2*gridspc)
    #domain edges
    dxem1 = tensor[[0,N-2],:,:,:]
    dxe = tensor[[1,N-1],:,:,:]
    dyem1 = tensor[:,[0,M-2],:,:]
    dye = tensor[:,[1,M-1],:,:]
    out[[0,N-1],:,0] = out[[0,N-1],:,0] + (dxe[:,:,0,0] - dxem1[:,:,0,0])/gridspc
    out[[0,N-1],:,1] = out[[0,N-1],:,1] + (dxe[:,:,1,0] - dxem1[:,:,1,0])/gridspc
    out[:,[0,M-1],0] = out[:,[0,M-1],0] + (dye[:,:,0,1] - dyem1[:,:,0,1])/gridspc
    out[:,[0,M-1],1] = out[:,[0,M-1],1] + (dye[:,:,1,1] - dyem1[:,:,1,1])/gridspc
    return out
    
    
def testme():
    import main
    gridspc = 0.2
    N = 6
    M = 11
    l = main.makeGrid(N,M,gridspc)
    
    #test vectorGrad
    f = np.zeros(l.shape)
    f[:,:,0] = l[:,:,0]**2 + l[:,:,1]**2
    f[:,:,1] = 2*l[:,:,0]
    gfa = vectorGrad(f,gridspc,N,M)
    gf = np.zeros(gfa.shape)
    gf[:,:,0,0] = 2*l[:,:,0]
    gf[:,:,0,1] = 2*l[:,:,1]
    gf[:,:,1,0] = 2*np.ones((N,M))
    gf[:,:,1,1] = np.zeros((N,M))
    print('gfa[N-1,0,:,:]')
    print(gfa[ N-1,0,:,:])
    print('gf[ N-1,0,:,:]')
    print(gf[  N-1,0,:,:])
    print(np.max(np.abs(gfa-gf)))
    
    # #test tensorDiv
    # F = np.zeros((N,M,2,2))
    # F[:,:,0,0] = l[:,:,0]**2 + l[:,:,1]
    # F[:,:,1,0] = 2*l[:,:,0]
    # F[:,:,0,1] = l[:,:,1]
    # F[:,:,1,1] = l[:,:,0] + l[:,:,1]**2
    # dFa = tensorDiv(F,gridspc,N,M)
    # dF = np.zeros(dFa.shape)
    # dF[:,:,0] = 2*l[:,:,0] + 1
    # dF[:,:,1] = 2*l[:,:,1] + 2
    # print('dFa[0,1,:]')
    # print(dFa[ 0,1,:])
    # print('dF[ 0,1,:]')
    # print(dF[  0,1,:])
    # print(np.max(np.abs(dFa-dF)))
    # print(np.abs(dFa-dF))
    
def profileme():
    gridspc = 0.1
    N = 1000
    M = 1000
    l = np.zeros((N,M,2))
    x = np.zeros((N,1))
    x[:,0] = np.linspace(0+gridspc/2,0+N*gridspc-gridspc/2,N)
    l[:,:,0] = np.tile(x,(1,M))
    y = np.linspace(1+gridspc/2,1+M*gridspc-gridspc/2,M)
    l[:,:,1] = np.tile(y,(N,1))
    
    #profile vectorGrad
    f = apply(np.random.rand,l.shape)
    gfa = vectorGrad(f,gridspc,N,M)
    
        
if __name__ == '__main__':
    profileme()
    
    
