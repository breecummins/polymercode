import numpy as np
import os
import sys

class CubicRegStokeslet(object):
    '''
    Regularized 2D Stokeslet, cube power. eps is the blob parameter, 
    mu is dynamic fluid viscosity.
    
    '''
    def __init__(self,eps,mu):
        self.eps = eps        
        self.mu = mu    
        
    def kernValsOriginal(self,pt,nodes):
        '''
        Build the matrix rows for the original Riemann sum method. 'pt' is the point 
        where we want to know velocity and 'nodes' are the locations of the forces.
        
        '''
        dif = pt - nodes
        r2 = (dif**2).sum(1) + self.eps**2
        xdiff = dif[:,0]
        ydiff = dif[:,1]
        H1 = (2*self.eps**2/r2 - np.log(r2))/(8*np.pi*self.mu)
        H2 = (2/r2)/(8*np.pi*self.mu)
        N = nodes.shape[0]
        row1 = np.zeros((2*N,))
        row2 = np.zeros((2*N,))
        ind = 2*np.arange(N) 
        row1[ind] = (H1 + (xdiff**2)*H2)
        row1[ind+1] = ((xdiff*ydiff)*H2)
        row2[ind+1]= (H1 + (ydiff**2)*H2)
        row2[ind] = row1[ind+1]
        return row1, row2
                        
    def makeMatrixOriginal(self,obspts,nodes):
        '''
        Build the regularized Stokeslet matrix explicitly. Used for finding 
        the forces located at 'nodes' that will ensure known velocities at 'obspts'.
        'obspts' is M by 2 and 'nodes' is N x 2. 
        
        '''
        #Coordinates must be 2D.
        #assert all( [len(a.shape) == 2 and a.shape[1] == 2 for a in nodes, obspts] )
        mat = np.zeros((2*obspts.shape[0],2*nodes.shape[0]))
        for k in range(obspts.shape[0]):
            pt = obspts[k,:]
            mat[2*k,:], mat[2*k+1,:] = self.kernValsOriginal(pt,nodes)
        return mat
        
    def linOp(self,obspts,nodes,f):
        '''
        Calculates velocity at 'obspts' due to 'f' located at 'nodes'.
        'obspts' is M by 2 and 'nodes' and 'f' are N x 2.
        
        '''
        #Coordinates must be 2D.
        #assert all( [len(a.shape) == 2 and a.shape[1] == 2 for a in nodes, obspts, f] )
        output = np.zeros((2*obspts.shape[0],))
        for k in range(obspts.shape[0]):
            pt = obspts[k,:]
            row1, row2 = self.kernValsOriginal(pt,nodes)
            output[2*k] = (row1*f.flat).sum()
            output[2*k+1] = (row2*f.flat).sum()
        return output

class CubicRegStokesletDeriv(object):
    '''Derivative of regularized 2D Stokeslet, cube power. eps is the blob parameter, mu is dynamic fluid viscosity.'''
    def __init__(self,eps,mu):
        self.eps = eps        
        self.mu = mu    
        
    def calcIntegral(self,pt,nodes,f,F):
        '''
        del_v matrix action on 'f' for 'pt'. 'pt' is the point where we want to know 
        velocity and 'nodes' are the locations of the forces 'f'. 'F' is the deformation 
        matrix associated with 'pt'.
        
        '''
        dif = pt - nodes
        re2 = (dif**2).sum(1) + self.eps**2
        h2 = 1/re2
        dh2 = -2/re2**2 #derivative over r
        dh1 = -1/re2 + dh2*self.eps**2 #derivative over r
        fdotl = (f*dif).sum(1) #dot product for all points
        #transpose matrix products with Jacobian
        Fdx = F[0,0]*dif[:,0] + F[1,0]*dif[:,1]  
        Fdy = F[0,1]*dif[:,0] + F[1,1]*dif[:,1] 
        Fdif = [Fdx,Fdy] 
        Ffx = F[0,0]*f[:,0] + F[1,0]*f[:,1]
        Ffy = F[0,1]*f[:,0] + F[1,1]*f[:,1]
        Ff = [Ffx,Ffy] 
        delv = np.zeros((2,2))
        for i in range(2):
            for k in range(2):
                val = dh1*f[:,i]*Fdif[k] + dh2*dif[:,i]*fdotl*Fdif[k] + h2*fdotl*F[i,k] + h2*dif[:,i]*Ff[k]
                delv[i,k] = val.sum() #summing over j (l1, l2 coordinates), and summing over all nodes (integration)
        return delv/(4*self.mu*np.pi)
        
    def linOp(self,obspts,nodes,f,F):
        '''Calculates velocity at 'obspts' due to 'f' located at 'nodes'. 'F' contains the Jacobians of obspts.'''
        output = np.zeros((obspts.shape[0],2,2))
        for k in range(obspts.shape[0]):
            pt = obspts[k,:]
            Fh = F[k,:,:]
            output[k,:,:] = self.calcIntegral(pt,nodes,f,Fh)
        return output
