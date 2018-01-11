import CubicStokeslet2D as cm
import numpy as np
eps=1.0
mu=1.0

#########################
# test matrix mult
#########################
obspts=np.array([[1.,3.],[34.,8.]])
#pt=pt[np.newaxis,:]
nodes=np.arange(10,dtype='double').reshape((5,2))
print(nodes)
f=np.random.rand(*nodes.shape)
print(f)
out =cm.matmult(eps,mu,obspts,nodes,f)
print(out)

#python answer
output = np.zeros((2*obspts.shape[0],))
for k in range(obspts.shape[0]):
    pt = obspts[k,:]
    dif = pt - nodes
    r2 = (dif**2).sum(1) + eps**2
    xdiff = dif[:,0]
    ydiff = dif[:,1]
    H1 = (2*eps**2/r2 - np.log(r2))/(8*np.pi*mu)
    H2 = (2/r2)/(8*np.pi*mu)
    N = nodes.shape[0]
    row1 = np.zeros((2*N,))
    row2 = np.zeros((2*N,))
    ind = 2*np.arange(N) 
    row1[ind] = (H1 + (xdiff**2)*H2)
    row1[ind+1] = ((xdiff*ydiff)*H2)
    row2[ind+1]= (H1 + (ydiff**2)*H2)
    row2[ind] = row1[ind+1]
    output[2*k] = (row1*f.flat).sum()
    output[2*k+1] = (row2*f.flat).sum()
#error
print('Error is ....')
print(np.max(np.abs(output-out)))

##############################
# test derivative kernel
##############################
obspts = -5 + 10*np.array([[0.5, 7.1], [4.2, 4.6], [3.1, 8.2]])
#print(obspts)
nodes=-10 + 20*np.array([[1, 14.2], [8.4, 9.2], [6.2, 16.4], [19.8,2.3], [12.0,14.9]])
#print(nodes)
f=np.array([[1, 14.2], [8.4, 9.2], [6.2, 16.4], [19.8,2.3], [12.0,14.9]])/20.0
#print(f)
F = np.array([[[1.1,0.2],[0.15,0.95]],[[1.05,0.01],[0.1,0.92]],[[0.9,0.15],[0.2,1.01]]])
#print(F)
out =cm.derivop(eps,mu,obspts,nodes,f,F)
print(out)

#python output
output = np.zeros((obspts.shape[0],2,2))
for k in range(obspts.shape[0]):
    pt = obspts[k,:]
    Fh = F[k,:,:]
    dif = pt - nodes
    re2 = (dif**2).sum(1) + eps**2
    h2 = 1/re2
    dh2 = -2/re2**2 #derivative over r
    dh1 = -1/re2 + dh2*eps**2 #derivative over r
    fdotl = (f*dif).sum(1) #dot product for all points
    #transpose matrix products with Jacobian
    Fdx = Fh[0,0]*dif[:,0] + Fh[1,0]*dif[:,1]  
    Fdy = Fh[0,1]*dif[:,0] + Fh[1,1]*dif[:,1] 
    Fdif = [Fdx,Fdy] 
    Ffx = Fh[0,0]*f[:,0] + Fh[1,0]*f[:,1]
    Ffy = Fh[0,1]*f[:,0] + Fh[1,1]*f[:,1]
    Ff = [Ffx,Ffy] 
    delv = np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            val = dh1*f[:,i]*Fdif[j] + dh2*dif[:,i]*fdotl*Fdif[j] + h2*fdotl*Fh[i,j] + h2*dif[:,i]*Ff[j]
            delv[i,j] = val.sum() #summing over j (l1, l2 coordinates), and summing over all nodes (integration)
    output[k,:,:] = delv/(4*mu*np.pi)
#error
print(output)

print('Error is ....')
print(np.max(np.abs(output-out)))
print('Relative error is ....')
print(np.max(np.abs(output-out))/np.max(np.abs(output)))


print('Exact solution test...')
#set variables
print('Variable values...')
eps=1.0
mu=1.0
obspts=np.array([[2.,2.],[-2.,2.]])
print(obspts)
nodes=np.array([0.,0.])
nodes=nodes[np.newaxis,:]
print(nodes)
f=np.array([0.,-0.5])
f=f[np.newaxis,:]
print(f)
F = np.array([ [[1.,0.],[0.,1.]], [[1.,0.],[0.,1.]] ])
print(F)


#########################
# test matrix mult
#########################
print('Testing matrix multiplication.')
out =cm.matmult(eps,mu,obspts,nodes,f)
print(out)

#exact solution
exact = np.array([-1./(18*np.pi), 1./(4*np.pi)*(np.log(3)/2 - 5/18.), 1./(18*np.pi), 1./(4*np.pi)*(np.log(3)/2 - 5/18.)])
print(exact)

#error
print('Error between exact and C')
print(exact-out)
print(np.max(np.abs(out-exact)))

##############################
# test derivative kernel
##############################
print('Testing derivative kernel.')
out =cm.derivop(eps,mu,obspts,nodes,f,F)
print(out)

#exact solution
exact = np.array([ [[2./(81*np.pi)-1./(36*np.pi), 2./(81*np.pi)-1./(36*np.pi)],
[19./(324*np.pi), 19./(324*np.pi)-1./(18*np.pi)]],[[2./(81*np.pi)-1./(36*np.pi),
-2./(81*np.pi)+1./(36*np.pi)], [-19./(324*np.pi), 19./(324*np.pi)-1./(18*np.pi)]] ])
print(exact)

#error
print('Error between exact and C')
print(exact-out)
print(np.max(np.abs(out-exact)))



