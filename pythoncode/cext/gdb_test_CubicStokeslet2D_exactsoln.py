import CubicStokeslet2D as cm
import numpy as np
import vecode.RegularizedStokeslets2D as RS2D

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

#python output
blob = RS2D.CubicRegStokeslet(eps,mu)
output = blob.linOp(obspts,nodes,f)
print(output)

#exact solution
exact = np.array([-1./(18*np.pi), 1./(4*np.pi)*(np.log(3)/2 - 5/18.), 1./(18*np.pi), 1./(4*np.pi)*(np.log(3)/2 - 5/18.)])
print(exact)

#error
print('Error between exact and python')
print(exact-output)
print(np.max(np.abs(output-exact)))
print('Error between exact and C')
print(exact-out)
print(np.max(np.abs(out-exact)))
print('Error between C and python')
print(output-out)
print(np.max(np.abs(output-out)))

##############################
# test derivative kernel
##############################
print('Testing derivative kernel.')
out =cm.derivop(eps,mu,obspts,nodes,f,F)
print(out)

#python output
blobd = RS2D.CubicRegStokesletDeriv(eps,mu)
output = blobd.linOp(obspts,nodes,f,F)
print(output)

#exact solution
exact = np.array([ [[2./(81*np.pi)-1./(36*np.pi), 2./(81*np.pi)-1./(36*np.pi)],
[19./(324*np.pi), 19./(324*np.pi)-1./(18*np.pi)]],[[2./(81*np.pi)-1./(36*np.pi),
-2./(81*np.pi)+1./(36*np.pi)], [-19./(324*np.pi), 19./(324*np.pi)-1./(18*np.pi)]] ])
print(exact)

#error
print('Error between exact and python')
print(exact-output)
print(np.max(np.abs(output-exact)))
print('Error between exact and C')
print(exact-out)
print(np.max(np.abs(out-exact)))
print('Error between C and python')
print(output-out)
print(np.max(np.abs(output-out)))

