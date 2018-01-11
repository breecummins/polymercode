import CubicStokeslet2D as cm
import numpy as np

#set variables
M = np.array([[1.,2.],[3.,4.]])
M = np.array([M,M,M])

#exact soln
Minv = np.array([[4.,-2.],[-3.,1.]])/(-2.)
Minv = np.array([Minv,Minv,Minv])

#########################
# test C code
#########################
out =cm.matinv2x2(M)
print(out)

#error
print('Error between exact and C')
print(Minv-out)
print(np.max(np.abs(out-Minv)))

