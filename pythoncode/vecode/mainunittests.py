import numpy as nm

if __name__ == '__main__':
    #unit tests for matrix inverse; my code is working and faster (apparently). Could not get the timeit module to work.
    from main import matInv2x2
    from scipy.linalg import inv
    import time
    # M=nm.array([[1,2],[3,4]])
    M = nm.array([[8.2,3.4],[5.7, 0.2]])
    start = time.clock()
    B = matInv2x2(M)
    print(time.clock()-start)
    print(B)
    start = time.clock()
    B2=inv(M)
    print(time.clock()-start)
    print(B2)

    # #unit tests for 2D trap rule
    # from main import makeGrid, trapRule2D
    # N = 201
    # M = 201
    # sidelength = 2
    # h = sidelength/float(N-1)
    # l = makeGrid(N,M,h)
    # 
    # # #test 1, constant term.    Trap and lower-left Riemann should be indentical to the real answer. They are.
    # # V = 3*nm.ones(l.shape[0:2])
    # # realans = 12
    # 
    # # #test 2, trapezoid. Hmmm, trap is still the same as riemann.  Maybe I should get out of the linear regime.
    # # V = 3*nm.ones(l.shape[0:2])
    # # ind = nm.nonzero(l[:,1,0] < 0.5)
    # # jnd = nm.nonzero(l[:,1,0] > 1.5)
    # # V[ind,:] = 6*l[ind,:,0]
    # # V[jnd,:] = 12 - 6*l[jnd,:,0]
    # # realans = 9
    # 
    # # #test 3, quadratic. Turns out you need an antisymmetric function in order to make trap and riemann different. Anyway, this function appears to be working.
    # # V = l[:,:,0]**2 + l[:,:,1]**2
    # # realans = 32./3
    #     
    # approx = trapRule2D(V,h)
    # llriemann = h**2 * nm.sum(V[:-1,:-1])
    # urriemann = h**2 * nm.sum(V[1:,1:])
    # print(urriemann)
    # print(nm.abs(realans-urriemann))
    # print(approx)
    # print(nm.abs(realans-approx))
    # print(llriemann)
    # print(nm.abs(realans-llriemann))
