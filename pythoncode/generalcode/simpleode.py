import numpy as nm
from scipy.integrate import ode
import mat2py

def f(t,y):
    return nm.array([ y[1], -y[0] ])

if __name__ == '__main__':
    y0, t0 = nm.array([0.25, 0.75]), 0
    r = ode(f).set_integrator('dopri5')
    r.set_initial_value(y0,t0)
    t1 = 4*nm.pi
    dt = 4*nm.pi/1000
    approx=nm.empty(0,)
    while r.successful() and r.t < t1+dt/2.:
        r.integrate(r.t+dt)
        approx = nm.append(approx,r.y[0])
    t = nm.arange(0,t1+dt/2.,dt)    
    realsoln = 0.25*nm.cos(t) + 0.75*nm.sin(t)
    mat2py.write('testsimpleode.mat', {'t':t,'approx':approx,'realsoln':realsoln})
    