import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

u=110.0 # note that here we use the speed in ft/s
tau1=3.0
tau2=5.0
v4=60.0
s0=2500
p0=0.5
dx=55.0


L=4000
x= np.arange(0,4000,1)


def sending(x):
    flow = 2500*x/4000.0
    return flow


def phi1(x):
    v=60.0
    deltav=(4000-x)/4000.0*(v-10.0)
    pi1=deltav/75.0/tau1
    desire=pi1/u
    return desire

def phi2(x):
    pi2=np.exp(-1.0*x/800)/tau2
    desire=pi2/u
    return desire


s=[]
penetration=[]
pe=p0
pne=1-p0

for i in range(0,70,1):
    I1 = quad(phi1, i*dx, (i+1)*dx) # fraction of discretionary LC flow from i*dx to (i+1)*dx
    print('the probability of discretionary lane changes', I1[0])
    I2 = quad(phi2, i*dx, (i+1)*dx) # fraction of mandatory LC flow from i*dx to (i+1)*dx
    print('the probability of mandatory lane changes', I2[0])
    p=pne*I1[0]+pe*I2[0] # this is the fraction of sending flow that makes lane change
    print('the fraction of LC flow', p)
    s1=s0*(1-p)
    s.append(s1)
    s0=s1

plt.figure()
plt.plot(s)
# plt.plot(x,pi2(x)*sending(x))
# plt.plot(x,pi2(x))
# plt.plot(x,sending(x))
plt.show()