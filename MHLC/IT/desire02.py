import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()




u=110.0 # note that here we use the speed in ft/s
tau1=3.0
tau2=1.5 # it shows that the time tau2 will determine the shape of the curve
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
    v=40.0
    deltav=(4000-x)/4000.0*(v-10.0)
    pi1=deltav/75.0/tau1
    desire=pi1/u
    return desire

def phi2(x):
    pi2=np.exp(-1.0*x/800)/tau2
    desire=pi2/u
    return desire

def pi2(x):
    prob=np.exp(-1.0*x/800)
    return prob


s=[]
penetration=[]
pe=p0
pne=1-p0
dmlc=[]
prob=[]
cum=[]
sum=0


for i in range(70,1,-1):
    I1 = quad(phi1, (i-1)*dx, i*dx) # fraction of discretionary LC flow from i*dx to (i+1)*dx
    print('the probability of discretionary lane changes', I1[0])
    I2 = quad(phi2, (i-1)*dx, i*dx) # fraction of mandatory LC flow from i*dx to (i+1)*dx
    print('the probability of mandatory lane changes', I2[0])
    p=pne*I1[0]+pe*I2[0] # this is the fraction of sending flow that makes lane change
    print('the fraction of LC flow', p)
    s1=s0*(1-p)
    pe=pe*(1-I2[0])*s0/s1
    pne=1.0-pe
    prob.append(pi2(i*dx))
    s.append(s1)
    penetration.append(pe)
    dmlc.append(s0*pe*I2[0])
    cum.append(sum)
    sum=sum+s0*pe*I2[0]
    s0=s1

s=s[::-1] # we need to inverse the list, so the distance can actually start from 0 to 4000
dmlc = dmlc[::-1]
penetration=penetration[::-1]
prob=prob[::-1]
cum =cum[::-1]
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
x = np.linspace(0,4000,69)
xvals = np.linspace(0, 4000, 500)
dmlc = np.interp(xvals, x, dmlc)
s = np.interp(xvals, x, s)
penetration=np.interp(xvals, x, penetration)
prob = np.interp(xvals, x, prob)
cum = np.interp(xvals, x, cum)


ax1.set_title('The exponential desire of mandatory lane changes')
ax1.plot(xvals, prob,sns.xkcd_rgb["pale red"], lw=3)
ax2.set_title('The sending flow along distance to gore')
ax2.set_ylabel('flow(veh/h)', fontsize=10)
ax3.set_xlabel('distance to gore (ft)', fontsize=10)
ax2.plot(xvals, s, sns.xkcd_rgb["medium green"], lw=3)
ax3.set_title('The flow of mandatory lane changes')
ax3.set_ylabel('flow(veh/h)', fontsize=10)
ax3.plot(xvals, dmlc, sns.xkcd_rgb["denim blue"], lw=3)
ax4.set_title('The cumulative flow of mandatory lane changes')
ax4.set_ylabel('flow(veh/h)', fontsize=10)
ax4.set_xlabel('distance to gore (ft)', fontsize=10)
ax4.plot(xvals, cum,color="#9b59b6",lw=3)

sns.axes_style("darkgrid")

plt.show()



