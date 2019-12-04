import numpy as np
import matplotlib.pyplot as plt
from math import exp

u = 75.0
kj = 200.0
w = 15.0
kc = w * kj / (w + u)
Q =  u * kc

pe=0.4
xi=0.04
tua=10.0

NA=[0]

arrival=7200*np.random.rand(1,5)
print(arrival)

arrival=[390,929,3027,6920,6607]
arrival=np.array(arrival)
y=np.array([0,0,0,0,0])

print 'load upstream density'
upstream_den = np.loadtxt('upstream_den')


Tdemand=np.loadtxt('Tdemand')
Tsupply=np.loadtxt('Tsupply')

rate=[]
fraction=[]
frac2=[]
frac3=[]
sum=0
sendingflow=[]
meanrate=[]

for i in range(7199):
    k=upstream_den[i]
    sendflow=min(Q,u*k)
    sendingflow.append(sendflow)
    dmlc=sendflow*pe*xi/tua
    meanrate.append(dmlc)
    sum=sum+dmlc/3600*0.5
    rate.append(np.random.poisson(dmlc))
    NA.append(sum)
    fraction.append(min(1,Tsupply[i]/(dmlc+Tdemand[i])))
    if i in arrival:
        frac3.append((720+3000)/Tsupply[i])
    # rate is the demand flow of DMLC vehicles from the origin lane

np.array(rate)
np.array(fraction)
np.array(NA)
np.array(meanrate)

print('waiting time:',frac3)


demand=rate+Tdemand
supply=Tsupply
fraction=supply/demand
frac2 = supply

plt.figure(1)
plt.plot(rate)
plt.plot(sendingflow)
# plt.plot(demand)
plt.plot(supply)
plt.plot(arrival,y,marker='s',markersize=8,label="DMLC occuring time")
plt.legend(numpoints=1,loc='best')
# plt.title('Competition between two lanes using IT Principle')
plt.text(500,2300,'The demand of through flow on the target lane')
plt.text(500,200,'The supply function of the target lane')
# plt.text(500,-120,'The demand of DMLC flow from the origin lane')

plt.ylabel('flow (veh/h)')
plt.xlabel('time (0.5s)')

# plt.figure(2)
# plt.plot(fraction)
# plt.plot(NA)
# rectangle = plt.Rectangle((10, 10), 100, 100, fc='r')
# plt.plot(arrival)
# plt.title('The fraction of desired flow advancing to the target lane by IT Principle')
# plt.text(500,30,'The fraction of last minute flow advancing to the target lane')
#
# plt.ylabel('Fraction')
# plt.xlabel('Time')



plt.figure(3)
plt.plot(meanrate)
plt.plot(supply)
rectangle1 = plt.Rectangle((390, 0), 60, 720/60, fc='r',label="MLC waiting time")
rectangle2 = plt.Rectangle((929, 0), 66, 720/66, fc='r')
rectangle3 = plt.Rectangle((3027, 0), 40, 720/40, fc='r')
rectangle4 = plt.Rectangle((6607, 0), 48, 720/48, fc='r')
rectangle5 = plt.Rectangle((6920, 0), 23, 720/23, fc='r')
plt.gca().add_patch(rectangle1)
plt.gca().add_patch(rectangle2)
plt.gca().add_patch(rectangle3)
plt.gca().add_patch(rectangle4)
plt.gca().add_patch(rectangle5)
plt.legend(numpoints=1)

plt.ylabel('flow (veh/h)')
plt.xlabel('Time (0.5s)')
plt.text(2500,50,'The rate of last-minute exiting vehicles')
plt.text(1500,150,'The available capacity on the target lane')



# plt.figure(4)
# sup_2=supply*0.5
# plt.plot(rate)
# plt.plot(sup_2)
# plt.title('The demand of DMLC flow and the allocated capacity')
# plt.ylabel('flow')
# plt.xlabel('Time')


plt.show()

