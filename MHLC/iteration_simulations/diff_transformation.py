import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()



real_speed = np.loadtxt('processed_speed')
simu_speed = np.loadtxt('upspeed')
para_speed = np.loadtxt('para_result')


plt.figure()
plt.xlabel('time (seconds)')
plt.ylabel('speed (mi/h)')
sns.lineplot(data=real_speed,alpha=0.7)
sns.lineplot(data=para_speed, linewidth=3)
sns.lineplot(data=simu_speed, linewidth=1.5)
plt.legend(['data','CTM','CTM + LC'])

def diff_tran(a):
    trans = []
    for i in range(1,len(a),1):
        trans.append(np.log(a[i]/a[i-1]))
    trans = np.array(trans)
    return trans

## doing the difference transformation of the three time series
real = diff_tran(real_speed)
simu = diff_tran(simu_speed)
para = diff_tran(para_speed)


plt.figure()
plt.xlabel('time (seconds)')
plt.ylabel('speed (mi/h)')
sns.lineplot(data=real,linewidth=1.5,alpha=0.7)
sns.lineplot(data=simu, linewidth=1.5)
sns.lineplot(data=para, linewidth=1.5)
plt.legend(['data','CTM','CTM + LC'])
plt.show()

