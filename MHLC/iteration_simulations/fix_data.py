import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set()

# this code plots the comparison among the CTM_cont, CTM_

real_speed = np.loadtxt('processed_speed')
simu_speed = np.loadtxt('upspeed')
para_speed = np.loadtxt('para_result')

std1 = np.std(real_speed)
std2 = np.std(simu_speed)
std3 = np.std(para_speed)

print('the std of speed data is',std1)
print('the std of CTM+LC method is',std2)
print('the std of CTM method is',std3)

# the heavy congestion threshold is defined as 20 mi/h
threshold = 20.0
cong_data = np.array(filter(lambda x: x <= threshold, real_speed))
cong_lc = np.array(filter(lambda x: x <= threshold, simu_speed))
cong_para = np.array(filter(lambda x: x <= threshold, para_speed))

def cong_ratio(data):
    return len(data)/3600.0

print('the congestion ratio of speed data is',cong_ratio(cong_data))
print('the congestion ratio of CTM+LC method is',cong_ratio(cong_lc))
print('the congestion ratio of CTM method is',cong_ratio(cong_para))


print('the average amplitude of speed data is',np.mean(cong_data))
print('the average amplitude of CTM+LC method is',np.mean(cong_lc))
print('the average amplitude of CTM method is',np.mean(cong_para))
# plt.figure()
# plt.xlabel('time (seconds)')
# plt.ylabel('speed (mi/h)')
# sns.lineplot(data=cong_data,alpha=0.7)
# sns.lineplot(data=cong_lc, linewidth=3)
# sns.lineplot(data=cong_para, linewidth=1.5)
# plt.legend(['data','CTM','CTM + LC'])
# plt.show()


plt.figure()
plt.xlabel('time (seconds)')
plt.ylabel('speed (mi/h)')
sns.lineplot(data=real_speed,alpha=0.7)
sns.lineplot(data=para_speed, linewidth=3)
sns.lineplot(data=simu_speed, linewidth=1.5)
plt.legend(['data','CTM_cont','CTM_disc'])
plt.show()

# jam_density = 150.0
# wave_speed = 15.0
#
# def speet2flow(v):
#     if v==np.NaN:
#         flow = 0.0
#     if v==0.0:
#         flow = 0.0
#     else:
#         flow=jam_density/(1.0/v+1.0/wave_speed)
#     return flow

# sum=0.0
# for i in range(len(real_speed)):
#     sum+=speet2flow(real_speed[i])
# capacity_lc = sum/3600.0
# print('the capacity given by speed data',capacity_lc)
#
# sum=0.0
# for i in range(len(simu_speed)):
#     sum+=speet2flow(simu_speed[i])
# capacity_lc = sum/3600.0
# print('the capacity using LC_method is',capacity_lc)
#
# sum=0.0
# for i in range(len(para_speed)):
#     sum+=speet2flow(para_speed[i])
# capacity_para = sum/3600.0
# print('the capacity using para_method is',capacity_para)

#
# plt.figure()
# plt.plot(real_speed)
# plt.title('interpolated speed data at the upstream location')
# plt.show()