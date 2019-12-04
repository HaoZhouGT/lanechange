import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# print 'load trajectory data'
# time_list = np.loadtxt('time_list')
# loc_list = np.loadtxt('loc_list')
# # print('location list is',time_list)
# speed_list = np.loadtxt('speed_list')
#
# data = []
# for i in range(113397):
#     if 3025<loc_list[i]<3080:
#         data.append([time_list[i],speed_list[i]])
# data = np.array(data)
# print(data[:,0])
# print('the length of data',len(data))
# for the data, they just plotted the discrete points
# how to get point values

# plt.figure()
# plt.title('speed data')
# plt.scatter(data[:,0],data[:,1])
# plt.show()

# we should create a list with only origin data, then estimate the predicting performance
# you need to compute the average speed in that cell at each time step
# real_speed=np.zeros(3600)
# for i in range(3600):
#     temp_list = []
#     for j in range(len(data[:,0])):
#         if i == int(data[:,0][j]):
#             temp_list.append(data[:,1][j])
#     temp_list = np.array(temp_list)
#     real_speed[i] = np.mean(temp_list)

# np.savetxt('realspeed',real_speed)
real_speed = np.loadtxt('realspeed')
# you cannot plot NaN , you can only skip them when computing the error
simu_speed = np.loadtxt('upspeed')
para_speed = np.loadtxt('para_result')
plt.figure()
plt.xlabel('time (seconds)')
plt.ylabel('speed (mi/h)')
sns.lineplot(data=real_speed,alpha=0.7)
sns.lineplot(data=para_speed, linewidth=3)
sns.lineplot(data=simu_speed, linewidth=1.5)
plt.legend(['data','CTM','CTM + LC'])
plt.show()

jam_density = 150.0
wave_speed = 15.0

def speet2flow(v):
    if v==np.NaN:
        flow = 0.0
    if v==0.0:
        flow = 0.0
    else:
        flow=jam_density/(1.0/v+1.0/wave_speed)
    return flow

sum=0.0
for i in range(len(simu_speed)):
    sum+=speet2flow(simu_speed[i])
capacity_lc = sum/3600.0

print('the capacity using LC_method is',capacity_lc)

sum=0.0
for i in range(len(para_speed)):
    sum+=speet2flow(para_speed[i])
capacity_para = sum/3600.0
print('the capacity using para_method is',capacity_para)


sum=0.0
count = 0
for i in range(len(real_speed)):
    # print('the real flow is',speet2flow(real_speed[i]))
    flow = speet2flow(real_speed[i])
    if np.isnan(flow)!=True:
        sum+=flow
        count+=1
capacity_data = sum/count
print('the capacity given by data is',capacity_data)

speed_data = []
for i in range(len(real_speed)):
    if np.isnan(real_speed[i])!=True:
        speed_data.append(real_speed[i])
speed_data = np.array(speed_data)

plt.figure()
plt.plot(speed_data)
plt.title('filtered speed data at the upstream location')
plt.show()