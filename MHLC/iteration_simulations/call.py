import iteration

# we call iteration code to run the simulation for N times
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()

ff_speed = 75.0
jam_density = 200.0
wave_speed = 15.0
kc = wave_speed * jam_density / (ff_speed + wave_speed)


def speed2flow(v):
    if v==0.0:
        flow = 0.0
    else:
        flow=jam_density/(1.0/v+1.0/wave_speed)
    return flow


# the number of iterations is N
N = 1000
iter_results =[]
for k in range(N):
    print('Conducting the',k,'th simulation')
    iter_results.append(iteration.main())
iter_results = np.array(iter_results)
# save and plot the mean speed for each iteration
# save and plot the s.t.d of speed for each iteration
average_speed = []
std = []
for m in range(N):
    speed_data = np.array(iter_results[m, :])
    average_speed.append(np.mean(speed_data))
    std.append(np.std(speed_data))
average_speed = np.array(average_speed)
std = np.array(std)
np.savetxt('average_speed',average_speed)
np.savetxt('standard_deviation',std)
mean_speed = np.mean(average_speed)
mean_std = np.mean(std)


capacity_LC = []
for m in range(N):
    speed_data = np.array(iter_results[m, :])
    sum = 0.0
    for i in range(len(speed_data)):
        sum+=speed2flow(speed_data[i])
    capacity_lc = sum/3600.0
    capacity_LC.append(capacity_lc)

capacity_LC=np.array(capacity_LC)
np.savetxt('capacity_simu',capacity_LC)
mean_cap = np.mean(capacity_LC)


threshold = 20.0
congestion_speed = []
congestion_ratio = []
for m in range(N):
    cong_data = np.array(filter(lambda x: x <= threshold, iter_results[m, :]))
    mean = np.mean(cong_data) # this is the average speed for congestion state in one simulation
    ratio = len(cong_data)/3600.0
    congestion_speed.append(mean)
    congestion_ratio.append(ratio)
congestion_speed=np.array(congestion_speed)
congestion_ratio=np.array(congestion_ratio)
np.savetxt('congestion_amplitude',congestion_speed)
np.savetxt('congestion_ratio',congestion_ratio)
average_congestion_speed = np.mean(congestion_speed) # this is to get the average congestion speed across multiple simulations
average_ratio = np.mean(congestion_ratio) # this is to get the average congestion speed across multiple simulations

print('after',N,'times of simulations, the overall average speed is',mean_speed)
print('after',N,'times of simulations, the average capacity is',mean_cap)
print('after',N,'times of simulations, the average standard deviation of speed is',mean_std)
print('after',N,'times of simulations, the average congestion speed is',average_congestion_speed)
print('after',N,'times of simulations, the average congestion ratio is',average_ratio)


plt.figure()
sns.distplot(capacity_LC)
plt.title('the capacity for each iteration')
plt.xlabel('capacity (veh/h)')
plt.ylabel('frequency')


plt.figure()
sns.distplot(average_speed)
plt.title('the average speed for each iterations')
plt.xlabel('speed (mi/h)')
plt.ylabel('frequency')


plt.figure()
sns.distplot(std)
plt.title('the std of speed for each iterations')
plt.xlabel('speed (mi/h)')
plt.ylabel('frequency')


plt.figure()
sns.distplot(congestion_speed)
plt.title('the average congestion speed for each iterations')
plt.xlabel('speed (mi/h)')
plt.ylabel('frequency')



# plt.xlabel('time (seconds)')
# plt.ylabel('speed (mi/h)')
# real_speed = np.loadtxt('processed_speed')
# sns.lineplot(data=real_speed, alpha=0.7)
# the congestion is defined with threshold value of 20
# plt.title('speed data')
plt.figure()
sns.distplot(congestion_ratio)
plt.title('the congestion ratio for each iterations')
plt.xlabel('congestion ratio')
plt.ylabel('frequency')
plt.show()