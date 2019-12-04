import numpy as np

# mlc_mean = 5
# waiting_mean = 18
# randomList = []
# randomList.append((3000, 21836, 40, 1, 0))
# randomList.append((2817, 25620, 20, 1, 0))  # note that we can change the speed of initial DLC speed
# Num_mlc = np.random.poisson(mlc_mean)
# for i in range(Num_mlc):
#     loc = 2560
#     time = 23700 + int((27300 - 23700) * np.random.rand())
#     speed = 20
#     type = 0
#     waiting = np.random.poisson(waiting_mean)
#     randomList.append((loc, time, speed, type, waiting))  # note that we can change the speed of initial DLC speed
# print('the disturb list is', randomList)

import iteration
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()


capacity_lc = np.loadtxt('capacity_simu')
capacity_lc = capacity_lc-560
average_speed = np.loadtxt('average_speed')
standard_d = np.loadtxt('standard_deviation')
congestion_speed = np.loadtxt('congestion_amplitude')
congestion_speed = congestion_speed
congestion_ratio = np.loadtxt('congestion_ratio')
congestion_ratio = congestion_ratio + 0.01

#
# # plt.figure(figsize=(12,3))
# # plt.subplot(1,4,1)
# average_speed = np.loadtxt('average_speed')
# # sns.distplot(average_speed)
# # plt.xlabel('average speed (mi/h)')
# #
# #
# # plt.subplot(1,4,2)
# average_speed = np.loadtxt('average_speed')
# # sns.distplot(standard_d)
# # plt.xlabel('s.t.d. of speed (mi/h)')
# #
# #
# # plt.subplot(1,4,3)
# congestion_speed = np.loadtxt('congestion_amplitude')
# # sns.distplot(congestion_speed)
# # plt.xlabel('congestion amplitude (mi/h)')
# # # plt.ylabel('frequency')
# #
# #
# # # plt.xlabel('time (seconds)')
# # # plt.ylabel('speed (mi/h)')
# # # real_speed = np.loadtxt('processed_speed')
# # # sns.lineplot(data=real_speed, alpha=0.7)
# # # the congestion is defined with threshold value of 20
# # # plt.title('speed data')
# # plt.subplot(1,4,4)
# congestion_ratio = np.loadtxt('congestion_ratio')
# # congestion_ratio = congestion_ratio + 0.01
# # sns.distplot(congestion_ratio)
# # plt.xlabel('congestion ratio')
# # # plt.ylabel('frequency')
# #
# # plt.show()

length = 6
width = 4.5


plt.subplot(2,2,1)
sns.distplot(capacity_lc)
# plt.title('The capacity in iterative simulations')
plt.scatter(1540,0,s=20,c='r')
# plt.text(1540,-0.0015,'Data')
plt.scatter(1614,0,s=30,c='g')
# plt.text(1614,-0.0015,'CTM_para')
plt.xlabel('capacity (veh/h)')
plt.ylabel('probability density')


plt.subplot(2,2,2)
sns.distplot(standard_d)
# plt.title('The s.t.d of speed in iterative simulations')
plt.scatter(12.27,0,s=30,c='r')
# plt.text(12.27,-0.05,'Data')
plt.scatter(9.86,0,s=30,c='g')
# plt.text(9.86,-0.05,'CTM_para')
plt.xlabel('s.t.d. of speed (mi/h)')
plt.ylabel('probability density')

plt.subplot(2,2,3)
sns.distplot(congestion_speed)
# plt.title('The congestion amplitude in iterative simulations')
plt.scatter(11.491,0,s=30,c='r')
# plt.text(11.491,-0.05,'Data')
plt.scatter(13.116,0,s=30,c='g')
# plt.text(13.116,-0.05,'CTM_para')
plt.xlabel('congestion amplitude (mi/h)')
plt.ylabel('probability density')


plt.subplot(2,2,4)
sns.distplot(congestion_ratio)
# plt.title('The congestion ratio in iterative simulations')
plt.scatter(0.0783,0,s=30,c='r')
# plt.text(0.0783,-2,'Data')
plt.scatter(0.0202,0,s=30,c='g')
# plt.text(0.0202,-2,'CTM_para')
plt.xlabel('congestion ratio')
plt.ylabel('probability density')


# plt.figure(figsize=(length,width))
# sns.distplot(capacity_lc)
# # plt.title('the capacity for each iterations')
# plt.scatter(1540,0,s=20,c='r')
# plt.text(1540,-0.0015,'Data')
# plt.scatter(1614,0,s=30,c='g')
# plt.text(1614,-0.0015,'CTM_para')
# plt.xlabel('capacity (veh/h)')
# plt.ylabel('frequency')


# plt.figure(figsize=(length,width))
# sns.distplot(average_speed)
# # plt.title('the average speed for each iterations')
# plt.scatter(39,0,s=20,c='r')
# plt.text(39,-0.05,'Data')
# # plt.xlabel('speed (mi/h)')
# # plt.ylabel('frequency')
#
#
# # plt.figure(figsize=(length,width))
# # sns.distplot(standard_d)
# # # plt.title('the s.t.d of speed for each iteration')
# # plt.scatter(12.27,0,s=30,c='r')
# # plt.text(12.27,-0.05,'Data')
# # plt.scatter(9.86,0,s=30,c='g')
# # plt.text(9.86,-0.05,'CTM_para')
# # plt.xlabel('s.t.d. of speed (mi/h)')
# # # plt.ylabel('frequency')
#
#
# plt.figure(figsize=(length,width))
# sns.distplot(congestion_speed)
# # plt.title('the average congestion speed for each iterations')
# plt.scatter(11.491,0,s=30,c='r')
# plt.text(11.491,-0.05,'Data')
# plt.scatter(13.116,0,s=30,c='g')
# plt.text(13.116,-0.05,'CTM_para')
# plt.xlabel('congestion amplitude (mi/h)')
# # plt.ylabel('frequency')
#
#
#
# # plt.xlabel('time (seconds)')
# # plt.ylabel('speed (mi/h)')
# # real_speed = np.loadtxt('processed_speed')
# # sns.lineplot(data=real_speed, alpha=0.7)
# # the congestion is defined with threshold value of 20
# # plt.title('speed data')
# plt.figure(figsize=(length,width))
# sns.distplot(congestion_ratio)
# # plt.title('the congestion ratio for each iterations')
# plt.scatter(0.0783,0,s=30,c='r')
# plt.text(0.0783,-2,'Data')
# plt.scatter(0.0202,0,s=30,c='g')
# plt.text(0.0202,-2,'CTM_para')
# plt.xlabel('congestion ratio')
# plt.ylabel('frequency')
plt.show()



print('the average mean speed after 1000 times is',np.mean(average_speed))
print('the average std speed after 1000 times is',np.mean(standard_d))
print('the average congestion speed after 1000 times is',np.mean(congestion_speed))
print('the average congestion ratio after 1000 times is',np.mean(congestion_ratio))

