import numpy as np
import os
import matplotlib.pyplot as plt
import time




def plot_trajectory(l,begin_t,end_t,from_x,to_x):
    print('load real trajectory data of lane ' + str(l)+' for model evaluation')
    dir = os.getcwd() + '\lane'+str(l)
    time_list = np.loadtxt(dir+'\\time_list')
    loc_list = np.loadtxt(dir+'\loc_list')
    speed_list = np.loadtxt(dir+'\speed_list')
    T1 = begin_t.tm_hour * 3600 + begin_t.tm_min * 60 + begin_t.tm_sec
    T2 = end_t.tm_hour * 3600 + end_t.tm_min * 60 + end_t.tm_sec
    T = T2 - T1
    ext = [0, T, from_x, to_x]
    asp = (ext[1] - ext[0]) * 1.0 / (from_x - to_x) / 3
    plt.figure()
    plt.title('trajectory data of lane '+ str(l))
    plt.xlabel('Time (seconds)')
    plt.ylabel('Location(ft)')
    plt.scatter(time_list, loc_list, c=speed_list, s=2, lw=0, cmap='jet_r') # show the trajectory data
    plt.colorbar()
    plt.gca().invert_yaxis()
    # plt.axis([0, T, from_x, to_x, 0])
    plt.show()


if __name__ == '__main__':
    begin_t = time.strptime('6:35:00', '%H:%M:%S')
    end_t = time.strptime('7:35:00', '%H:%M:%S')
    from_x = 3850
    to_x = 1150
    plot_trajectory(4,begin_t,end_t,from_x,to_x)
