# !/opt/local/bin/python
import time
import numpy as np
import godunov_disturb as god
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def generate_lc(mlc_mean, waiting_mean):
    randomList = []
    randomList.append((3000, 21836, 40, 1, 0))
    randomList.append((2817, 25620, 20, 1, 0))  # note that we can change the speed of initial DLC speed
    Num_mlc = np.random.poisson(mlc_mean)
    for i in range(Num_mlc):
        loc = 2560
        time = 23700 + int((27300 - 23700) * np.random.rand())
        speed = 20
        type = 0
        waiting = np.random.poisson(waiting_mean)
        randomList.append((loc, time, speed, type, waiting))
    return randomList


if __name__ == '__main__':

    # modelling time and space domain
    begin_t = time.strptime('6:35:00', '%H:%M:%S')
    end_t = time.strptime('7:35:00', '%H:%M:%S')
    from_x = 3850 # in unit of ft
    to_x = 1150

    #############################################################

    # randomList = []
    # randomList.append((3000,21836,40,1,0))
    # randomList.append((2817,25620,20,1,0)) # note that we can change the speed of initial DLC speed
    # Num_mlc = np.random.poisson(mlc_mean)
    # for i in range(Num_mlc):
    #     loc = 2560
    #     time = 23700+int((27300-23700)*np.random.rand())
    #     speed = 20
    #     type = 0
    #     waiting = np.random.poisson(waiting_mean)
    #     randomList.append((loc,time,speed,type,waiting))  # note that we can change the speed of initial DLC speed
    # print('the disturb list is',randomList)
    #############################################################

    # print 'load trajectory data'
    time_list = np.loadtxt('time_list')
    loc_list = np.loadtxt('loc_list')
    speed_list = np.loadtxt('speed_list')

    # fundamental diagram parameters
    ff_speed = 75.0
    jam_density = 240.0
    wave_speed = 15.0
    kc = wave_speed * jam_density/(ff_speed + wave_speed)
    # simulation parameters
    simu_timestep = 0.5 # note that the simulation delta_t should be a float number, 1.0 not 1
    simu_xstep = simu_timestep/3600*ff_speed*5280

    # print 'load upstream density'
    upstream_den = np.loadtxt('upstream_den')

    # print 'load downstream capacity'
    downstream_cap = np.loadtxt('downstream_cap')

    # print 'load lane change inflow'
    inflow = np.loadtxt('inflow')

    # print 'load lane change outflow'
    outflow = np.loadtxt('outflow')

    # print 'load capacity model with DLC'

    cap = np.loadtxt('capacity') # the capacity was pre-calculated and pre-defined here.

    # print 'start godunov simulation'

    alpha = 0.0
    beta = 0.0
    print 'alpha=',alpha
    print 'beta=',beta

    lag=2
    # print 'DLC lag=',lag/2, 'seconds'

    mlc_mean = 5
    waiting_mean = 18

    iter_results = []

    plt.figure()

    T1 = begin_t.tm_hour*3600+begin_t.tm_min*60+begin_t.tm_sec
    T2 = end_t.tm_hour*3600+end_t.tm_min*60+end_t.tm_sec
    T = T2 - T1
    ext = [0, T, from_x, to_x]
    asp = (ext[1]-ext[0])*1.0/(from_x-to_x)/3

    T1 = begin_t.tm_hour*3600+begin_t.tm_min*60+begin_t.tm_sec
    T2 = end_t.tm_hour*3600+end_t.tm_min*60+end_t.tm_sec
    T = T2 - T1
    ext = [0, T, from_x, to_x]
    asp = (ext[1]-ext[0])*1.0/(from_x-to_x)/3


    for i in range(1):
        print('doing the',i,'th simulation')
        randomList = generate_lc(mlc_mean,waiting_mean)
        print('the dmlc list is',randomList)
        # print 'conduct the Godunov Simulation'
        godmodel_speed = god.Godunov(ff_speed, wave_speed,
                           jam_density, simu_timestep,randomList,
                           from_x, to_x, begin_t, end_t,
                           inflow, outflow,
                           cap, upstream_den, alpha, beta,lag)

        plt.imshow(np.transpose(godmodel_speed)[::-1], aspect=asp, extent=ext,
                   cmap='jet_r')  # show the godmodel result, the speed of cell
        plt.colorbar()
        plt.title('Simulation results')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Location(ft)')
        # extract the speed profile at an upstream location
        # up_loc = godmodel_speed[:, 15]
        # revised = []
        # for j in range(len(up_loc)):
        #     if j % 2 == 0: # for 7200 time points, only extract 3600 of them
        #         revised.append(up_loc[j])
        # plt.figure()
        # plt.plot(revised)
        # # add speed profile of ith simulation to the iteration results
        # iter_results.append(revised)
    # iter_results = np.array(iter_results)



    # plt.figure()
    # plt.xlabel('time (seconds)')
    # plt.ylabel('speed (mi/h)')
    # real_speed = np.loadtxt('processed_speed')
    # sns.lineplot(data=real_speed, alpha=0.7)
    # plt.title('speed data')
    # sns.lineplot(data = iter_results[1,:])
    plt.show()

