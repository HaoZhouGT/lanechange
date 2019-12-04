# !/opt/local/bin/python
import time
import numpy as np
import godunov_disturb as god
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # modelling time and space domain
    begin_t = time.strptime('6:35:00', '%H:%M:%S')
    end_t = time.strptime('7:35:00', '%H:%M:%S')
    from_x = 3850 # in unit of ft
    to_x = 1150

    def extract_disturb_record(filename):
        '''extract the disturbance time and location from csvfile

        '''
        rec_list = []
        with open(filename, 'rb') as csvfile:
            for line in csvfile:
                words = line.replace('\r\n', '').split(',')
                # print words
                loc = float(words[0])
                time_obj = time.strptime(words[1], '%H:%M:%S')
                initialspeeed = float(words[2])
                rec_list.append((loc, time_obj,initialspeeed)) # this is the way to add row by row
        return rec_list


    print 'load trajectory data'
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

    print 'load upstream density'
    upstream_den = np.loadtxt('upstream_den')

    print 'load downstream capacity'
    downstream_cap = np.loadtxt('downstream_cap')

    print 'load lane change inflow'
    inflow = np.loadtxt('inflow')

    print 'load lane change outflow'
    outflow = np.loadtxt('outflow')

    print 'load capacity model with DLC'

    cap = np.loadtxt('capacity') # the capacity was pre-calculated and pre-defined here.

    print 'start godunov simulation'

    alpha = 0.0
    beta = 0.02
    print 'alpha=',alpha
    print 'beta=',beta

    lag=20
    print 'DLC lag=',lag



    print 'extract DLC starting location before simulation'
    disturb_rec = extract_disturb_record('dlc_in_speed.csv')

    print 'conduct the Godunov Simulation'
    godmodel_density, godmodel_speed = god.Godunov(ff_speed, wave_speed,
                           jam_density, simu_timestep,disturb_rec,
                           from_x, to_x, begin_t, end_t,
                           inflow, outflow,
                           cap, upstream_den, alpha, beta,lag)

    T1 = begin_t.tm_hour*3600+begin_t.tm_min*60+begin_t.tm_sec
    T2 = end_t.tm_hour*3600+end_t.tm_min*60+end_t.tm_sec
    T = T2 - T1
    ext = [0, T, from_x, to_x]
    asp = (ext[1]-ext[0])*1.0/(from_x-to_x)/3
    plt.subplots(2, 1, sharex='col')
    plt.subplot(211)
    plt.title('speed data')
    plt.scatter(time_list, loc_list, c=speed_list, s=2, lw=0, cmap='jet_r')
    plt.gca().invert_yaxis()
    plt.colorbar()

    plt.subplot(212)
    plt.imshow(np.transpose(godmodel_speed)[::-1], aspect=asp,extent=ext, cmap='jet_r') # show the godmodel result, the speed of cell
    plt.colorbar()
    plt.title('Simulated speed by Godunov Method')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Location(ft)')
    plt.show()

    # ms = god.evaluate_model(time_list, loc_list, speed_list, godmodel_speed,
    #                         simu_timestep, simu_xstep, from_x, to_x)
    # print ms
