# !/opt/local/bin/python
import time
import numpy as np
import Godunov_mhlc as god
import matplotlib.pyplot as plt
from plotter import*
# fundamental diagram parameters
ff_speed = 75.0
jam_density = 150.0
wave_speed = 15
kc = wave_speed * jam_density / (ff_speed + wave_speed)
# simulation parameters
simu_timestep = 0.5
simu_xstep = simu_timestep / 3600 * ff_speed * 5280



if __name__ == '__main__':

    # modelling time and space domain
    begin_t = time.strptime('6:35:00', '%H:%M:%S')
    end_t = time.strptime('7:35:00', '%H:%M:%S')
    from_x = 3850
    to_x = 1150

    # plot_trajectory(4, begin_t,end_t,from_x,to_x)
    # plot_trajectory(5, begin_t,end_t,from_x,to_x)
    # plot_trajectory(6, begin_t,end_t,from_x,to_x)

    print('load upstream density')
    upstream_den4 = np.loadtxt('.\lane4\upstream_den')
    upstream_den5 = np.loadtxt('.\lane5\upstream_den')
    upstream_den6 = np.loadtxt('.\lane6\upstream_den')


    print('load downstream capacity')
    downstream_cap4 = np.loadtxt('.\lane4\downstream_cap')
    downstream_cap5 = np.loadtxt('.\lane5\downstream_cap')
    downstream_cap6 = np.loadtxt('.\lane6\downstream_cap')

    # print('load lane change inflow')
    # inflow4 = np.loadtxt('.\lane4\inflow')
    # inflow5 = np.loadtxt('.\lane5\inflow')
    # inflow6 = np.loadtxt('.\lane6\inflow')
    #
    # print('load lane change outflow')
    # outflow4 = np.loadtxt('.\lane4\outflow')
    # outflow5 = np.loadtxt('.\lane5\outflow')
    # outflow6 = np.loadtxt('.\lane6\outflow')


    # print('load capacity model after consideration of DLC')
    # cap = np.loadtxt('capacity')

    print('start godunov simulation for three lanes simultaneoulsy')


    alpha = 0.0
    beta = 0.0

    alpha4 = 0.0
    alpha5 = 0.0
    alpha6 = 0.0

    beta4 = 0.0
    beta5 = 0.0
    beta6 = 0.0
    # if you generate lane changes, data should be discarded to avoid duplicates


    ##66, change the input from one element for single lane to a list of three elements for three lane

    # inflow = [inflow4, inflow5, inflow6]
    # outflow = [outflow4, outflow5, outflow6]
    downstream_cap = [downstream_cap4,downstream_cap5,downstream_cap6]
    upstream_den = [upstream_den4, upstream_den5, upstream_den6]
    # alpha = [alpha4, alpha5, alpha6]
    # beta = [beta4, beta5, beta6]

    godmodel = god.Godunov(ff_speed, wave_speed,
                           jam_density, simu_timestep,
                           from_x, to_x, begin_t, end_t,
                           downstream_cap, upstream_den)

    # T1 = begin_t.tm_hour*3600+begin_t.tm_min*60+begin_t.tm_sec
    # T2 = end_t.tm_hour*3600+end_t.tm_min*60+end_t.tm_sec
    # T = T2 - T1
    # ext = [0, T, from_x, to_x]
    # asp = (ext[1]-ext[0])*1.0/(from_x-to_x)/3
    # plt.figure(9)
    # plt.subplot(211)
    # plt.imshow(np.transpose(godmodel)[::-1], aspect=asp,extent=ext, cmap='jet_r') # show the simulatioon result
    # # The bounding box in data coordinates that the image will fill. The image is stretched individually along x and y to fill the box.
    # plt.colorbar()
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('Location(ft)') trajecoty data
    # plt.colorbar()
    # plt.subplot(212)
    # plt.scatter(time_list, loc_list, c=speed_list, s=2, lw=0, cmap='jet_r') # show the
    # plt.show()
    print('job done')


