import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import math
u = 75.0
delta_x = 55.0
w = 15.0
p_exit = 0.8
tau = 3.0
threshold_speed = 5.0 # larger threshold means greater impact on congestion


DLC_value = [1,2,3,4,5]
MLC_value = []

def mlc_desire(x,V): # pi is related to origin lane location, as well as the the target lane speed
    tau = 0.5 # tau is the time for a smooth MLC
    # note that pi has the unit of 1/s
    return np.exp(-1.0*x/(632+5.94*V))/tau

def dlc_desire(current_speed, target_speed):
    tau = 3.0
    delta_v = max(0, target_speed-current_speed)
    return delta_v/u/tau


def linear_accel(v): # accel is in ft/s
    # input v should be in km/h
    return 3.4*(1-v/123.8)
# we should assume the exiting proportion to be 1.0
# actually it is a parameter calibrated by the desire function,
# it should not be an independent variable
# def exit_prob(e0, loc): # the proportion of exiting vehicles
#     # x = np.linspace(0, 4000, 500)
#     # cumu = np.loadtxt('remain_prob')
#     # # note that e0 represents the proportion of exiting vehicles at 4000 upstream of the gore
#     # remain_prob = (1 - cumu) * e0
#     # f = interp1d(x, remain_prob)
#     return f(loc)
    # it returns the proportion of exiting vehicles


def supply(Q,w,kj,k): # downstream supply
    return  min(Q, w*(kj-k))

def demand(Q,u,k): # upstream demand
    return min(Q, k * u)

def Flux(u, w, kj, k, kd, cap_u, cap_d):
    return min(supply(cap_d, w, kj, kd), demand(cap_u, u, k)) # the flow from current cell to downstream cell

def den_to_v(den, u, w, kj): # unit is mi/hr
    kc = w/(u+w)*kj
    if den<=kc:
        return u
    else:
        q = w*max((kj-den),0.0)
        return q/den

def speed_2_cap(speed,w,kj): # speed unit is mi/h
    return w * speed * kj / (w + speed)

def cap_to_density(q,u,w,kj):
    Q = w * u * kj / (w + u)
    return max(kj-Q/w,kj-q/w)

def dlc_desire(v,v_target,u,tau=3.0): # tau is from Laval 2006
    return max(0,v_target-v)/u/tau


## to implement the impact of DLC and MLC, the key is to use the capacity vector
# every time step, we not only update density vector, also update the capacity vector for the sending function
# because the DLC can be modeled as moving bottleneck, MLC as lane drop
# they can both be modeled by constrained sending function

def Godunov(u, w, kj, delta_t, from_x, to_x, begin_t, end_t, downstream_cap, upstream_dens):
    '''implements the Godunov scheme

    ##66, it's better to create a list of list, not multi-dimension matrix
    '''
    Q = w * u * kj / (w + u)
    kc = kj * w / (u + w)
    T1 = begin_t.tm_hour*3600+begin_t.tm_min*60+begin_t.tm_sec
    T2 = end_t.tm_hour*3600+end_t.tm_min*60+end_t.tm_sec
    T = T2 - T1
    dx = delta_t * u*5280.0/3600
    iL = int((from_x-to_x) / dx) # number of road cells
    iT = int(T / delta_t) # number of time cells

    k0 = [[],[],[]] # create a list of three lists, vacant lists
    ##66. initialize the initial density for three lanes, assign upstream density
    for i in range(3):
        k0[i] = [0.0]*iL
        k0[i][0] = upstream_dens[i][0]

    # for i in range(0, iL):
    #    k0.append(initial)
    k0_v = [[0.0]*iL for i in range(3)] # this is the speed list
    for i in range(3):
        k0_v[i][0] = den_to_v(upstream_dens[i][0], u, w, kj)



    sol = [[],[],[]]
    sol_v = [[],[],[]] # speed solution
    # sol is a list of three matrix, with (T*iL) size
    # when the time goes on, we keep appending new columns of data to the three matrix in the sol list
    for lane in range(3):
        sol[lane].append(k0[lane])
        sol_v[lane].append(k0_v[lane])


    # for every time step, we update the density of each road cell, from upstream cell to downstream cell
    # k is the list to denote density of road cells for current time, thus we keep appending new road cell density to
    # list k, until reaching the downstream cell
    k = [[],[],[]]
    # k_v is the companion list to record speed corresponding with density
    k_v = [[],[],[]]

    # k0_v and k0 are used as the old k_v and old k for each update

    # add a new road cell list, mark MLC positions and set their sending capacity to be zero
    # we have sending capacity list here, so just generate MLC particles and update the sending capacity list

    # by default, the sending and receiving capacity is critical capacity
    # then we modify it in every time step if we find DLC or MLC
    default_cap = Q * np.ones(iL) # for every time step, the capacity is a list of road cells
    sending_cap0 = Q*np.ones(iL)
    sending_cap1 = Q*np.ones(iL)
    sending_cap2 = Q*np.ones(iL)
    receiving_cap0 = Q*np.ones(iL)
    receiving_cap1 = Q*np.ones(iL)
    receiving_cap2 = Q*np.ones(iL)

    # we create a new time-space matrix to show the occurrence and duration of generated MLC particles
    MLC56=[1*np.zeros(iL) for i in range(iT-1)]
    DLC54=[1*np.zeros(iL) for i in range(iT-1)]
    DLC45=[1*np.zeros(iL) for i in range(iT-1)]
    DLC_lst = [] # create a DLC list to track DLC trajectories


    ############## over time, we process three lane simutaneously #############################
    for i in range(0, iT-1):
    ################## let's deal with lane 4 with index 0, let's generate DLC from lane 4 to lane 5
        # note that if DLC or MLC is activated, we should change sending_cap here
        receiving_cap0[-1] = downstream_cap[0][i]
        k[0] = [upstream_dens[0][i+1]]
        k_v[0] = [den_to_v(upstream_dens[0][i+1], u,w,kj)]

        # lc45 = np.zeros(iL)  # desired lc flow rate
        # phi45 = np.zeros(iL)  # real MLC flow rate
        # flow45 = np.zeros(iL) #LC flow from 4 to 5
        for j in range(1, iL-1):
            s = k0[0][j] + 1/u * (Flux(u, w, kj, k0[0][j-1], k0[0][j], sending_cap0[j-1], receiving_cap0[j]) -
                                     Flux(u, w, kj, k0[0][j], k0[0][j+1], sending_cap0[j], receiving_cap0[j+1]))
            if s<0:
                s = 0
            k[0].append(s)
            k_v[0].append(den_to_v(s, u, w, kj))
            # lc45[j] = demand(Q, u, k0[0][j]) / 3600.0 * dlc_desire(den_to_v(k0[0][j], u, w, kj),
            #                                                                       den_to_v(k0[1][j], u, w, kj),
            #                                                                       u) / delta_x  # dlc rate, unit is veh/s/ft
            # phi45[j] = lc45[j] # the real lc flow (veh/s/ft)
            # dlc45_num = np.random.poisson(phi45[j] * delta_x * delta_t)
            # if dlc45_num>=1:
            #     flow45[j] = dlc45_num/tau*3600
            #     DLC45[i][j] = 1
            # if dlc from 4 to 5 is generated, add lc flow 45 to the update scheme of lane 5

    # for the last cell, aka downstream cell, put the corresponding density
        downstream_cell_density = cap_to_density(downstream_cap[0][i],u,w,kj)
        k[0].append(downstream_cell_density)
        k_v[0].append(den_to_v(downstream_cell_density,u,w,kj))

        # now update, save result of current step to previous step and then move forward
        k0_v[0] = k_v[0]
        k0[0] = k[0]
    ##  every time step, after processing lane 4, we should reset its capacity to default
        sending_cap0 = Q * np.ones(iL)
        receiving_cap0[:-1] = Q * np.ones(iL - 1)


    ############ lane 5, it has 2 kinds of lc flow, DLC to lane4 and MLC to lane6

        receiving_cap1[-1] = downstream_cap[1][i]
        k[1] = [upstream_dens[1][i+1]]
        k_v[1] = [den_to_v(upstream_dens[1][i+1], u,w,kj)]
        desire_mlc = np.zeros(iL) # desire vector
        lc56 = np.zeros(iL) # desired lc flow rate
        phi56 = np.zeros(iL) # real MLC flow rate
        lc54 = np.zeros(iL)
        phi54 = np.zeros(iL) # real DLC flow rate
        gamma56 = np.ones(iL)
        gamma54 = np.ones(iL)
        for j in range(1, iL-1):
            s = k0[1][j] + 1/u * (Flux(u, w, kj, k0[1][j-1], k0[1][j], sending_cap1[j-1], receiving_cap1[j]) -
                                     Flux(u, w, kj, k0[1][j], k0[1][j+1], sending_cap1[j], receiving_cap1[j+1]))
            if s<0:
                s = 0
            # if np.random.poisson(delta_t*delta_x*mlc_desire((iL-j)*delta_x),den_to_v(k0[2][j],u,w,kj))>=1:
            k[1].append(s)
            k_v[1].append(den_to_v(s, u, w, kj))
            desire_mlc[j] = mlc_desire(3800-j*delta_x,k0_v[2][j]) # k0_v[2][j] is the target lane speed
            through_flow = demand(Q,u,k0[2][j])
            lc56[j] = p_exit* demand(Q,u,k0[1][j])/3600.0*desire_mlc[j]/delta_x # mlc rate, unit is veh/s/ft
            lc54[j] = (1-p_exit) * demand(Q,u,k0[1][j])/3600.0*dlc_desire(den_to_v(k0[1][j],u,w,kj),den_to_v(k0[0][j],u,w,kj),u)/delta_x # dlc rate, unit is veh/s/ft
            lc_demand = lc56[j]*3600*delta_x
            downstream_supply = supply(Q,w,kj,k0[2][j+1])
            gamma56[j] = min(1.0,downstream_supply/(through_flow+lc_demand))
            gamma54[j] =min(1.0,supply(Q,w,kj,k0[0][j+1])/(demand(Q,u,k0[0][j])+lc54[j]*3600*delta_x))
            phi56[j] = lc56[j]*gamma56[j] # the real lc flow (veh/s/ft)
            phi54[j] = lc54[j]*gamma54[j] # the real lc flow (veh/s/ft)

            ######### deal with MLC, set the sending capacity of MLC cell on lane 5 to zero
            # print('the expected # of mlc in this timestep is ', str(phi56[j]*delta_x*delta_t))
            # print('for road cell' + str(j), ' at time step ',str(i),' the generated # of MLC particles', np.random.poisson(phi56[j]*delta_x*delta_t)) # the unit is correct here
            MLC = [] # MLC list is used to record the position of MLC vehicles on lane 5
            if np.random.poisson(phi56[j]*delta_x*delta_t)>=1:
                # print('MLC generated, at time step '+str(i)+' and road cell '+str(j))
                MLC.append(j)
                sending_cap1[j] = 0.0
            if sending_cap1[j] == 0.0: # after MLC occurs, and before it is finally discharged, mark it to one
                MLC56[i][j] = 1
            for count, cell in enumerate(MLC): # loop over the MLC list, check whether target lane is moving, if so, release it
                if den_to_v(k0[2][cell], u, w, kj) >= threshold_speed:
                    sending_cap1[cell] = Q
                    MLC.remove(cell)


            ######### deal with DLC, set the sending/receiving capacity of DLC cell on target lane 4 to zero
            # first, generate new DLC vehicles and append it to the DLC info list
            if np.random.poisson(phi54[j] * delta_x * delta_t) >= 1:
                # print('DLC generated at cell position '+str(j)+ ' at time step '+str(i))
                DLC_lst.append([j*delta_x, k0_v[1][j]*1.47])
                # append a dlc vehicle information(position, speed) speed is ft/s
                # track every DLC with linear accel model, set capacity to zero
            if len(DLC_lst)>0: # make sure DLC list is not vacant
                for d in range(len(DLC_lst)-1): # note that cell_position is in a loop
                    # for each loop, cell_position is the cell position of DLC vehicle d
                    current_cell_position = int(math.floor(DLC_lst[d][0]/delta_x))
                    # if DLC speed is lower than downstream speed, accelerate
                    if current_cell_position <= iL-2 and DLC_lst[d][1] < 1.47*k0_v[0][current_cell_position+1]:
                        # first implement accel, and update speed,
                        updated_dlc_speed = min(DLC_lst[d][1]+3.28*delta_t*linear_accel(1.10*DLC_lst[d][1]),1.47*k_v[0][current_cell_position+1])
                        # updated_dlc_speed = DLC_lst[d][1] + 3.28*delta_t*linear_accel(1.10*DLC_lst[d][1])
                        print('not catching up downstream speed '+ str(k_v[0][current_cell_position+1])+', will accelerate to '+ str(0.681*updated_dlc_speed))
                        DLC_lst[d][1] = updated_dlc_speed # speed in ft/s
                    # then its next position would be
                        DLC_lst[d][0] = DLC_lst[d][0]+DLC_lst[d][1]*delta_t
                        next_cell_position = int(math.floor(DLC_lst[d][0]/delta_x))
                        sending_cap0[next_cell_position] = 0.0 # we should set constraint to next time step
                        DLC54[i][next_cell_position] = 1
                ######################################################################################

        # for the last cell, aka downstream cell, put the corresponding density
        downstream_cell_density = cap_to_density(downstream_cap[1][i],u,w,kj)
        k[1].append(downstream_cell_density)
        k_v[1].append(den_to_v(downstream_cell_density,u,w,kj))

        ## generate MLC particles and update the constraint on sending_capacity model

        k0_v[1] = k_v[1]
        k0[1] = k[1]

    # starting processing lane 6 (index2)
        receiving_cap2[-1] = downstream_cap[2][i]
        k[2] = [upstream_dens[2][i+1]]
        k_v[2] = [den_to_v(upstream_dens[2][i+1], u,w,kj)]
        for j in range(1, iL-1):
            s = k0[2][j] + 1/u * (Flux(u, w, kj, k0[2][j-1], k0[2][j], sending_cap2[j-1], receiving_cap2[j]) -
                                     Flux(u, w, kj, k0[2][j], k0[2][j+1], sending_cap2[j-1], receiving_cap2[j+1]))
            if s<0:
                s = 0
            k[2].append(s)
            k_v[2].append(den_to_v(s, u, w, kj))
        # for the last cell, aka downstream cell, put the corresponding density
        downstream_cell_density = cap_to_density(downstream_cap[2][i],u,w,kj)
        k[2].append(downstream_cell_density)
        k_v[2].append(den_to_v(downstream_cell_density,u,w,kj))

        # now update, save result of current step to previous step and then move forward
        k0_v[2] = k_v[2]
        k0[2] = k[2]

        ## save time-space speed matrix to sol and sol_v list
        for lane in range(3):
            sol[lane].append(k0[lane])
            sol_v[lane].append(k0_v[lane])

    print('multi-lane simulation over time T is done')
    # np array here transform two diamension list to a matrix for further plotting



    ext = [0, T,from_x, to_x]
    asp = (ext[1]-ext[0])*1.0/(from_x-to_x)/3 # this is to compute the y-x aixs display ratio

    # plt.figure()
    plt.subplots(3,1,sharex=True)
    plt.subplot(311)
    plt.title('the simulated speed for lane 4')
    # plt.title('simulated speed for lane4')
    plt.imshow(np.transpose(sol_v[0])[::-1], aspect = asp, extent=ext, cmap = 'jet_r')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Location(ft)')
    plt.colorbar()

    plt.subplot(312)
    plt.title('the simulated speed for lane 5')
    # plt.title('simulated speed for lane5')
    plt.imshow(np.transpose(sol_v[1])[::-1], aspect = asp, extent=ext, cmap = 'jet_r')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Location(ft)')
    plt.colorbar()

    plt.subplot(313)
    # plt.title('simulated speed for lane6')
    plt.title('the simulated speed for lane 6')
    plt.imshow(np.transpose(sol_v[2])[::-1], aspect = asp, extent=ext, cmap = 'jet_r')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Location(ft)')
    plt.colorbar()


    plt.figure()
    plt.title('the simulated speed for lane 4')
    plt.imshow(np.transpose(sol_v[0])[::-1], aspect = asp, extent=ext, cmap = 'jet_r')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Location(ft)')
    plt.colorbar()


    plt.figure()
    plt.title('occurrence and duration of MLCs')
    plt.imshow(np.transpose(MLC56)[::-1], aspect = asp, extent=ext, cmap = 'Reds')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Location(ft)')

    plt.figure()
    plt.title('trajectories DLCs from lane 5 to lane 4')
    plt.imshow(np.transpose(DLC54)[::-1], aspect = asp, extent=ext, cmap = 'Greens')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Location(ft)')

    # plt.figure()
    # plt.title('trajectories DLCs from lane 4 to lane 5')
    # plt.imshow(np.transpose(DLC45)[::-1], aspect = asp, extent=ext, cmap = 'Greys')
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('Location(ft)')

    plt.show()
    print('MHLC CTM simulation done')
    return sol_v,MLC56,DLC45


def evaluate_model(time_list, loc_list, speed_list, model, delta_t, delta_x, from_x, to_x):

    ABS = []
    MS = []
    predicted = []
    observed = []
    for i in range(len(time_list)):
        n_t = int(time_list[i]/delta_t)
        if n_t==model.shape[0]:
            n_t = n_t - 1
        n_x = int((from_x-loc_list[i])/delta_x)
        if n_x==model.shape[1]:
            n_x =n_x - 1
        if n_x != model.shape[1]-1 and n_x != 0 and n_t != 0:
            MS.append(abs(speed_list[i]-model[n_t, n_x]))
            predicted.append(model[n_t, n_x])
            observed.append(speed_list[i])
        ABS.append(abs(speed_list[i]-model[n_t, n_x]))
    np.savetxt('predicted', np.array(predicted))
    np.savetxt('observed', np.array(observed))
    plt.figure()
    plt.scatter(time_list, loc_list, c=ABS, s=2, lw=0, cmap='jet')
    plt.colorbar()
    plt.xlim(0, 3600)
    plt.ylim(from_x, to_x)
    plt.show()

    sq_error = np.sqrt(sum([aserror**2 for aserror in MS])/len(MS))
    return sq_error


if __name__ == '__main__':
    print('please give input')
    # Godunov(100.0, 20.0, 150.0, 0.5, 2.0, 4.0/60)
