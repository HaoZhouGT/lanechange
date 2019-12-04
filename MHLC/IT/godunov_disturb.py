import numpy as np
import matplotlib.pyplot as plt
from math import exp


def dmlcspeed(t,Vc=5.0,V0=30.0):
    Vc=Vc*1.467
    V0 = V0 * 1.467
    v=V0-8.0*t# v should be in unit of ft/s
    if v<0:
        v=0.0
    return v # return the speed v(t) with a constant deceleration rate 8 ft/s^2


def dlcspeed(t,Vc=75.0,V0=10.0,beta=0.06):
    Vc=Vc*1.467
    V0 = V0 * 1.467
    v=Vc-(Vc-V0)*exp(-1*beta*t)# v should be in unit of ft/s
    return v # return the speed v(t) with a linear acceleration model



def F(u, w, kj, ku, kd, cap_u, cap_d):
    supply = lambda Q, w, kj, k: min(Q, w*(kj-k))
    demand = lambda Q, u, k: min(Q, k*u)
    d_u = demand(cap_u, u, ku)
    s_d = supply(cap_d, w, kj, kd)
    return min(d_u, s_d) # calculate the supply of the upstream and the demand of downstream

def transFlow(u, w, kj, ku, kd, cap_u):
    m1 = min (u*ku,w*(kj-kd))
    m2 = min (w*(kj-kd),cap_u)
    return min(m1,m2)


def change_to_sec(time):
    '''change the time object to absolute seconds
    '''
    return time.tm_hour*3600 + time.tm_min*60 + time.tm_sec

def den_to_v(den, v, w, kj):
    kc = w/(v+w)*kj
    if 0<=den<=kc:
        speed = v
    else:
        q = w*(kj-den)
        speed = q/den
    return speed


def initial_cond(x, kj, u, w):
    kc = kj * w / (u + w)
    if x <= 1.5 and x >0.8:
        return kj
    elif x > 1.5:
        return kc
    else:
        return kc/2
        # kc/2 could be an arbitary value provided.

def mindisturb(disturbrec,begin_t,delta_t):
    disturbing=[]
    dlc=[]
    dmlc=[]
    for rec in disturbrec:
        tt = change_to_sec(rec[1])
        n = int((tt - change_to_sec(begin_t)) / delta_t)
        disturbing.append(n)
    for rec in disturbrec:
        tt = change_to_sec(rec[1])
        n = int((tt - change_to_sec(begin_t)) / delta_t)
        if rec[3]==1:
            dlc.append(n)
        if rec[3]==0:
            dmlc.append(n)
    return disturbing,dlc,dmlc
# marker 1 means DLC and 0 means DMLC
# this function is to derive the time list for DLC vehicles


def Godunov(u, w, kj, delta_t, disturb_rec, from_x, to_x,
            begin_t, end_t, inflow, outflow,
            cap, upstream_dens,
            alpha=1.0, beta=1.0,lag=10):
    Q = w * u * kj / (w + u)
    kc = kj * w / (u + w)
    T1 = begin_t.tm_hour * 3600 + begin_t.tm_min * 60 + begin_t.tm_sec
    T2 = end_t.tm_hour * 3600 + end_t.tm_min * 60 + end_t.tm_sec
    T = T2 - T1
    dx = delta_t * u * 5280.0 / 3600
    iL = int((from_x - to_x) / dx)
    iT = int(T / delta_t)
    sol = []
    sol_v = []  # speed solution
    # for i in range(0, iL):
    #    k0.append(initial)
    k0_v = [0.0] * iL # this is the first k0_v, at time 0, all density was assumed
    k0_v[0] = den_to_v(upstream_dens[0], u, w, kj)
    # at initial time, all density were assumed zero.
    k0 = [0.0] * iL # k0[] records the density at previous time step and k[] will record the updated value
    k0[0] = upstream_dens[0]

    sol_v.append(k0_v)
    sol.append(k0)

    disturbinglist,dlclist,dmlclist = mindisturb(disturb_rec,begin_t,delta_t) #dlclist is the time list for DMLC vehicles
    t = 0  # type:
    # supply=[]
    print 'initial time step',t
    while t< iT-2:
        if t not in disturbinglist:
            print 'not in DLC, time step of', t
            k = [upstream_dens[t]] # the initial value of k and k_v list, for further append
            k_v = [den_to_v(upstream_dens[t], u, w, kj)]
            for j in range(1, iL - 1): #starting from 1 is to skip the upstream boundary cell
                # if j == 20:
                if j != iL - 2:
                    s = k0[j] + 1 / u * (transFlow(u, w, kj, k0[j - 1], k0[j],cap[j - 1,t]) -
                                         transFlow(u, w, kj, k0[j], k0[j+1],cap[j,t])
                                         + alpha*inflow[j, t] - beta*outflow[j, t])
                else:
                    inflo = F(u, w, kj, k0[j - 1], k0[j], cap[j, t - 1], cap[j, t])
                    # print 'inflow at downstream boundary',inflo
                    ouflo = F(u, w, kj, k0[j], k0[j + 1], cap[j + 1, t - 1], cap[j + 1, t])
                    # print 'outflow to downstream cell',ouflo
                    s = k0[j] + 1 / u * (inflo - ouflo + alpha * inflow[j, t] - beta * outflow[j, t])
                    # print 'density of the cell last to downstream', s
                if s < 0:
                    s = 1.0
                k.append(s)  # k is another list recording the renewed density of all road cells at time t+1
                k_v.append(den_to_v(s, u, w, kj))
            k.append(0) # final value of k and k_v, the downstream value
            k_v.append(den_to_v(0, u, w, kj))
            k0_v = k_v
            k0 = k  # k0 is the density of previous step, now update it with current result
            # print 'density list at time t is',k
            sol.append(k) # append the current density result to the solution
            sol_v.append(k_v)
            t = t + 1  # if i is not the beginning of DLC, it conducts DLC and updates
        else:# it means the timestep t is in the dlclist
            if t in dmlclist:
                print 'begin entering DMLC, and the time step is',t
                # first we would like to find which DLC the time has reached
                n=0
                cell=1
                initialspeed=0
                loc=0
                # variables in the loop cannot be found after it
                for rec in disturb_rec: # this is the record of disturbing vehicles
                    tt = change_to_sec(rec[1])
                    if t==int((tt - change_to_sec(begin_t))/delta_t):
                        loc = rec[0]  # this is the initial location of the DLC vehicle
                        initialspeed = rec[2]
                        waiting = rec[4]
                        cell = int((from_x - loc) / dx) # cell is the location of the DMLC
                        n = int((tt - change_to_sec(begin_t))/delta_t)
                    print'now begin the waiting process'
                    print'the timestep of the DMLC is',n
                flag = 0
                print 'waiting time is',waiting/2
                while flag<=waiting:# initial DLC speed usually lower than downstream traffic speed
                    print'time step is',t
                    volDLC = dmlcspeed((t - n) * delta_t, 0.0, initialspeed) # do it when n==i,initial speed 10 mi/h
                    print 'the speed of DMLC is',volDLC
                    travel = volDLC*delta_t # the travel distance of DLC
                    loc = loc - travel # the location of DLC cell for next step t+1
                    print 'DMLC location is',loc
                        # if cell<int((from_x - loc)/dx):
                        #     k0[cell+1]=k0[cell]
                    cell = int((from_x - loc) / dx) # the cell of DMLC for the next step t+1
                    print 'DMLC is at cell',cell,'at time',t
                    cap[cell, t] = 0.0  # set the capactiy of t+1,cell to zero
                    k=[upstream_dens[t + 1]]
                    k_v = [den_to_v(upstream_dens[t + 1], u, w, kj)]
                            # actually, here the time step i already become i+1, we should do Godunov again
                    for j in range(1, iL - 1):  # from second cell to the upstream of the downstream
                        if j != iL - 2:
                                s = k0[j] + 1 / u * (transFlow(u, w, kj, k0[j - 1], k0[j], cap[j - 1, t]) -
                                                         transFlow(u, w, kj, k0[j], k0[j+1], cap[j, t]) + alpha * inflow[
                                                             j, t] - beta * outflow[j, t])
                        else:
                                inflo = F(u, w, kj, k0[j - 1], k0[j], cap[j, t - 1], cap[j, t])
                                ouflo = F(u, w, kj, k0[j], k0[j + 1], cap[j + 1, t - 1], cap[j + 1, t])
                                s = k0[j] + 1 / u * (inflo - ouflo + alpha * inflow[j, t] - beta * outflow[j, t])
                        if s < 0:
                            s = 1.0
                        k.append(s)  # k is another list recording the renewed density of all road cells at time t+1
                        k_v.append(den_to_v(s, u, w, kj))
                    k[cell+1] = kc
                    k_v[cell]=volDLC
                    k.append(0)  # final value of k and k_v, the downstream value
                    k_v.append(den_to_v(0, u, w, kj))
                    k0_v = k_v
                    k0 = k  # renew the list of density at time t, with the estimated density list t+1
                    sol.append(k)
                    sol_v.append(k_v)
                    print 'the speed of DLC is', volDLC
                        # print 'the density of DLC downstream traffic is', k0[cell + 1]
                    t = t + 1
                    flag+=1
                    # while not, it will end the loop and exit

            if t in dlclist:
                print 'begin entering DLC', t
                for rec in disturb_rec:
                    loc = rec[0]  # this is the initial location of the DLC vehicle
                    tt = change_to_sec(rec[1])
                    initialspeed = rec[2]
                    cell = int((from_x - loc) / dx)
                    n = int((tt - change_to_sec(begin_t)) / delta_t)
                    if n == t:  # this is to pinpoint the DLC start time n
                        print'at first, DLC is lagging for some seconds with very low speed'
                        for mm in range(lag):
                            volDLC = dlcspeed((t - n) * delta_t, 75.0, initialspeed,0.06)  # do it when n==i,initial speed 10 mi/h
                            loc = loc - volDLC * 1.467 * delta_t  # the update of loc, cell only moves one timestep, speed unit should be foot/seconds
                            # if cell<int((from_x - loc)/dx):
                            #     k0[cell+1]=k0[cell]
                            cell = int((from_x - loc) / dx)
                            print 'the DLC is in cell', cell, 'at timestep', t
                            cap[cell, t] = 0.0  # set the capacity of the initial cell to zero, then do the Godunov
                            k = [upstream_dens[t + 1]]  # the inital value of k and k_v list, for further append
                            k_v = [den_to_v(upstream_dens[t + 1], u, w,
                                            kj)]  # prepare to do Godunov and record density result
                            for j in range(1, iL - 1):  # from second cell to the upstream of the downstream
                                if j != iL - 2:
                                    # print 'the inflow from upstream cell is',transFlow(u, w, kj, k0[j - 1], k0[j], cap[j - 1, t])
                                    # print 'the outflow to upstream cell is',transFlow(u, w, kj, k0[j], k0[j + 1], cap[j, t])
                                    s = k0[j] + 1 / u * (transFlow(u, w, kj, k0[j - 1], k0[j], cap[j - 1, t]) -
                                                         transFlow(u, w, kj, k0[j], k0[j + 1], cap[j, t])
                                                         + alpha * inflow[j, t] - beta * outflow[j, t])
                                else:
                                    inflo = F(u, w, kj, k0[j - 1], k0[j], cap[j, t - 1], cap[j, t])
                                    ouflo = F(u, w, kj, k0[j], k0[j + 1], cap[j + 1, t - 1], cap[j + 1, t])
                                    s = k0[j] + 1 / u * (inflo - ouflo + alpha * inflow[j, t] - beta * outflow[j, t])
                                if s < 0:
                                    s = 1.0
                                k.append(s)  # save the density result for current step
                                k_v.append(den_to_v(s, u, w, kj))
                            k[cell + 1] = kc
                            k_v[cell] = volDLC  # the DLC cell was assumed with a constant speed
                            k.append(0)  # final value of k and k_v, the downstream value
                            k_v.append(den_to_v(0, u, w, kj))
                            k0_v = k_v  # k0 and k0_v are used for recording density at previous time
                            k0 = k  # after doing Godunov, k0 and k0_v should be updated
                            sol.append(k)
                            sol_v.append(k_v)
                            print 'the speed of DLC is', volDLC
                            # print 'the density of DLC downstream traffic is', k0[cell + 1]
                            # print 'the initial speed of DLC downstream traffic is', k0_v[cell + 1]
                            t = t + 1  # means the DLC start to move forward
                        print'now begin DLC linear accel trajectory'
                        while cell < iL - 3 and loc > to_x and volDLC < k0_v[cell + 1]-15:  # initial DLC speed usually lower than downstream traffic speed
                            print'time step is', t
                            volDLC = dlcspeed((t - n) * delta_t, 75.0, initialspeed,
                                           0.06)  # do it when n==i,initial speed 10 mi/h
                            travel = volDLC * 1.467 * delta_t  # the travel distance of DLC
                            loc = loc - travel  # the location of DLC cell for next step t+1
                            print 'DLC location is', loc
                            if cell < int((from_x - loc) / dx):
                                k0[cell + 1] = k0[cell]
                            cell = int((from_x - loc) / dx)  # the cell of DLC for the next step t+1
                            print 'DLC is at cell', cell, 'at time', t
                            cap[cell, t] = 0.0  # set the capactiy of t+1,cell to zero
                            k = [upstream_dens[t + 1]]
                            k_v = [den_to_v(upstream_dens[t + 1], u, w, kj)]
                            # actually, here the time step i already become i+1, we should do Godunov again
                            for j in range(1, iL - 1):  # from second cell to the upstream of the downstream
                                if j != iL - 2:
                                    s = k0[j] + 1 / u * (transFlow(u, w, kj, k0[j - 1], k0[j], cap[j - 1, t]) -
                                                         transFlow(u, w, kj, k0[j], k0[j + 1], cap[j, t]) + alpha *
                                                         inflow[
                                                             j, t] - beta * outflow[j, t])
                                    if j == cell:
                                        print'transfer flow from cell', j, 'to cell', j + 1, 'is', transFlow(u, w, kj,
                                                                                                             k0[j],
                                                                                                             k0[j + 1],
                                                                                                             cap[j, t])
                                        print'before Godunov, the new DLC cell k0[', j, '] is', k0[j]
                                        print'the following cell of DLC is k0[', j - 1, '] is', k0[j - 1]
                                        print'after transmission, new the DLC cell density is', s
                                else:
                                    inflo = F(u, w, kj, k0[j - 1], k0[j], cap[j, t - 1], cap[j, t])
                                    ouflo = F(u, w, kj, k0[j], k0[j + 1], cap[j + 1, t - 1], cap[j + 1, t])
                                    s = k0[j] + 1 / u * (inflo - ouflo + alpha * inflow[j, t] - beta * outflow[j, t])
                                if s < 0:
                                    s = 1.0
                                k.append(s)  # k is list recording the density of all road cells at time t+1
                                k_v.append(den_to_v(s, u, w, kj))
                            k[cell + 1] = kc
                            k_v[cell] = volDLC
                            k.append(0)  # final value of k and k_v, the downstream value
                            k_v.append(den_to_v(0, u, w, kj))
                            k0_v = k_v
                            k0 = k  # renew the list of density at time t, with the estimated density list t+1
                            sol.append(k)
                            sol_v.append(k_v)
                            print 'the speed of DLC is', volDLC
                            # print 'the density of DLC downstream traffic is', k0[cell + 1]
                            print 'the speed of downstream traffic is', k0_v[cell + 1]
                            t = t + 1
                        # while not, it will end the loop and exit


    z = np.array(sol)
    z_v = np.array(sol_v)
    return z,z_v




def evaluate_model(time_list, loc_list, speed_list, model, delta_t, delta_x, from_x, to_x):

    ABS = []
    MS = []
    predicted = []
    observed = []
    # test all the records in real trajectory, for any record, first find its corresponding time and space cell
    # use the real speed value, compare with the predicted cell speed, calculate the absolute error.
    for i in range(len(time_list)-2):
        n_t = int(time_list[i]/delta_t)
        if n_t == model.shape[0]:
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

    # plt.figure(9)
    plt.subplots(2, 1, sharex='col')
    plt.subplot(211)
    plt.title('speed data')
    plt.scatter(time_list, loc_list, c=speed_list, s=2, lw=0, cmap='jet')
    plt.gca().invert_yaxis()
    plt.colorbar()

    plt.subplot(212)
    plt.scatter(time_list, loc_list, c=ABS, s=2, lw=0, cmap='jet')
    plt.title('the square error between simulation and real speed')
    plt.colorbar()
    plt.xlim(0, 3600)
    plt.ylim(from_x, to_x)
    plt.show()

    sq_error = np.sqrt(sum([aserror**2 for aserror in MS])/len(MS))
    return sq_error



if __name__ == '__main__':
    Godunov(100.0, 20.0, 150.0, 0.5, 2.0, 4.0/60)
