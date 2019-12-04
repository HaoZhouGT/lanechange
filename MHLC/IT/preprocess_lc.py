# !/opt/local/bin/python
from sets import Set
import time
import numpy as np
import matplotlib.pyplot as plt
from math import exp



def speed(Vc,V0,beta,t):
    v=Vc-(Vc-V0)*exp(-1*beta*t)
    return v # return the speed v(t) with a linear acceleration model


def den_to_v(den, v, w, kj):
    kc = w/(v+w)*kj
    if den<=kc:
        return v
    else:
        q = w*(kj-den)
        return q/den

class veh_record(object):
    def __init__(self, veh_id, time, loc, velocity):
        self.veh_id = veh_id
        self.time = time
        self.loc = loc
        self.velocity = velocity


def extract_record(file_name):
    '''extract record from the csv file.

    Keyword arguments:
        file_name -- dataset name in string

    Returns: list

    '''
    rec_list = []
    with open(file_name, 'rb') as csvfile:
        for line in csvfile:
            words = line.split(',')
            veh_id = int(words[0])
            time_obj = time.strptime(words[2], '%H:%M:%S')
            # time_tup = (time_obj.tm_hour, time_obj.tm_min, time_obj.tm_sec)
            loc = float(words[3])
            velocity = float(words[4])
            rec_list.append(veh_record(veh_id, time_obj, loc, velocity))
        return rec_list


def extract_lc_record(filename):
    '''extract lane change records from csv files

    '''
    rec_list = []
    with open(filename, 'rb') as csvfile:
        for line in csvfile:
            words = line.replace('\r\n', '').split(',')
            time_obj = time.strptime(words[1], '%H:%M:%S')
            loc = float(words[0])
            rec_list.append((loc, time_obj))
    return rec_list


def extract_count_record(filename):
    '''extract record from the count data csvfile

    '''
    rec_list = []
    with open(filename, 'rb') as csvfile:
        for line in csvfile:
            words = line.replace('\r\n', '').split(',')
            # print words
            time_obj = time.strptime(words[0], '%H:%M:%S')
            rec_list.append(time_obj)
    return rec_list


def extract_disturb_record(filename):
    '''extract the disturbance time and location from csvfile

    '''
    rec_list = []
    with open(filename, 'rb') as csvfile:
        for line in csvfile:
            words = line.replace('\r\n','').split(',')
            # print words
            loc = float(words[0])
            time_obj = time.strptime(words[1], '%H:%M:%S')
            rec_list.append((loc,time_obj))
    return rec_list


def gen_trajectory(filename, begin_t, end_t, from_x, to_x):
    begin_t_sec = change_to_sec(begin_t)
    end_t_sec = change_to_sec(end_t)
    time_list = []
    location = []
    velocity_list = []
    with open(filename, 'rb') as csvfile:
        for line in csvfile:
            words = line.replace('\r\n', '').split(',')
            time_obj = time.strptime(words[2], '%H:%M:%S')
            time_sec = change_to_sec(time_obj)
            loc = float(words[3])
            velocity = float(words[4])
            if (time_sec >= begin_t_sec and time_sec <= end_t_sec and loc >= to_x and loc <= from_x):
                time_list.append(time_sec-begin_t_sec)
                location.append(loc)
                if velocity < 0:
                    velocity = 0
                if velocity > 75:
                    velocity = 75
                velocity_list.append(velocity)
    plt.figure(1)
    plt.title('trajectory and speed data ')
    plt.scatter(time_list, location, c=velocity_list,s=2, lw=0, cmap='jet_r')
    plt.colorbar()
    plt.xlim(0, end_t_sec-begin_t_sec)
    plt.ylim(from_x, to_x)
    plt.show()
    return time_list, location, velocity_list

# function takes the rec_list data contains all the vehicles arriving at the first cell (upstream)
# which means we cannot change the upstream location in this question
# the rec_list records only arrival time, in ranked sequence, from 6:32---9:32
def compute_arri_model(rec_list, begin_t, end_t):
    rec_list_sec = [change_to_sec(rec) for rec in rec_list] # list comprehension
    begin_t_sec = change_to_sec(begin_t)
    end_t_sec = change_to_sec(end_t)

    rec_cand_list_sec = [rec for rec in rec_list_sec if rec <= end_t_sec]
    k1 = 0
    for i in range(len(rec_list)):
        if rec_list_sec[i] - begin_t_sec >= 0:
            k1 = i
            break
    k2 = 0
    for i in range(len(rec_list)):
        if rec_list_sec[i] - end_t_sec >= 0:
            k2 = i
            break
    shift = 10
    rec_cand_list_sec = rec_list_sec[k1-shift:k2] # the arrival records between [begin_t, end_t]
    flow = []
    for t in range(begin_t_sec, end_t_sec+1):
        for i in range(len(rec_cand_list_sec)-1):
            if t >= rec_cand_list_sec[i] and t <= rec_cand_list_sec[i+1]: # not every second has coming vehicles
                flow_i = 1.0*shift/(rec_cand_list_sec[i]- rec_cand_list_sec[i-10])*3600 # average time headway between 10 vehicles
                flow_i1 = 1.0*shift/(rec_cand_list_sec[i+1]- rec_cand_list_sec[i-9])*3600
                flow.append((flow_i+flow_i1)/2)
    return flow
# this function computes the arrival flow rate at every second



# it needs the arrival_flow_model at every second, the upstream speed model and time cell units
# the function takes in arriving_flow_model as input parameter, which contains arriving flow at every second.
def predict_upstream_den(kc, ff_speed, s_buffer, arrival_flow_model,
                         upstream_speed_model, delta_t, boundary_delta_t):
    t_limit = len(upstream_speed_model)*boundary_delta_t
    density = []
    t = 0
    s = []
    while(t < t_limit):
        t_n = int(t)
        remaining_t = t-t_n
        flow = arrival_flow_model[t_n] + remaining_t*(arrival_flow_model[t_n+1]- arrival_flow_model[t_n])
        t_n_s = int(t/boundary_delta_t)
        remaining_t = t - t_n_s*boundary_delta_t
        # print 't_n_s', t_n_s
        # print 't', t
        # print 'remaining_t', remaining_t
        if t_n_s < len(upstream_speed_model)-1:
            speed = upstream_speed_model[t_n_s]+ remaining_t*(upstream_speed_model[t_n_s+1]
                           - upstream_speed_model[t_n_s])/boundary_delta_t
        else:
            speed = upstream_speed_model[t_n_s]
        s.append(speed)
        if speed < ff_speed-s_buffer: # assume that when speed is lower than speed-buffer, it is in congestion
            density.append(kc)
        else:
            density.append(flow/ff_speed) ## if speed is less than (ff_speed-s_buffer), does it mean congestion?
        t = t + delta_t
    plt.figure(4)
    plt.title('the upstream density along simulation time')
    plt.plot(density)
    print 'predicted upstream density',density
    # plt.plot(upstream_speed_model)
    return density
# it returns the density value of the boundary cell at every simulation_step_time delta_t
# density is a list


def up_den_time(upstream_density,begin_t,end_t,simu_delta_t):
    density=[]
    tmax=len(upstream_den)
    iT=int(tmax/simu_delta_t)
    i=0
    while (i*simu_delta_t<tmax):
        index=int(i*simu_delta_t)
        density.append(upstream_density[index])
        i=i+1
    return density


# before compute the lane change flow, the prerequisite lc_list must be given.
def compute_lc_model(lc_list, begin_t, end_t, delta_t, delta_x, from_x, to_x):
    '''compute the flow caused by lane changes
    lc_delta_t = 12.0#seconds
    lc_delta_x = 200.0#ft
    '''
    begin_t_sec = change_to_sec(begin_t)
    end_t_sec = change_to_sec(end_t)
    n_t = int((end_t_sec - begin_t_sec)/delta_t)+1
    # print n_t
    n_x = int((from_x - to_x)/delta_x)+1
    # print n_x
    lc_list_cand = [lc for lc in lc_list if lc[0] <= from_x and lc[0] >= to_x
                    and change_to_sec(lc[1]) <= change_to_sec(end_t)
                    and change_to_sec(lc[1]) >= change_to_sec(begin_t)] # this is to extract effective lane change recs
    lc_flow = np.zeros((n_x, n_t)) # initialize the lc_flow model
    for lc in lc_list_cand: # traverse all effective lane change records in lc_list
        m = int((from_x-lc[0])/delta_x)
        n = int((change_to_sec(lc[1])-change_to_sec(begin_t))/delta_t)
        lc_flow[m, n] += 1.0/delta_t*3600.0 # the number of lane changes happened in cell m during time slot n
    # lc_flow = 1.0*lc_flow/delta_t/delta_x
    # plt.figure(5)
    # plt.title('the lane change vehicle count before prediction')
    # ext = [begin_t_sec-23700, end_t_sec-23700, from_x, to_x]  # 23700 is only a number for horizontal shift
    # asp = (ext[1] - ext[0]) * 1.0 / (from_x-to_x)/ 3  # determine the ratio of y and x axis
    # plt.imshow(lc_flow[::-1], aspect=asp, extent=ext, cmap='jet')
    # plt.colorbar()
    # plt.show()

    return lc_flow
# the returned result is a matrix recording the lane changes happened in (space i, time j).
# Note that the time and space resolution here is different from simuulation time and space step
# And the unit here is processed to be consistent with derivatives of k and q.


def predict_lc_flow(lc_model, lc_delta_x, lc_delta_t, delta_x, delta_t,from_x, to_x):
    x_limit = lc_model.shape[0]*lc_delta_x
    t_limit = lc_model.shape[1]*lc_delta_t

    x_n = int(x_limit/delta_x)
    t_n = int(t_limit/delta_t)  # type: int # find the position of the simulation time-cell

    res = np.zeros([x_n, t_n])

    for i in range(0, x_n):
        x = i*delta_x
        for j in range(0, t_n):
            t = j*delta_t
            in_x = int(x/lc_delta_x)
            in_t = int(t/lc_delta_t) # find where the simulation cell locates in lane-change model
            res[i, j] = lc_model[in_x, in_t]
    plt.figure(6)
    plt.title('predict the lane change flow on each simulation time-cell')
    ext = [0, t_limit, from_x, to_x]  # ext means the range of x and y axis
    asp = (ext[1] - ext[0]) * 1.0 / (from_x-to_x)/ 3  # determine the ratio of y and x axis
    plt.imshow(res[::-1], aspect=asp, extent=ext, cmap='jet')
    plt.colorbar()
    plt.show()
    return res



def change_to_sec(time):
    '''change the time object to absolute seconds
    '''
    return time.tm_hour*3600 + time.tm_min*60 + time.tm_sec


def gen_initial(timestamp, from_x, to_x, delta_x, rec_list):
    ''' get initial condition by counting vehicles, but could be error prone
    since the trajectory data does not include all vehicles
    Keyword arguments:
        rec_list -- vehicle records in list
        timestamp -- time stamp in seconds
    Returns: list

    '''
    s = from_x
    s_list = []
    count = []
    while(s > to_x):
        count.append(0)
        s_list.append(s)
        s = s - delta_x
    i = 0
    for s in s_list:
        appeared = Set()
        for rec in rec_list:
            if rec.time == timestamp:
                if rec.loc <= s and rec.loc > s - delta_x: # calculate the density of the downstream boundary
                    appeared.add(rec.veh_id)
        count[i] = len(appeared)
        i = i+1
    count = [c/delta_x for c in count]
    return count



def gen_boundary_speed_model(delta_t, delta_x, w, rec_list,boundary, begin_t, end_t):
    ''' generate boundary condition by estimating the speed at the boundary
    Keyword arguments:
        delta_t -- boundary delta t
        delta_x -- boundary delta x
        T -- total time period in second
        w -- wave speed in feet/s
    Returns: list
    '''
    boundary_speed = []
    begin_t_sec = change_to_sec(begin_t)
    end_t_sec = change_to_sec(end_t)
    iT = int(round((end_t_sec-begin_t_sec)/delta_t))
    rec_list_candi = [rec for rec in rec_list if boundary<= rec.loc <= boundary + delta_x
                      and change_to_sec(rec.time) <= end_t_sec
                      and change_to_sec(rec.time) >= begin_t_sec]
    # when calculating downstream, it is perfect ok. But when calculates upstream,
    for i in range(0, iT):
        temp_speed = []
        for rec in rec_list_candi:
            if i*delta_t <= change_to_sec(rec.time)-begin_t_sec <= (i+1)*delta_t:
                temp_speed.append(rec.velocity)
        if len(temp_speed) != 0:
            boundary_speed.append(sum(temp_speed) / len(temp_speed))
        if len(temp_speed) == 0:
            boundary_speed.append(50)
        # if i==27:
        #     boundary_speed.append(boundary_speed[24])
    plt.figure(3)
    plt.title('boundary average speed at every boundary_delta_t ')
    plt.plot(range(0, len(boundary_speed)), boundary_speed)
    plt.show()
    med = 10  # for those time cells without speed data, we set them as -1. But later we used 10 to replace -1.
    filtered_speed = [speed if speed != -1 else med for speed in boundary_speed]
    return filtered_speed
    # the returned result is a list recording the boundary speed





# it takes the boundary speed
def gen_boundary_cap_model(delta_t, delta_x, w, kj,rec_list, boundary, begin_t, end_t):
    ''' generate boundary condition by estimating the speed at the boundary
    Keyword arguments:
        rec_list -- the records of boundary, such as upstream and downstream extraced data
        delta_t -- time interval in seconds, actually it is boundary_delta_t
        delta_x -- space interval in foot, in fact it is boundary_delta_x
    Returns: list
    '''
    w = w*5280.0/3600.0
    boundary_speed = []
    begin_t_sec = change_to_sec(begin_t)
    end_t_sec = change_to_sec(end_t)
    iT = int(round((end_t_sec-begin_t_sec)/delta_t))
    rec_list_candi = [rec for rec in rec_list if rec.loc <= boundary + delta_x
                      and change_to_sec(rec.time) <= end_t_sec
                      and change_to_sec(rec.time) >= begin_t_sec]
    for i in range(0, iT):
        temp_speed = []
        for rec in rec_list_candi:
            rec_sec = change_to_sec(rec.time) - begin_t_sec
            below_line = rec.loc - w*(float(rec_sec)-i * delta_t) - boundary
            # print 'below', below_line
            above_line = rec.loc - w*(float(rec_sec)-(i+1)*delta_t)-boundary
            # print 'above', above_line
            if above_line >= 0 and below_line <= 0:
                temp_speed.append(rec.velocity) # extract velocity from records into temp_speed list
        if len(temp_speed) != 0:
            boundary_speed.append(sum(temp_speed) / len(temp_speed)) # boundary speed is a list recording the average speed in i*delta_t
        else:
            boundary_speed.append(-1)
    med = 10  # np.median(np.array(boundary_speed))
    filtered_speed = [speed if speed != -1 else med for speed in boundary_speed]
    f = lambda s, w, kj: (s*w)/(s+w)*kj # actually it is the supply capacity of the downstream boundary
    w = w*3600.0/5280.0
    # print w, kj
    cap_model = [f(s, w, kj) for s in filtered_speed]
    print 'the actual flow at boundary in every boundary_delta_t unit', cap_model
    plt.figure(2)
    plt.title('boundary average speed data in every boundary_delta_t unit')
    plt.plot(filtered_speed)
    plt.figure(3)
    plt.title('the boundary capacity model at boundary_delta_t ')
    plt.plot(range(len(boundary_speed)), cap_model)
    plt.show()
    return cap_model
    # the returned result is a list, recording the actual average flow at every boundary_delta_t unit.
    # note that the boundary model only returns in unit of the boundary resolution, boundary_delta_t, not in simu_step

def predict_boundary_cap(cap_model, model_step, timestep):
    '''predict the capacity at the timestamp based on the cap_model using linear interpolation
    timestamp in seconds
    '''
    t = 0.0
    model_limit = len(cap_model)*model_step
    cap = []
    while(t<model_limit):
        ind_t1 = t/model_step
        ind = int(ind_t1)
        if ind < len(cap_model)-1:
            t1 = t - ind*model_step
            capacity_t = cap_model[ind] + t1*(cap_model[ind+1] - cap_model[ind])/model_step
            cap.append(capacity_t)
        else:
            cap.append(cap_model[-1])
        t = t + timestep
    # plt.figure(5)
    # plt.title('predicted downstream boundary capacity at simulation time steps')
    # plt.plot(cap)
    # plt.show()
    return cap


# let predict capacity only considers the downstream
def predict_capacity(cap, boundary_cap, from_x, to_x, delta_x, begin_t, end_t, delta_t):
    n_x = int((from_x-to_x)/delta_x)
    n_t = int((change_to_sec(end_t)-change_to_sec(begin_t))/delta_t)
    capacity = np.ones((n_x,n_t))*cap
    T1 = begin_t.tm_hour * 3600 + begin_t.tm_min * 60 + begin_t.tm_sec
    T2 = end_t.tm_hour * 3600 + end_t.tm_min * 60 + end_t.tm_sec
    T = T2 - T1
    ext = [0, T, from_x, to_x] # ext means the range of x and y axis
    asp = (ext[1] - ext[0]) * 1.0 / (from_x - to_x) / 3  # determine the ratio of y and x axis
    for i in range(n_t):
        capacity[-1, i] = boundary_cap[i]
    plt.figure(10)
    plt.title('the capacity considering disturbance')
    plt.imshow(capacity[::-1],aspect=asp,extent=ext, cmap='jet_r')
    plt.colorbar()
    plt.show()
    return capacity


if __name__ == '__main__':

    # modelling time and space domain
    begin_t = time.strptime('6:35:00', '%H:%M:%S')
    end_t = time.strptime('7:35:00', '%H:%M:%S')
    from_x = 3850
    to_x = 1150

    time_list, loc_list, speed_list = gen_trajectory('cam_all.csv',begin_t, end_t,from_x, to_x)
    np.savetxt('time_list', time_list)
    np.savetxt('loc_list', loc_list)
    np.savetxt('speed_list', speed_list)
    # note that we computed the speed list from real trajectories for comparison with simulation results

    # fundamental diagram parameters
    ff_speed = 75.0
    jam_density = 240.0
    wave_speed = 15.0
    kc = wave_speed * jam_density/(ff_speed + wave_speed) # kc is used for estimating upstream density
    # simulation parameters
    simu_timestep = 0.5
    simu_xstep = simu_timestep/3600*ff_speed*5280

    # resolution for estimating speed at the boundary
    boundary_delta_t = 9.0
    boundary_delta_x = 50


    # downstream and upstream file are dependent, but actually be included in the cam_all.csv
    downstream_res = extract_record('cam_6.csv')
    downstream_boundary = to_x
    upstream_res = extract_record('cam_2.csv') # data with location from 3600~3800
    upstream_boundary = from_x - boundary_delta_x # it is used for a proper input for gen_boundary_speed_model

    print 'Estimating downstream capacity '
    # print 'wave_speed', wave_speed
    downstream_cap_model = gen_boundary_cap_model(boundary_delta_t,
                                                  boundary_delta_x,
                                                  wave_speed,
                                                  jam_density,
                                                  downstream_res,
                                                  downstream_boundary,
                                                  begin_t, end_t)

    print 'begin predict downstream capacity for simulation'
    downstream_cap = predict_boundary_cap(downstream_cap_model,
                                            boundary_delta_t,
                                            simu_timestep)
    np.savetxt('downstream_cap', np.array(downstream_cap))
    # print 'downstream capacity',len(downstream_cap)



    print 'Estimating upstream average speed at every boundary_delta t'
    upstream_speed_model = gen_boundary_speed_model(boundary_delta_t,
                                                    boundary_delta_x,
                                                    wave_speed,
                                                    upstream_res,
                                                    upstream_boundary,
                                                    begin_t, end_t)


    print 'begin estimating upstream arrival flow at every second'
    #TODO
    rc_arrival = extract_count_record('count5_new.csv')
    arrival_flow_model = compute_arri_model(rc_arrival, begin_t, end_t)
    np.savetxt('arrival_flow', np.array(arrival_flow_model))
    print 'the length of upstream arrival_flow_model',len(arrival_flow_model)



    print 'begin estimating upstream density'
    #TODO
    # the function uses a buffer value of 15, is this right?
    upstream_den = predict_upstream_den(kc, ff_speed, 15, arrival_flow_model,
                                        upstream_speed_model, simu_timestep,
                                        boundary_delta_t)
    np.savetxt('upstream_den', np.array(upstream_den))
    # print 'the length of upstream density is',len(upstream_den)


# it shows the disturb record was used in predicting capacity for all road cells at all time units
#     print 'extract disturb record and estimate capacity of all cells'
#     disturb_rec = extract_disturb_record('disturb_in.csv')
#     capacity = predict_capacity(ff_speed*kc, downstream_cap, disturb_rec, from_x, to_x, simu_xstep, begin_t, end_t,
#                                 simu_timestep)
#     np.savetxt('capacity', capacity)


    res=extract_record('cam_all.csv')
    initial = gen_initial(begin_t, from_x, to_x, simu_xstep, res)
    # initial=25.0*np.ones(49)


    # compute lane change density at a scale of lc_delta_t*lc_delta_x
    lc_delta_t = 12.0#seconds
    lc_delta_x = 200.0#ft
    # actually the resolution for lane change model is really big,how will it affect our result?
    in_list = extract_lc_record('inflow.csv')
    out_list = extract_lc_record('outflow.csv')

    print 'compute lane change inflow model'
    inflow_model = compute_lc_model(in_list, begin_t, end_t, lc_delta_t,
                                   lc_delta_x, from_x, to_x)
    inflow = predict_lc_flow(inflow_model, lc_delta_x, lc_delta_t,
                             simu_xstep, simu_timestep, from_x, to_x)
    np.savetxt('inflow', np.array(inflow))


    print 'compute lane change outflow model'
    outflow_model = compute_lc_model(out_list, begin_t, end_t, lc_delta_t,
                                    lc_delta_x, from_x, to_x)
    outflow = predict_lc_flow(outflow_model, lc_delta_x, lc_delta_t,
                             simu_xstep, simu_timestep, from_x, to_x)
    np.savetxt('outflow', np.array(outflow))


    print 'set the downstream capacity value'
    capacity = predict_capacity(ff_speed*kc, downstream_cap,
                                from_x, to_x, simu_xstep, begin_t, end_t,
                                simu_timestep)

    np.savetxt('capacity', capacity)



# # here we could set a loop, with different alpha and belta values, and set up a cost-function as the optimization
#     # values, to slove the optimization problem?
#     print 'start godunov simulation'
#     alpha = 1
#     belta = 1
#     print 'alpha',alpha
#     print 'belta',belta
#
#
#     godmodel = god.Godunov(ff_speed, wave_speed,
#                            jam_density, simu_timestep,
#                            from_x, to_x, begin_t, end_t,
#                            inflow, outflow,
#                            downstream_cap, upstream_den,initial,alpha, belta)
#     # the Godunov function will plot two figures
#
#     # it seems the godunov simulation does not take in the initial condition as an input
#     # it does not give alpha and belta as paramters, alpha and belta are default parameters
#
#     god.evaluate_model(time_list, loc_list, speed_list, godmodel, simu_timestep,simu_xstep, from_x, to_x)
#     # time_lst, loc_list and speed_list are derived by the real trajectories and used as imput to evaluate the
#     # simulation result
