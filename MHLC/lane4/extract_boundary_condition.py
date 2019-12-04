import time
import matplotlib.pyplot as plt
import numpy as np

class veh_record(object):
    def __init__(self, veh_id, time, loc, velocity):
        self.veh_id = veh_id
        self.time = time
        self.loc = loc
        self.velocity = velocity


##66, this is a popular method, to define a class, then put the class into a list

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


def extract_boundary_record(file_name, boundary_bottom, boundary_upper):
    '''extract only the boundary record from the csv file.
    later those boundary data will be used to generate boundary capacity

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
            if boundary_bottom<loc and loc<boundary_upper:
                rec_list.append(veh_record(veh_id, time_obj, loc, velocity))
        return rec_list

def change_to_sec(time):
    '''change the time object to absolute seconds

    '''
    return time.tm_hour*3600 + time.tm_min*60 + time.tm_sec

def gen_boundary_cap_model(delta_t, delta_x, w, kj,
                           rec_list, boundary, begin_t, end_t):
    ''' generate boundary condition by estimating the speed at the boundary
    because the speed is only thing we got

    Keyword arguments:
        delta_t -- time interval in seconds
        delta_x -- space interval in foot
        T -- total time period in second
        w -- wave speed in feet/s

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
    #66, this is to filter the data, such that only records within boundary range are selected

    for i in range(0, iT): #66, create a timelist, prepare to compute average speed and put them in list
        temp_speed = []
        for rec in rec_list_candi:
            rec_sec = change_to_sec(rec.time) - begin_t_sec
            below_line = rec.loc - w*(float(rec_sec)-i * delta_t) - boundary
            # print 'below', below_line
            above_line = rec.loc - w*(float(rec_sec)-(i+1)*delta_t)-boundary
            # print 'above', above_line
            if above_line >= 0 and below_line <= 0:
                temp_speed.append(rec.velocity)
        if len(temp_speed) != 0:
            boundary_speed.append(sum(temp_speed) / len(temp_speed))
        else:
            boundary_speed.append(-1)
    med = 60 # guess the med is a default value, for lack of data at the beginning, average speed is assumed to be 60 km/h
    # np.median(np.array(boundary_speed))
    filtered_speed = [speed if speed != -1 else med for speed in boundary_speed]
    f = lambda s, w, kj: (s*w)/(s+w)*kj
    w = w*3600.0/5280.0
    cap_model = [f(s, w, kj) for s in filtered_speed]
    print('the origin downstream capacity data is',cap_model)
    plt.figure()
    plt.title('the origin downstream boundary speed data is')
    plt.plot(filtered_speed)
    plt.figure()
    plt.title('downstream capacity series is')
    plt.plot(range(len(boundary_speed)), cap_model)
    plt.show()
    return cap_model


def predict_boundary_cap(cap_model, model_step, timestep):
    '''predict the capacity at the timestamp based on the cap_model using linear
    interpolation

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
    plt.figure()
    plt.title('interpolated downstream capacity from begin_t to end_t')
    plt.plot(cap)
    plt.show()
    return cap

if __name__ == '__main__':
    # modelling time and space domain
    begin_t = time.strptime('6:35:00', '%H:%M:%S')
    end_t = time.strptime('7:35:00', '%H:%M:%S')
    from_x = 3850
    to_x = 1150


    # fundamental diagram parameters
    ff_speed = 75.0
    jam_density = 150.0
    wave_speed = 15.0
    kc = wave_speed * jam_density / (ff_speed + wave_speed)
    # simulation parameters
    simu_timestep = 0.5
    simu_xstep = simu_timestep / 3600 * ff_speed * 5280

    # resolution for estimating speed at the boundary
    boundary_delta_t = 9.0
    boundary_delta_x = 110



    print('begin compute downstream capacity from raw data')
    downstream_res = extract_boundary_record('lane4.csv', 1110, 1350)
    downstream_boundary = to_x

    #66, need FD parameters to transfer speed value to density
    downstream_cap_model = gen_boundary_cap_model(boundary_delta_t,
                                                  boundary_delta_x,
                                                  wave_speed,
                                                  jam_density,
                                                  downstream_res,
                                                  downstream_boundary,
                                                  begin_t, end_t)

    print('begin predict downstream capacity model given simulation step')
    downstream_cap = predict_boundary_cap(downstream_cap_model,
                                            boundary_delta_t,
                                            simu_timestep)

    np.savetxt('downstream_cap', np.array(downstream_cap))
    print('nubmer of time steps of the downstream cap', len(downstream_cap))