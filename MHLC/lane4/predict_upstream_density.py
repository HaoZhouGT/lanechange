
import matplotlib.pyplot as plt
import time
import numpy as np


def change_to_sec(time):
    '''change the time object to absolute seconds

    '''
    return time.tm_hour*3600 + time.tm_min*60 + time.tm_sec

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


def compute_arri_model(rec_list, begin_t, end_t):
    rec_list_sec = [change_to_sec(rec.time) for rec in rec_list]
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
    rec_cand_list_sec = rec_list_sec[k1-shift:k2]
    flow = []
    for t in range(begin_t_sec, end_t_sec+1):
        for i in range(len(rec_cand_list_sec)-1):
            if t >= rec_cand_list_sec[i] and t <= rec_cand_list_sec[i+1]:
                flow_i = 1.0*shift/(rec_cand_list_sec[i]
                                    - rec_cand_list_sec[i-10])*3600
                flow_i1 = 1.0*shift/(rec_cand_list_sec[i+1]
                                     - rec_cand_list_sec[i-9])*3600
                flow.append((flow_i+flow_i1)/2)
    return flow


def predict_upstream_den(kc, ff_speed, s_buffer, arrival_flow_model,
                         upstream_speed_model, delta_t, boundary_delta_t):
    t_limit = len(upstream_speed_model)*boundary_delta_t
    density = []
    t = 0
    s = []
    while(t < t_limit):
        t_n = int(t)
        remaining_t = t-t_n
        flow = arrival_flow_model[t_n] + remaining_t*(arrival_flow_model[t_n+1]
                                                      - arrival_flow_model[t_n])
        t_n_s = int(t/boundary_delta_t)
        remaining_t = t - t_n_s*boundary_delta_t
        print 't_n_s', t_n_s
        print 't', t
        print 'remaining_t', remaining_t
        if t_n_s < len(upstream_speed_model)-1:
            speed = upstream_speed_model[t_n_s]
            + remaining_t*(upstream_speed_model[t_n_s+1]
                           - upstream_speed_model[t_n_s])/boundary_delta_t
        else:
            speed = upstream_speed_model[t_n_s]
        s.append(speed)
        if speed < ff_speed-s_buffer:
            density.append(kc)
        else:
            density.append(flow/ff_speed)
        t = t + delta_t
    plt.figure()
    plt.title('predicted upstream density')
    plt.plot(density)
    # plt.plot(upstream_speed_model)
    return density


def gen_boundary_speed_model(delta_t, delta_x, w, rec_list,
                             boundary, begin_t, end_t):
    ''' generate boundary condition by estimating the speed at the boundary

    Keyword arguments:
        delta_t -- time interval in seconds
        delta_x -- space interval in foot
        T -- total time period in second
        w -- wave speed in feet/s

    Returns: list

    '''

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
                temp_speed.append(rec.velocity)
        if len(temp_speed) != 0:
            boundary_speed.append(sum(temp_speed) / len(temp_speed))
        else:
            boundary_speed.append(-1)

    med = 60  #66, by defalut set it to 60 for lane 4
    #  np.median(np.array(boundary_speed))
    filtered_speed = [speed if speed != -1 else med for speed in boundary_speed]
    plt.figure()
    plt.plot(range(0, len(filtered_speed)), filtered_speed)
    plt.show()
    return filtered_speed



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
    kc = wave_speed * jam_density/(ff_speed + wave_speed)
    # simulation parameters
    simu_timestep = 0.5
    simu_xstep = simu_timestep/3600*ff_speed*5280

    # # resolution for estimating speed at the boundary
    # boundary_delta_t = 9.0
    # boundary_delta_x = 110
    #
    # upstream_res = extract_record('lane4.csv')
    # upstream_boundary = from_x - boundary_delta_x
    #
    # print 'begin estimating upstream speed'
    # upstream_speed_model = gen_boundary_speed_model(boundary_delta_t,
    #                                                 boundary_delta_x,
    #                                                 wave_speed,
    #                                                 upstream_res,
    #                                                 upstream_boundary,
    #                                                 begin_t, end_t)
    #
    #
    # ##66, since we notice the upstream boundary speed is less than free flow speed
    # # the density can be replace with critical density, because the sending function is the same for any congested part
    #
    # print 'begin estimating upstream density'

    upstream_den = [kc for i in range(int((change_to_sec(end_t)-change_to_sec(begin_t))/simu_timestep))]
    print('number of upsteam density list is ',len(upstream_den))
    np.savetxt('upstream_den', np.array(upstream_den))
