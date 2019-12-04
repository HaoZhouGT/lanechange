
import time
import numpy as np
import matplotlib.pyplot as plt


def change_to_sec(time):
    '''change the time object to absolute seconds

    '''
    return time.tm_hour*3600 + time.tm_min*60 + time.tm_sec


def extract_lc_record(filename):
    '''
    extract lane change records from csv files

    '''
    rec_list = []
    with open(filename, 'rb') as csvfile:
        for line in csvfile:
            words = line.replace('\r\n', '').split(',')
            time_obj = time.strptime(words[1], '%H:%M:%S')
            loc = float(words[0])
            rec_list.append((loc, time_obj))
    return rec_list

def compute_lc_model(lc_list, begin_t, end_t, delta_t, delta_x, from_x, to_x):
    '''compute the flow caused by lane changes
    ##66, note that the input file is a list of lane changes
    first element is location and second column is time

    '''
    begin_t_sec = change_to_sec(begin_t)
    end_t_sec = change_to_sec(end_t)
    n_t = int((end_t_sec - begin_t_sec)/delta_t)+1
    print n_t
    n_x = int((from_x - to_x)/delta_x)+1
    print n_x
    lc_list_cand = [lc for lc in lc_list if lc[0] <= from_x and lc[0] >= to_x
                    and change_to_sec(lc[1]) <= change_to_sec(end_t)
                    and change_to_sec(lc[1]) >= change_to_sec(begin_t)]
    lc_flow = np.zeros((n_x, n_t)) # initialize the lc_flow model
    for lc in lc_list_cand:
        m = int((from_x-lc[0])/delta_x)
        n = int((change_to_sec(lc[1])-change_to_sec(begin_t))/delta_t)
        lc_flow[m, n] += 1  #66 this is the count of LC number in that time-space cell
    lc_flow = 1.0*lc_flow/delta_t/delta_x #66 note that it should be LC flow rate
    return lc_flow


def predict_lc_flow(lc_model, lc_delta_x, lc_delta_t, delta_x, delta_t):
    x_limit = lc_model.shape[0]*lc_delta_x
    t_limit = lc_model.shape[1]*lc_delta_t

    x_n = int(x_limit/delta_x)
    t_n = int(t_limit/delta_t)

    res = np.zeros([x_n, t_n])

    for i in range(0, x_n):
        x = i*delta_x
        for j in range(0, t_n):
            t = j*delta_t
            in_x = int(x/lc_delta_x)
            in_t = int(t/lc_delta_t)
            res[i, j] = lc_model[in_x, in_t]*delta_x*3600
    plt.figure()
    plt.title('the predicted lane change flow rate with given delta_t and delta_x as average scale')
    plt.imshow(res[::-1], aspect=20)
    plt.colorbar()
    plt.show()
    return res


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

    # compute LC flow rate at a scale of lc_delta_t*lc_delta_x
    lc_delta_t = 12.0  # seconds
    lc_delta_x = 200.0  # ft
    in_list = extract_lc_record('inflow.csv')
    out_list = extract_lc_record('outflow.csv')

    print 'compute lane change inflow model'
    inflow_model = compute_lc_model(in_list, begin_t, end_t, lc_delta_t,
                                    lc_delta_x, from_x, to_x)

    inflow = predict_lc_flow(inflow_model, lc_delta_x, lc_delta_t,
                             simu_xstep, simu_timestep)

    np.savetxt('inflow', np.array(inflow))

    print 'compute lane change outflow model'
    outflow_model = compute_lc_model(out_list, begin_t, end_t, lc_delta_t,
                                     lc_delta_x, from_x, to_x)

    outflow = predict_lc_flow(outflow_model, lc_delta_x, lc_delta_t,
                              simu_xstep, simu_timestep)

    np.savetxt('outflow', np.array(outflow))