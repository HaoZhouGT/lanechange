import matplotlib.pyplot as plt


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
    med = 10  # np.median(np.array(boundary_speed))
    filtered_speed = [speed if speed != -1 else med for speed in boundary_speed]
    f = lambda s, w, kj: (s*w)/(s+w)*kj
    w = w*3600.0/5280.0
    print w, kj
    cap_model = [f(s, w, kj) for s in filtered_speed]
    print cap_model
    plt.figure(2)
    plt.plot(filtered_speed)
    plt.figure(3)
    plt.plot(range(len(boundary_speed)), cap_model)
    plt.show()
    return cap_model


def predict_boundary_cap(boundary_cap_model, model_step, timestep):
    '''predict the capacity at the timestamp based on the cap_model using linear
    interpolation

    timestamp in seconds

    '''
    t = 0.0
    model_limit = len(boundary_cap_model)*model_step
    cap = []
    while(t<model_limit):
        ind_t1 = t/model_step
        ind = int(ind_t1)
        if ind < len(boundary_cap_model)-1:
            t1 = t - ind*model_step
            capacity_t = boundary_cap_model[ind] + t1*(boundary_cap_model[ind+1] - boundary_cap_model[ind])/model_step
            cap.append(capacity_t)
        else:
            cap.append(boundary_cap_model[-1])
        t = t + timestep
    plt.figure(5)
    plt.plot(cap)
    plt.show()
    return cap

if __name__ == '__main__':

    gen_boundary_cap_model(delta_t, delta_x, w, kj,
                           rec_list, boundary, begin_t, end_t)
    predict_boundary_cap
