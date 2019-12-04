import time
from math import exp


def change_to_sec(time):
    '''change the time object to absolute seconds
    '''
    return time.tm_hour*3600 + time.tm_min*60 + time.tm_sec


def speed(t,Vc=33.0,V0=10.0,beta=0.01):
    v=Vc-(Vc-V0)*exp(-1*beta*t)
    v=2.237*v # turn m/s to mi/h
    return v # return the speed v(t) with a linear acceleration model

def mindisturb(disturbrec,from_x,begin_t,delta_t,delta_x):
    a=[]
    for rec in disturbrec:
        loc = rec[0]  # this is the initial location of the DLC vehicle
        tt = change_to_sec(rec[1])
        m = int((from_x - loc) / delta_x)
        n = int((tt - change_to_sec(begin_t)) / delta_t)
        a.append(n)
    return a


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
            rec_list.append((loc, time_obj))
    return rec_list


if __name__ == '__main__':

    # modelling time and space domain
    # begin_t = time.strptime('6:35:00', '%H:%M:%S')
    # end_t = time.strptime('7:35:00', '%H:%M:%S')
    # from_x = 3850
    # to_x = 1150
    # simu_timestep = 0.5
    # simu_xstep = simu_timestep/3600*75*5280
    #
    # print 'extract DLC starting location before simulation'
    # disturb_rec = extract_disturb_record('dlc_in.csv')
    # reclist = mindisturb(disturb_rec,from_x,begin_t,simu_timestep,simu_xstep)
    # print reclist
    # if 52 in reclist:
    #     print 'yes'
    res = speed(10.0, Vc=33.0, V0=10.0, beta=1.0)
    print res
