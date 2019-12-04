# !/opt/local/bin/python
from sets import Set
import time
import numpy as np
import matplotlib.pyplot as plt


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



def extract_count_record(filename):
    '''extract record from the count data csvfile

    '''
    rec_list = []
    with open(filename, 'rb') as csvfile:
        for line in csvfile:
            words = line.replace('\r\n', '').split(',')
            print words
            time_obj = time.strptime(words[0], '%H:%M:%S')
            rec_list.append(time_obj)
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
    # plt.figure(1)
    # plt.scatter(time_list, location, c=velocity_list,
    #             s=2, lw=0, cmap='jet_r')
    # plt.colorbar()
    # plt.xlim(0, end_t_sec-begin_t_sec)
    # plt.ylim(from_x, to_x)
    # # plt.gca().invert_yaxis()
    # plt.show()
    return time_list, location, velocity_list








def change_to_sec(time):
    '''change the time object to absolute seconds

    '''
    return time.tm_hour*3600 + time.tm_min*60 + time.tm_sec




if __name__ == '__main__':

    # modelling time and space domain
    begin_t = time.strptime('6:35:00', '%H:%M:%S')
    end_t = time.strptime('7:35:00', '%H:%M:%S')
    from_x = 3850
    to_x = 1150

    time_list, loc_list, speed_list = gen_trajectory('lane7.csv',
                                                     begin_t, end_t,
                                                     from_x, to_x)
    np.savetxt('time_list', time_list)
    np.savetxt('loc_list', loc_list)
    np.savetxt('speed_list', speed_list)

    T1 = begin_t.tm_hour*3600+begin_t.tm_min*60+begin_t.tm_sec
    T2 = end_t.tm_hour*3600+end_t.tm_min*60+end_t.tm_sec
    T = T2 - T1
    ext = [0, T, from_x, to_x]
    asp = (ext[1]-ext[0])*1.0/(from_x-to_x)/3
    plt.subplot(111)
    plt.title('trajectory of lane 7')
    plt.scatter(time_list, loc_list, c=speed_list, s=2, lw=0, cmap='jet_r')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.show()



