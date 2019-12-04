import numpy as np
import matplotlib.pyplot as plt

def F(u, w, kj, ku, kd, cap_u, cap_d):
    supply = lambda Q, w, kj, k: min(Q, w*(kj-k))
    demand = lambda Q, u, k: min(Q, k*u)
    d_u = demand(cap_u, u, ku)
    s_d = supply(cap_d, w, kj, kd)
    return min(d_u, s_d)


def den_to_v(den, v, w, kj):
    kc = w/(v+w)*kj
    if den<=kc:
        return v
    else:
        q = w*(kj-den)
        return q/den


def initial_cond(x, kj, u, w):
    kc = kj * w / (u + w)
    if x <= 1.5 and x >0.8:
        return kj
    elif x > 1.5:
        return kc
    else:
        return kc/2


def Godunov(u, w, kj, delta_t, from_x, to_x,
            begin_t, end_t, inflow, outflow,
            cap, upstream_dens,
            alpha=1.0, beta=1.0):
    '''implements the Godunov scheme

    '''
    Q = w * u * kj / (w + u)
    kc = kj * w / (u + w)
    T1 = begin_t.tm_hour*3600+begin_t.tm_min*60+begin_t.tm_sec
    T2 = end_t.tm_hour*3600+end_t.tm_min*60+end_t.tm_sec
    T = T2 - T1
    dx = delta_t * u*5280.0/3600
    iL = int((from_x-to_x) / dx)
    iT = int(T / delta_t)
    k0 = []
    sol = []
    sol_v = [] # speed solution
    # for i in range(0, iL):
    #    k0.append(initial)
    k0_v = [0.0]*iL
    k0_v[0] = den_to_v(upstream_dens[0], u, w, kj)

    k0 = [0.0]*iL
    k0[0] = upstream_dens[0]

    sol_v.append(k0_v)
    sol.append(k0)
    for i in range(0, iT-1):
        k = [upstream_dens[i+1]]
        k_v = [den_to_v(upstream_dens[i+1], u,w,kj)]
        for j in range(1, iL-1):
            if j != iL - 2:
                s = k0[j] + 1/u * (F(u, w, kj, k0[j-1], k0[j], cap[j-1, i], cap[j,i]) -
                                   F(u, w, kj, k0[j], k0[j+1], cap[j, i], cap[j+1,i])
                                   + alpha*inflow[j, i] - beta*outflow[j,i])
            else:
                inflo = F(u, w, kj, k0[j-1], k0[j], cap[j, i-1], cap[j, i])
                ouflo = F(u, w, kj, k0[j], k0[j+1], cap[j+1, i-1], cap[j+1, i])
                s = k0[j] + 1/u * (inflo - ouflo + alpha*inflow[j, i] - beta*outflow[j,i])
            if s<0:
                s = 0
            k.append(s)
            k_v.append(den_to_v(s, u, w, kj))
        k.append(0)
        k_v.append(den_to_v(0, u, w, kj))
        k0_v = k_v
        k0 = k
        sol.append(k)
        sol_v.append(k_v)

    z = np.array(sol)
    z_v = np.array(sol_v)

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 2, 1, projection='3d')
    # ax1 = fig.add_subplot(1, 2, 2)
    # p = ax.plot_surface(X, Y, z)
    ext = [0, T,from_x, to_x]
    asp = (ext[1]-ext[0])*1.0/(from_x-to_x)/3
    plt.figure(7)
    plt.imshow(np.transpose(z)[::-1], aspect = asp, extent=ext, interpolation='none')
    plt.colorbar()
    plt.xlabel('Time (seconds)')
    plt.ylabel('Location(ft)')
    plt.show()


    plt.figure(8)
    plt.imshow(np.transpose(z_v)[::-1], aspect = asp, extent=ext, cmap='jet_r', interpolation='none')
    plt.colorbar()
    plt.xlabel('Time (seconds)')
    plt.ylabel('Location(ft)')
    plt.show()
    return z_v


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
    '''
    plt.figure(9)
    plt.scatter(time_list, loc_list, c=ABS, s=2, lw=0, cmap='jet')
    plt.colorbar()
    plt.xlim(0, 3600)
    plt.ylim(from_x, to_x)
    plt.show()
    '''
    sq_error =np.sqrt(sum([aserror**2 for aserror in MS])/len(MS))
    return sq_error


if __name__ == '__main__':
    Godunov(100.0, 20.0, 150.0, 0.5, 2.0, 4.0/60)
