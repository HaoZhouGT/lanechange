import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

u = 75.0


def mlc_desire(x,V):
    tau = 0.5 # tau is the time for a smooth MLC
    #66, we need to ensure pi can integrate on delta t and delta x, then has the unit of 1
    pi= np.exp(-1.0*x/(632+5.94*V))/u
    return pi

x = np.linspace(0,4000,500)
# plt.figure()
# plt.plot(x,mlc_desire(x,30))
# plt.plot(x,mlc_desire(x,0))
# plt.show()

cumu = np.loadtxt('remain_prob')
remain_prob = (1-cumu)*0.5
f_exit = interp1d(x,remain_prob)



# def exit_prob_4(x):
