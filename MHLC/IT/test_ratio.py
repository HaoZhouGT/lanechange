# this is to implement the desire function and the IT principle

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

sns.set()

w=15.0
u=110.0 # note that here we use the speed in ft/s
tau1= 3.0
gamma= 1.0 # gamma will change the shape of the exponential distribution, which represents the influence of density
v4= 60.0
s0= 2500
p0= 0.5
dx= 55.0




def ratio(demand,v,coeff=1.0):
    '''
    this function implements the IT principle
    :param demand: the lane change demand from origin lane
    :param v: the target lane speed
    :return: the fraction of LC flow that can proceed
    the coffe is to adjust the priority of lane change demand from origin lane
    note here the through demand from target lane is always capacity which is 2500.0
    so the fraction can be estimated using the following function
    '''
    supply=150.0/(1/v+1/w)
    p=supply/(2500.0+demand)
    return min(1,coeff*p)



print(ratio())
