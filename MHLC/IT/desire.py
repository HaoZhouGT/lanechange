import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt


def func(x):
    return np.exp(-x)-np.exp(-2*x)


# f = lambda x: np.exp(-x)-np.exp(-2*x)
#
#
# def cdf(x):
#     a,b= integrate.quad(f,0,x)
#     return a
#
#
# x=np.linspace(1,4000,num=200)
# plt.figure()
# plt.plot(np.cumsum(x,f,))
# plt.show()
#
# print('the integration value is',cdf(8000))

x= np.arange(0,4000,1)
plt.plot(x,func(x))
plt.show()