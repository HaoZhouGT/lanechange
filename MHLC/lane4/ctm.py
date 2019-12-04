
import Godunov as god
import numpy as np
import time

# simulation range
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


print('load upstream density')
upstream_den = np.loadtxt('upstream_den')

print('load downstream capacity')
downstream_cap = np.loadtxt('downstream_cap')

print('load lane change inflow')
inflow = np.loadtxt('inflow')

print('load lane change outflow')
outflow = np.loadtxt('outflow')


alpha = 0.4
beta = 0.1


print('start godunov simulation')

godmodel = god.Godunov(ff_speed, wave_speed,
                       jam_density, simu_timestep,
                       from_x, to_x, begin_t, end_t,
                       inflow, outflow,
                       downstream_cap, upstream_den, alpha, beta)
