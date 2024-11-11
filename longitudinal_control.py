import numpy as np
import matplotlib.pyplot as plt

params = {
        "rho": 1.293,
        "C_d": 0.23,
        "A": 2.22,
        "f_roll": 0.015,
        "m": 1611,
        "g": 9.8,
        "F_trac_max": 10000,
    }

# define reference trajectory
v_ref = np.ones(2*60) * 5

# initialize variables
F_trac = np.zeros(len(v_ref))
error = np.zeros(len(v_ref))
v = np.zeros(len(v_ref))
dt_s = 1

kp = 0.01
ki = 0.1
kd = 0

u_p = np.zeros(len(v_ref))
u_i = np.zeros(len(v_ref))
u_d = np.zeros(len(v_ref))
u_pid = np.zeros(len(v_ref))
theta_acc = np.zeros(len(v_ref))
theta_brake = np.zeros(len(v_ref))

for i in range(len(v_ref)-1):
    # compute controller input
    u_p[i] = kp * error[i]
    
    if i > 0:
        u_i[i] = u_i[i-1] + ki * error[i]
    else:
        u_i[i] = ki * error[i]

    if i > 0:
        u_d[i] = kd * (error[i] * error[i-1])/dt_s
    else:
        u_d[i] = 0

    u_pid[i] = max(-1,min(1,u_p[i] + u_i[i] + u_d[i]))

    # scale based on maximum force values
    F_trac[i] = u_pid[i] * params["F_trac_max"]

    # compute trajectory step based on controller input
    F_air = 0.5*params["rho"] * params["C_d"]*params["A"] *v[i]**2
    F_grade = 0
    F_roll = params["f_roll"]*params["m"]*params["g"]*v[i]
    F_acc = F_trac[i] - F_air - F_roll - F_grade

    a = F_acc / params["m"]

    if a >= 0:
        theta_acc[i+1] = a
    elif a < 0:
        theta_brake[i+1] = -1 * a

    v[i+1] = v[i] + a * dt_s
    
    # compute error
    error[i+1] = v_ref[i+1] - v[i+1]

# plot data
fig, axes = plt.subplots(2, 2)
plt.tight_layout()
axes[0,0].plot(v_ref, label='v_ref')
axes[0,0].plot(v, label='v')
axes[0,0].set_ylabel("speed (m/s)")
axes[0,0].set_xlabel("time (s)")
axes[0,0].legend()

plt.tight_layout()
axes[1,0].plot(u_p, label='u_p')
axes[1,0].plot(u_i, label='u_i')
axes[1,0].plot(u_d, label='u_d')
axes[1,0].plot(u_pid, label='u_pid')

axes[1,0].set_ylabel("input")
axes[1,0].set_xlabel("time (s)")
axes[1,0].legend()

plt.tight_layout()
axes[0,1].plot(F_trac, label='F_trac')
axes[0,1].set_ylabel("Force (N)")
axes[0,1].set_xlabel("time (s)")
axes[0,1].legend()

ax4 = plt.subplot(222)
plt.tight_layout()
axes[1,1].plot(theta_acc, label='theta_acc')
axes[1,1].plot(theta_brake, label='theta_brake')
axes[1,1].set_ylabel("Pedal angles")
axes[1,1].set_xlabel("times (s)")
axes[1,1].legend()
plt.show()