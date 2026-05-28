import numpy as np
import scipy.io as sio
import inspect

# Parameters
d = 0.001
x_min, x_max = -1.0, 1.0
t_min, t_max = 0.0, 1.0

Nx = 201
Nt = 101

# Grids
x = np.linspace(x_min, x_max, Nx)
t = np.linspace(t_min, t_max, Nt)

dx = x[1] - x[0]
dt = t[1] - t[0]

# Solution array
u = np.zeros((Nt, Nx))

# different initial condition functions

def step_function(x):
    return np.where(x < 0, -1, 1)

def wave_1(x):
    return x**2 * np.cos(np.pi * x)

def tanh_function(x, eps=0.1):
    return np.tanh(x/(np.sqrt(2)*eps))

eps_values = [0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 2.0]

for epsilon in eps_values:
    # initial condition with different epsilon values for the tanh function
    u[0, :] = tanh_function(x, eps=epsilon)

    u_xx = np.zeros_like(u)
    # Time stepping
    for n in range(Nt - 1):
        u_xx = np.zeros(Nx)

        # interior second derivative
        u_xx[1:-1] = (u[n, 2:] - 2*u[n, 1:-1] + u[n, :-2]) / dx**2

        # Neumann BC through ghost-point Laplacian. A ghost point value is a point outside the domain, 
        # in this case we assume it has the same value as the point at the boundary due to neumann BC, 
        # so the second derivative at the boundary can be approximated as 2*(u[n, 1] - u[n, 0]) / dx^2 
        # for the left boundary and similarly for the right boundary.
        u_xx[0] = 2 * (u[n, 1] - u[n, 0]) / dx**2
        u_xx[-1] = 2 * (u[n, -2] - u[n, -1]) / dx**2

        reaction = 5*u[n, :] - 5*u[n, :]**3

        u[n+1, :] = u[n, :] + dt * (d*u_xx + reaction)



    # reshape for MATLAB format
    t_mat = t.reshape(1, -1)
    x_mat = x.reshape(1, -1)

    # calculate different measures of curvature for the solution
    u_xx = np.zeros_like(u)
    for n in range(Nt):
        for i in range(1, Nx - 1):
            u_xx[n, i] = (u[n, i+1] - 2*u[n, i] + u[n, i-1]) / dx**2

    u_tt = np.zeros_like(u)
    for n in range(1, Nt - 1):
        for i in range(Nx):
            u_tt[n, i] = (u[n+1, i] - 2*u[n, i] + u[n-1, i]) / dt**2

    dict_data={
        "t": t_mat,
        "x": x_mat,
        "u": u,
        "u_xx": u_xx,
        "u_tt": u_tt,
        "tanh_initial_condition": f"np.tanh(x/(np.sqrt(2)*{epsilon}))"
    }
    # Save to .mat file
    def save_to_mat(filename, dict_data):
        sio.savemat(filename, dict_data)

    save_to_mat(f"Allen_Cahn_tanh_{epsilon}.mat", dict_data)

    print(f"Saved Allen_Cahn_tanh_{epsilon}.mat")
    print("t shape:", t_mat.shape)
    print("x shape:", x_mat.shape)
    print("u shape:", u.shape)
