import numpy as np
import scipy.io as sio

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

# Initial condition
u[0, :] = x**2 * np.cos(np.pi * x)

# Boundary conditions
u[:, 0] = -1
u[:, -1] = -1

# Time stepping
for n in range(Nt - 1):
    for i in range(1, Nx - 1):

        u_xx = (u[n, i+1] - 2*u[n, i] + u[n, i-1]) / dx**2

        reaction = 5*u[n, i] - 5*u[n, i]**3

        u[n+1, i] = u[n, i] + dt*(d*u_xx + reaction)

    # enforce boundary conditions
    u[n+1, 0] = -1
    u[n+1, -1] = -1

# reshape for MATLAB format
t_mat = t.reshape(1, -1)
x_mat = x.reshape(1, -1)

# Save to .mat file
sio.savemat("Allen_Cahn_generated.mat", {
    "t": t_mat,
    "x": x_mat,
    "u": u
})

print("Saved Allen_Cahn_generated.mat")
print("t shape:", t_mat.shape)
print("x shape:", x_mat.shape)
print("u shape:", u.shape)
