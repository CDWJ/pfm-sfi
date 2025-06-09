# 
from hyperparameters import *
from taichi_utils import *
from mgpcg import *
from init_conditions import *
from io_utils import *
import torch
import sys
import shutil
import time
#
ti.init(arch=ti.cuda, device_memory_GB=10.5, debug = False)

dx = 1./res_y
from ibm_cloth import *
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

# uniform distribute particles
particles_per_cell_axis = 2
dist_between_neighbor = dx / particles_per_cell_axis

one_sixth = 1. / 6

# solver
boundary_mask = ti.field(ti.i32, shape=(res_x, res_y, res_z))
boundary_vel = ti.Vector.field(3, float, shape=(res_x, res_y, res_z))
boundary_types = ti.Matrix([[2, 1], [1, 1], [1, 1]], ti.i32) # boundaries: 1 means Dirichlet, 2 means Neumann
solver = MGPCG_3(boundary_types, N = [res_x, res_y, res_z], base_level=3)

# undeformed coordinates (cell center and faces)
X = ti.Vector.field(3, float, shape=(res_x, res_y, res_z))
X_x = ti.Vector.field(3, float, shape=(res_x+1, res_y, res_z))
X_y = ti.Vector.field(3, float, shape=(res_x, res_y+1, res_z))
X_z = ti.Vector.field(3, float, shape=(res_x, res_y, res_z+1))
center_coords_func(X, dx)
x_coords_func(X_x, dx)
y_coords_func(X_y, dx)
z_coords_func(X_z, dx)

# particle storage
initial_particle_num = (res_x + 10) * (res_y + 10) * (res_z + 10) * particles_per_cell
# initial_particle_num = (res_x) * (res_y) * (res_z) * particles_per_cell
particle_num = initial_particle_num * total_particles_num_ratio
# current_particle_num = ti.field(int, shape=1)
# particles_active = ti.field(float, shape=particle_num)
particles_pos = ti.Vector.field(3, float, shape=particle_num)
# particles_pos_backup = ti.Vector.field(3, float, shape=particle_num)
# particles_vel = ti.Vector.field(3, float, shape=particle_num)
particles_imp = ti.Vector.field(3, float, shape=particle_num)
particles_init_imp = ti.Vector.field(3, float, shape=particle_num)
# particles_init_imp_grad_m = ti.Vector.field(3, float, shape=particle_num)
particles_smoke = ti.Vector.field(4, float, shape=particle_num)
# particles_grad_smoke = ti.Matrix.field(4, 3, float, shape=particle_num)
particles_init_grad_smoke = ti.Matrix.field(3, 4, float, shape=particle_num)

# added for zhiqi's method
particles_lamb = ti.field(float, shape=particle_num)
particles_grad_lamb = ti.Vector.field(3, float, shape=particle_num)

particles_half_usquare = ti.field(float, shape=particle_num)
particles_grad_half_usquare = ti.Vector.field(3, float, shape=particle_num)

# back flow map
T_x = ti.Vector.field(3, float, shape=particle_num) # d_psi / d_x
T_y = ti.Vector.field(3, float, shape=particle_num) # d_psi / d_y
T_z = ti.Vector.field(3, float, shape=particle_num) # d_psi / d_z

F_x = ti.Vector.field(3, float, shape=particle_num) # d_psi / d_x
F_y = ti.Vector.field(3, float, shape=particle_num) # d_psi / d_y
F_z = ti.Vector.field(3, float, shape=particle_num) # d_psi / d_z

# T_x_grad_m = ti.Vector.field(3, float, shape=particle_num) # d_psi / d_x
# T_y_grad_m = ti.Vector.field(3, float, shape=particle_num) # d_psi / d_y
# T_z_grad_m = ti.Vector.field(3, float, shape=particle_num) # d_psi / d_z
# psi = ti.Vector.field(3, float, shape=particle_num) # x coordinate
# grad_T_x = ti.Matrix.field(3, 3, float, shape=particle_num)
# grad_T_y = ti.Matrix.field(3, 3, float, shape=particle_num)
# grad_T_z = ti.Matrix.field(3, 3, float, shape=particle_num)
# grad_T_init_x = ti.Matrix.field(3, 3, float, shape=particle_num)
# grad_T_init_y = ti.Matrix.field(3, 3, float, shape=particle_num)
# grad_T_init_z = ti.Matrix.field(3, 3, float, shape=particle_num)


# velocity storage
u = ti.Vector.field(3, float, shape=(res_x, res_y, res_z))
w = ti.Vector.field(3, float, shape=(res_x, res_y, res_z)) # curl of u
u_x = ti.field(float, shape=(res_x+1, res_y, res_z))
u_y = ti.field(float, shape=(res_x, res_y+1, res_z))
u_z = ti.field(float, shape=(res_x, res_y, res_z+1))
u_square = ti.field(float, shape=(res_x, res_y, res_z))

# gradm_x_grid = ti.Vector.field(2, float, shape=(res_x + 1, res_y, res_z))  # d_psi / d_x
# gradm_y_grid = ti.Vector.field(2, float, shape=(res_x, res_y + 1, res_z))  # d_psi / d_y
# gradm_z_grid = ti.Vector.field(2, float, shape=(res_x, res_y, res_z + 1))  # d_psi / d_z

# gradm_x = ti.Vector.field(2, float, shape=particle_num)
# gradm_y = ti.Vector.field(2, float, shape=particle_num)
# gradm_z = ti.Vector.field(2, float, shape=particle_num)

# P2G weight storage
p2g_weight = ti.field(float, shape=(res_x, res_y, res_z))
p2g_weight_x = ti.field(float, shape=(res_x + 1, res_y, res_z))
p2g_weight_y = ti.field(float, shape=(res_x, res_y + 1, res_z))
p2g_weight_z = ti.field(float, shape=(res_x, res_y, res_z + 1))

# impulse storage
# imp_x = ti.field(float, shape=(res_x + 1, res_y, res_z))
# imp_y = ti.field(float, shape=(res_x, res_y + 1, res_z))
# imp_z = ti.field(float, shape=(res_x, res_y, res_z + 1))

# APIC
C_x = ti.Vector.field(3, float, shape=particle_num)
C_y = ti.Vector.field(3, float, shape=particle_num)
C_z = ti.Vector.field(3, float, shape=particle_num)


# some helper storage
# init_u_x = ti.field(float, shape=(res_x+1, res_y, res_z)) # stores the "m0"
# init_u_y = ti.field(float, shape=(res_x, res_y+1, res_z))
# init_u_z = ti.field(float, shape=(res_x, res_y, res_z+1))
# err_u_x = ti.field(float, shape=(res_x+1, res_y, res_z)) # stores the roundtrip "m0"
# err_u_y = ti.field(float, shape=(res_x, res_y+1, res_z))
# err_u_z = ti.field(float, shape=(res_x, res_y, res_z+1))
# tmp_u_x = ti.field(float, shape=(res_x+1, res_y, res_z))
# tmp_u_y = ti.field(float, shape=(res_x, res_y+1, res_z))
# tmp_u_z = ti.field(float, shape=(res_x, res_y, res_z+1))

# CFL related
max_speed = ti.field(float, shape=())
# dts = torch.zeros(reinit_every)
total_t = ti.field(float, shape=())
# smoke
# init_smoke = ti.Vector.field(4, float, shape=(res_x, res_y, res_z))
smoke = ti.Vector.field(4, float, shape=(res_x, res_y, res_z))
tmp_smoke = ti.Vector.field(4, float, shape=(res_x, res_y, res_z))
# err_smoke = ti.Vector.field(4, float, shape=(res_x, res_y, res_z))

# display for levels
# levels_display = np.zeros((8 * res_x, 8 * res_y), dtype=np.float32)

# neural buffer
# nb = NeuralBuffer3(res_x = res_x, res_y = res_y, res_z = res_z, dx = dx)

@ti.kernel
def calc_max_speed(u_x: ti.template(), u_y: ti.template(), u_z: ti.template()):
    max_speed[None] = 1.e-3 # avoid dividing by zero
    for i, j, k in ti.ndrange(res_x, res_y, res_z):
        u = 0.5 * (u_x[i, j, k] + u_x[i+1, j, k])
        v = 0.5 * (u_y[i, j, k] + u_y[i, j+1, k])
        w = 0.5 * (u_z[i, j, k] + u_z[i, j, k+1])
        speed = ti.sqrt(u ** 2 + v ** 2 + w ** 2)
        ti.atomic_max(max_speed[None], speed)

# set to undeformed config
@ti.kernel
def reset_to_identity_grid(psi_x: ti.template(), psi_y: ti.template(), psi_z: ti.template(),
                    T_x: ti.template(), T_y: ti.template(), T_z: ti.template()):
    for I in ti.grouped(psi_x):
        psi_x[I] = X_x[I]
    for I in ti.grouped(psi_y):
        psi_y[I] = X_y[I]
    for I in ti.grouped(psi_z):
        psi_z[I] = X_z[I]
    for I in ti.grouped(T_x):
        T_x[I] = ti.Vector.unit(3, 0)
    for I in ti.grouped(T_y):
        T_y[I] = ti.Vector.unit(3, 1)
    for I in ti.grouped(T_z):
        T_z[I] = ti.Vector.unit(3, 2)

@ti.kernel
def reset_to_identity(psi: ti.template(), T_x: ti.template(), T_y: ti.template(), T_z: ti.template()):
    for i in psi:
        psi[i] = particles_pos[i]
    for I in ti.grouped(T_x):
        T_x[I] = ti.Vector.unit(3, 0)
    for I in ti.grouped(T_y):
        T_y[I] = ti.Vector.unit(3, 1)
    for I in ti.grouped(T_z):
        T_z[I] = ti.Vector.unit(3, 2)

@ti.kernel
def reset_T_to_identity(T_x: ti.template(), T_y: ti.template(), T_z: ti.template()):
    for I in ti.grouped(T_x):
        T_x[I] = ti.Vector.unit(3, 0)
    for I in ti.grouped(T_y):
        T_y[I] = ti.Vector.unit(3, 1)
    for I in ti.grouped(T_z):
        T_z[I] = ti.Vector.unit(3, 2)

@ti.kernel
def reset_F_to_identity(F_x: ti.template(), F_y: ti.template(), F_z: ti.template()):
    for I in ti.grouped(F_x):
        F_x[I] = ti.Vector.unit(3, 0)
    for I in ti.grouped(F_y):
        F_y[I] = ti.Vector.unit(3, 1)
    for I in ti.grouped(F_z):
        F_z[I] = ti.Vector.unit(3, 2)

# curr step should be in range(reinit_every)

@ti.kernel
def RK4_grid(psi_x: ti.template(), T_x: ti.template(), 
            u_x0: ti.template(), u_y0: ti.template(), u_z0: ti.template(), dt: float):

    neg_dt = -1 * dt # travel back in time
    for I in ti.grouped(psi_x):
        # first
        u1, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x[I], dx)
        dT_x_dt1 = grad_u_at_psi @ T_x[I] # time derivative of T
        # prepare second
        psi_x1 = psi_x[I] + 0.5 * neg_dt * u1 # advance 0.5 steps
        T_x1 = T_x[I] + 0.5 * neg_dt * dT_x_dt1
        # second
        u2, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x1, dx)
        dT_x_dt2 = grad_u_at_psi @ T_x1 # time derivative of T
        # prepare third
        psi_x2 = psi_x[I] + 0.5 * neg_dt * u2 # advance 0.5 again
        T_x2 = T_x[I] + 0.5 * neg_dt * dT_x_dt2 
        # third
        u3, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x2, dx)
        dT_x_dt3 = grad_u_at_psi @ T_x2 # time derivative of T
        # prepare fourth
        psi_x3 = psi_x[I] + 1.0 * neg_dt * u3
        T_x3 = T_x[I] + 1.0 * neg_dt * dT_x_dt3 # advance 1.0
        # fourth
        u4, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x3, dx)
        dT_x_dt4 = grad_u_at_psi @ T_x3 # time derivative of T
        # final advance
        psi_x[I] = psi_x[I] + neg_dt * 1./6 * (u1 + 2 * u2 + 2 * u3 + u4)
        T_x[I] = T_x[I] + neg_dt * 1./6 * (dT_x_dt1 + 2 * dT_x_dt2 + 2 * dT_x_dt3 + dT_x_dt4) # advance full

@ti.kernel
def RK4_T_forward(psi: ti.template(), T_x: ti.template(), T_y: ti.template(), T_z: ti.template(),
                  F_x: ti.template(), F_y: ti.template(), F_z: ti.template(), u_x0: ti.template(), u_y0: ti.template(), u_z0: ti.template(), dt: float):
    for i in psi:
        # if particles_active[i] == 1:
            # first
            u1, grad_u_at_psi = interp_u_MAC_grad_transpose(u_x0, u_y0, u_z0, psi[i], dx)
            dT_x_dt1 = grad_u_at_psi @ T_x[i]  # time derivative of T
            dT_y_dt1 = grad_u_at_psi @ T_y[i]  # time derivative of T
            dT_z_dt1 = grad_u_at_psi @ T_z[i]  # time derivative of T
            
            dF_x_dt1 = grad_u_at_psi.transpose() @ F_x[i]  # time derivative of F
            dF_y_dt1 = grad_u_at_psi.transpose() @ F_y[i]  # time derivative of F
            dF_z_dt1 = grad_u_at_psi.transpose() @ F_z[i]  # time derivative of F
            # prepare second
            psi_x1 = psi[i] + 0.5 * dt * u1  # advance 0.5 steps
            T_x1 = T_x[i] - 0.5 * dt * dT_x_dt1
            T_y1 = T_y[i] - 0.5 * dt * dT_y_dt1
            T_z1 = T_z[i] - 0.5 * dt * dT_z_dt1
            
            F_x1 = F_x[i] + 0.5 * dt * dF_x_dt1
            F_y1 = F_y[i] + 0.5 * dt * dF_y_dt1
            F_z1 = F_z[i] + 0.5 * dt * dF_z_dt1
            # second
            u2, grad_u_at_psi = interp_u_MAC_grad_transpose(u_x0, u_y0, u_z0, psi_x1, dx)
            dT_x_dt2 = grad_u_at_psi @ T_x1  # time derivative of T
            dT_y_dt2 = grad_u_at_psi @ T_y1  # time derivative of T
            dT_z_dt2 = grad_u_at_psi @ T_z1  # time derivative of T
            
            dF_x_dt2 = grad_u_at_psi.transpose() @ F_x1  # time derivative of F
            dF_y_dt2 = grad_u_at_psi.transpose() @ F_y1  # time derivative of F
            dF_z_dt2 = grad_u_at_psi.transpose() @ F_z1  # time derivative of F
            # prepare third
            psi_x2 = psi[i] + 0.5 * dt * u2  # advance 0.5 again
            T_x2 = T_x[i] - 0.5 * dt * dT_x_dt2
            T_y2 = T_y[i] - 0.5 * dt * dT_y_dt2
            T_z2 = T_z[i] - 0.5 * dt * dT_z_dt2
            
            F_x2 = F_x[i] + 0.5 * dt * dF_x_dt2
            F_y2 = F_y[i] + 0.5 * dt * dF_y_dt2
            F_z2 = F_z[i] + 0.5 * dt * dF_z_dt2
            # third
            u3, grad_u_at_psi = interp_u_MAC_grad_transpose(u_x0, u_y0, u_z0, psi_x2, dx)
            dT_x_dt3 = grad_u_at_psi @ T_x2  # time derivative of T
            dT_y_dt3 = grad_u_at_psi @ T_y2  # time derivative of T
            dT_z_dt3 = grad_u_at_psi @ T_z2  # time derivative of T
            
            dF_x_dt3 = grad_u_at_psi.transpose() @ F_x2  # time derivative of F
            dF_y_dt3 = grad_u_at_psi.transpose() @ F_y2  # time derivative of F
            dF_z_dt3 = grad_u_at_psi.transpose() @ F_z2  # time derivative of F
            # prepare fourth
            psi_x3 = psi[i] + 1.0 * dt * u3
            T_x3 = T_x[i] - 1.0 * dt * dT_x_dt3  # advance 1.0
            T_y3 = T_y[i] - 1.0 * dt * dT_y_dt3  # advance 1.0
            T_z3 = T_z[i] - 1.0 * dt * dT_z_dt3  # advance 1.0
            
            F_x3 = F_x[i] + 1.0 * dt * dF_x_dt3
            F_y3 = F_y[i] + 1.0 * dt * dF_y_dt3
            F_z3 = F_z[i] + 1.0 * dt * dF_z_dt3
            # fourth
            u4, grad_u_at_psi = interp_u_MAC_grad_transpose(u_x0, u_y0, u_z0, psi_x3, dx)
            dT_x_dt4 = grad_u_at_psi @ T_x3  # time derivative of T
            dT_y_dt4 = grad_u_at_psi @ T_y3  # time derivative of T
            dT_z_dt4 = grad_u_at_psi @ T_z3  # time derivative of T
            
            dF_x_dt4 = grad_u_at_psi.transpose() @ F_x3  # time derivative of F
            dF_y_dt4 = grad_u_at_psi.transpose() @ F_y3  # time derivative of F
            dF_z_dt4 = grad_u_at_psi.transpose() @ F_z3  # time derivative of F
            # final advance
            psi[i] = psi[i] + dt * 1. / 6 * (u1 + 2 * u2 + 2 * u3 + u4)
            T_x[i] = T_x[i] - dt * 1. / 6 * (dT_x_dt1 + 2 * dT_x_dt2 + 2 * dT_x_dt3 + dT_x_dt4)  # advance full
            T_y[i] = T_y[i] - dt * 1. / 6 * (dT_y_dt1 + 2 * dT_y_dt2 + 2 * dT_y_dt3 + dT_y_dt4)  # advance full
            T_z[i] = T_z[i] - dt * 1. / 6 * (dT_z_dt1 + 2 * dT_z_dt2 + 2 * dT_z_dt3 + dT_z_dt4)  # advance full
            
            F_x[i] = F_x[i] + dt * 1. / 6 * (dF_x_dt1 + 2 * dF_x_dt2 + 2 * dF_x_dt3 + dF_x_dt4)  # advance full
            F_y[i] = F_y[i] + dt * 1. / 6 * (dF_y_dt1 + 2 * dF_y_dt2 + 2 * dF_y_dt3 + dF_y_dt4)  # advance full
            F_z[i] = F_z[i] + dt * 1. / 6 * (dF_z_dt1 + 2 * dF_z_dt2 + 2 * dF_z_dt3 + dF_z_dt4)  # advance full


# u_x0, u_y0, u_z0 are the initial time quantities
# u_x1, u_y1, u_z0 are the current time quantities (to be modified)
@ti.kernel
def advect_u(u_x0: ti.template(), u_y0: ti.template(), u_z0: ti.template(),
            u_x1: ti.template(), u_y1: ti.template(), u_z1: ti.template(),
            T_x: ti.template(), T_y: ti.template(), T_z: ti.template(),
            psi_x: ti.template(), psi_y: ti.template(), psi_z: ti.template(), dx: float):
    # x velocity
    for I in ti.grouped(u_x1):
        u_at_psi, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x[I], dx)
        u_x1[I] = T_x[I].dot(u_at_psi)
    # y velocity
    for I in ti.grouped(u_y1):
        u_at_psi, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_y[I], dx)
        u_y1[I] = T_y[I].dot(u_at_psi)
    # z velocity
    for I in ti.grouped(u_z1):
        u_at_psi, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_z[I], dx)
        u_z1[I] = T_z[I].dot(u_at_psi)
        
        
@ti.kernel
def advect_u_grid(u_x0: ti.template(), u_y0: ti.template(), u_z0: ti.template(),
             u_x1: ti.template(), u_y1: ti.template(), u_z1: ti.template(), dx : float, dt : float):
    for I in ti.grouped(u_x1):
        p1 = X_x[I]
        v1, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p1, dx)
        p2 = p1 - v1 * dt * 0.5
        v2, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p2, dx)
        p3 = p1 - v2 * dt * 0.5
        v3, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p3, dx)
        p4 = p1 - v3 * dt
        v4, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p4, dx)
        p = p1 - (v1 + 2 * v2 + 2 * v3 + v4) / 6.0 * dt
        v5, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p, dx)
        u_x1[I] = ((v1 + 2 * v2 + 2 * v3 + v4) / 6.0)[0]
        # u_x1[I] = v5[0]
        
    for I in ti.grouped(u_y1):
        p1 = X_y[I]
        v1, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p1, dx)
        p2 = p1 - v1 * dt * 0.5
        v2, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p2, dx)
        p3 = p1 - v2 * dt * 0.5
        v3, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p3, dx)
        p4 = p1 - v3 * dt
        v4, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p4, dx)
        p = p1 - (v1 + 2 * v2 + 2 * v3 + v4) / 6.0 * dt
        v5, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p, dx)
        u_y1[I] = ((v1 + 2 * v2 + 2 * v3 + v4) / 6.0)[1]
        # u_y1[I] = v5[1]
        
    for I in ti.grouped(u_z1):
        p1 = X_z[I]
        v1, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p1, dx)
        p2 = p1 - v1 * dt * 0.5
        v2, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p2, dx)
        p3 = p1 - v2 * dt * 0.5
        v3, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p3, dx)
        p4 = p1 - v3 * dt
        v4, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p4, dx)
        p = p1 - (v1 + 2 * v2 + 2 * v3 + v4) / 6.0 * dt
        v5, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p, dx)
        u_z1[I] = ((v1 + 2 * v2 + 2 * v3 + v4) / 6.0)[2]
        # u_z1[I] = v5[2]


@ti.kernel
def advect_smoke(smoke0: ti.template(), smoke1: ti.template(), 
            psi_x: ti.template(), psi_y: ti.template(), psi_z: ti.template(), dx: float):
    # horizontal velocity
    for i, j, k in ti.ndrange(res_x, res_y, res_z):
        psi_c = 1./6 * (psi_x[i, j, k] + psi_x[i+1, j, k] + \
                        psi_y[i, j, k] + psi_y[i, j+1, k] + \
                        psi_z[i, j, k] + psi_z[i, j, k+1])
        smoke1[i,j,k] = interp_1(smoke0, psi_c, dx)

@ti.kernel
def clamp_smoke(smoke0: ti.template(), smoke1: ti.template(),
            psi_x: ti.template(), psi_y: ti.template(), psi_z: ti.template(), dx: float):
    # horizontal velocity
    for i, j, k in ti.ndrange(res_x, res_y, res_z):
        psi_c = 1./6 * (psi_x[i, j, k] + psi_x[i+1, j, k] + \
                        psi_y[i, j, k] + psi_y[i, j+1, k] + \
                        psi_z[i, j, k] + psi_z[i, j, k+1])
        mini, maxi = sample_min_max_1(smoke0, psi_c, dx)
        smoke1[i,j,k] = ti.math.clamp(smoke1[i,j,k], mini, maxi)

def init_vorts():
    init_vorts_oblique(X, u, smoke, tmp_smoke)
    # init_vorts_leapfrog(X, u)
    #init_vorts_headon(X, u)

def stretch_T_and_advect_particles(particles_pos, T_x, T_y, T_z, F_x, F_y, F_z, u_x, u_y, u_z, dt):
    RK4_T_forward(particles_pos, T_x, T_y, T_z, F_x, F_y, F_z, u_x, u_y, u_z, dt)

@ti.kernel
def update_particles_imp(particles_imp: ti.template(), particles_init_imp: ti.template(), grad_lamb: ti.template(), grad_half_usquare: ti.template(), T_x: ti.template(), T_y: ti.template(), T_z: ti.template()):
    for i in particles_imp:
        # if particles_active[i] == 1:
            T = ti.Matrix.cols([T_x[i], T_y[i], T_z[i]])
            # if forward_update_T:
            #     T = T.transpose()
            particles_imp[i] = T @ (particles_init_imp[i] - grad_lamb[i]) + grad_half_usquare[i]
@ti.kernel
def update_particles_imp_add_visc(particles_imp: ti.template(), particles_init_imp: ti.template(), grad_lamb: ti.template(), grad_half_usquare: ti.template(), T_x: ti.template(), T_y: ti.template(), curr_dt: ti.template()):
    for i in particles_imp:
        if particles_active[i] == 1:
            T = ti.Matrix.cols([T_x[i], T_y[i], T_z[i]])
            particles_imp[i] = T @ particles_init_imp[i]          
@ti.kernel
def update_particles_grad_smoke(particles_grad_smoke: ti.template(), particles_init_grad_smoke: ti.template(),
                                T_x: ti.template(), T_y: ti.template(), T_z: ti.template()):
    for i in particles_grad_smoke:
        # if particles_active[i] == 1:
            T = ti.Matrix.cols([T_x[i], T_y[i], T_z[i]])
            particles_grad_smoke[i] = (T @ particles_init_grad_smoke[i].transpose()).transpose()
            
            
            
@ti.kernel
def compute_particles_half_usquare_and_grad_u(particles_pos: ti.template(), particles_half_usquare: ti.template(), u_x: ti.template(), u_y: ti.template(), u_z:ti.template()):
    C_x.fill(0.0)
    C_y.fill(0.0)
    C_z.fill(0.0)
    for i in particles_half_usquare:
        # if particles_active[i] >= 1:
            p = particles_pos[i]

            u_x_p, grad_u_x_p = interp_grad_2(u_x, p, dx, BL_x=0.0, BL_y=0.5, BL_z=0.5)
            u_y_p, grad_u_y_p = interp_grad_2(u_y, p, dx, BL_x=0.5, BL_y=0.0, BL_z=0.5)
            u_z_p, grad_u_z_p = interp_grad_2(u_z, p, dx, BL_x=0.5, BL_y=0.5, BL_z=0.0)
            
            C_x[i] = grad_u_x_p
            C_y[i] = grad_u_y_p
            C_z[i] = grad_u_z_p
            
        
@ti.kernel
def accumulate_lamb(particles_grad_lamb: ti.template(), particles_pos: ti.template(), particles_grad_half_u: ti.template(), pressure: ti.template(), u: ti.template(), curr_dt: ti.template(), prev_dt: ti.template()):
    for i in particles_grad_half_u:
        p = particles_pos[i]
        nouse, grad_u = interp_grad_2(u, p, dx, BL_x=0.5, BL_y=0.5, BL_z=0.5)
        particles_grad_half_u[i] = grad_u * curr_dt
        
    # for I in ti.grouped(u):
    #     u[I] = pressure[I] - u[I]
        
    for i in particles_grad_lamb:
        p = particles_pos[i]
        nouse, grad_p = interp_grad_2(pressure, p, dx, BL_x=0.5, BL_y=0.5, BL_z=0.5)
        nouse, grad_u = interp_grad_2(u, p, dx, BL_x=0.5, BL_y=0.5, BL_z=0.5)
        F = ti.Matrix.rows([F_x[i], F_y[i], F_z[i]])
        particles_grad_lamb[i] += F @ (grad_p - grad_u) * prev_dt
            
            
@ti.kernel
def accumulate_lamb_j(particles_grad_lamb: ti.template(), particles_pos: ti.template(), particles_grad_half_u: ti.template(), pressure: ti.template(), u: ti.template(), curr_dt: ti.template()):
    for i in particles_grad_half_u:
        p = particles_pos[i]
        nouse, grad_u = interp_grad_2(u, p, dx, BL_x=0.5, BL_y=0.5, BL_z=0.5)
        particles_grad_half_u[i] = grad_u * curr_dt


@ti.kernel
def P2G(particles_pos: ti.template(), u_x: ti.template(), u_y: ti.template(), u_z: ti.template(), p2g_weight_x: ti.template(), p2g_weight_y: ti.template(), p2g_weight_z: ti.template(), T_x: ti.template(), T_y: ti.template(), T_z: ti.template(), curr_dt:float):
    u_x.fill(0.0)
    u_y.fill(0.0)
    u_z.fill(0.0)

    p2g_weight_x.fill(0.0)
    p2g_weight_y.fill(0.0)
    p2g_weight_z.fill(0.0)


    for i in particles_imp:

        ### real P2G ###
        # horizontal impulse
        pos = particles_pos[i] / dx
        # base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0)) + ti.Vector.unit(dim, 0)
        base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1) - 0.5 * ti.Vector.unit(dim, 2))
        for offset in ti.grouped(ti.ndrange(*((-1, 3),) * dim)):
            face_id = base_face_id + offset
            if 0 <= face_id[0] <= res_x and 0 <= face_id[1] < res_y and 0 <= face_id[2] < res_z:
                weight = N_2(pos[0] - face_id[0]) * N_2(pos[1] - face_id[1] - 0.5) * N_2(pos[2] - face_id[2] - 0.5)
                dpos = ti.Vector([face_id[0] - pos[0], face_id[1] + 0.5 - pos[1], face_id[2] + 0.5 - pos[2]]) * dx
                p2g_weight_x[face_id] += weight
                delta = C_x[i].dot(dpos)
                # print(particles_imp[i][0], weight, delta)
                if use_APIC:
                    u_x[face_id] += (particles_imp[i][0] + delta) * weight
                else:
                    u_x[face_id] += (particles_imp[i][0]) * weight

                # psi_x_grid[face_id] += (psi[i] + T.transpose() @ dpos) * weight

        # vertical impulse
        # base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1)) + ti.Vector.unit(dim, 1)
        base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0) - 0.5 * ti.Vector.unit(dim, 2))
        for offset in ti.grouped(ti.ndrange(*((-1, 3),) * dim)):
            face_id = base_face_id + offset
            if 0 <= face_id[0] < res_x and 0 <= face_id[1] <= res_y and 0 <= face_id[2] < res_z:
                weight = N_2(pos[0] - face_id[0] - 0.5) * N_2(pos[1] - face_id[1]) * N_2(pos[2] - face_id[2] - 0.5)
                dpos = ti.Vector([face_id[0] + 0.5 - pos[0], face_id[1] - pos[1], face_id[2] + 0.5 - pos[2]]) * dx
                p2g_weight_y[face_id] += weight
                delta = C_y[i].dot(dpos)
                if use_APIC:
                    u_y[face_id] += (particles_imp[i][1] + delta) * weight
                else:
                    u_y[face_id] += (particles_imp[i][1]) * weight

                # psi_y_grid[face_id] += (psi[i] + T.transpose() @ dpos) * weight

        base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0) - 0.5 * ti.Vector.unit(dim, 1))
        for offset in ti.grouped(ti.ndrange(*((-1, 3),) * dim)):
            face_id = base_face_id + offset
            if 0 <= face_id[0] < res_x and 0 <= face_id[1] < res_y and 0 <= face_id[2] <= res_z:
                weight = N_2(pos[0] - face_id[0] - 0.5) * N_2(pos[1] - face_id[1] - 0.5) * N_2(pos[2] - face_id[2])
                dpos = ti.Vector([face_id[0] + 0.5 - pos[0], face_id[1] + 0.5 - pos[1], face_id[2] - pos[2]]) * dx
                p2g_weight_z[face_id] += weight
                delta = C_z[i].dot(dpos)
                if use_APIC:
                    u_z[face_id] += (particles_imp[i][2] + delta) * weight
                else:
                    u_z[face_id] += (particles_imp[i][2]) * weight

                # psi_z_grid[face_id] += (psi[i] + T.transpose() @ dpos) * weight

    for I in ti.grouped(p2g_weight_x):
        if p2g_weight_x[I] > 0:
            u_x[I] /= p2g_weight_x[I]
            # u_x[I] += curr_dt * gravity
            # scale = 1. / p2g_weight_x[I]
            # u_x[I] *= scale
            # psi_x_grid[I] *= scale
            # imp_x[I] = u_x[I]

    for I in ti.grouped(p2g_weight_y):
        if p2g_weight_y[I] > 0:
            u_y[I] /= p2g_weight_y[I]
            # scale = 1. / p2g_weight_y[I]
            # u_y[I] *= scale
            # psi_y_grid[I] *= scale
            # imp_y[I] = u_y[I]

    for I in ti.grouped(p2g_weight_z):
        if p2g_weight_z[I] > 0:
            u_z[I] /= p2g_weight_z[I]
            # scale = 1. / p2g_weight_z[I]
            # u_z[I] *= scale
            # psi_z_grid[I] *= scale
            # imp_z[I] = u_z[I]

@ti.kernel
def P2G_smoke(particles_smoke: ti.template(), particles_init_grad_smoke: ti.template(), particles_pos: ti.template(),
              smoke: ti.template(),  T_x: ti.template(), T_y: ti.template(), T_z: ti.template(), p2g_weight: ti.template()):
    smoke.fill(0.)
    p2g_weight.fill(0.)

    for i in particles_smoke:
        ### update T ###
        T_transpose = ti.Matrix.cols([T_x[i], T_y[i], T_z[i]])

        ### real P2G_smoke ###
        pos = particles_pos[i] / dx
        # base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0)) + ti.Vector.unit(dim, 0)
        base_face_id = int(pos - 0.5)
        for offset in ti.grouped(ti.ndrange(*((-1, 3),) * dim)):
            face_id = base_face_id + offset
            if 0 <= face_id[0] < res_x and 0 <= face_id[1] < res_y and 0 <= face_id[2] < res_z:
                weight = N_2(pos[0] - face_id[0] - 0.5) * N_2(pos[1] - face_id[1] - 0.5) * N_2(pos[2] - face_id[2] - 0.5)
                dpos = ti.Vector([face_id[0] + 0.5 - pos[0], face_id[1] + 0.5 - pos[1], face_id[2] + 0.5 - pos[2]]) * dx
                p2g_weight[face_id] += weight
                delta = dpos @ (T_transpose @ particles_init_grad_smoke[i])
                # print(particles_imp[i][0], weight, delta)
                if use_APIC_smoke:
                    smoke[face_id] += (particles_smoke[i] + delta) * weight
                else:
                    smoke[face_id] += particles_smoke[i] * weight

    for I in ti.grouped(p2g_weight):
        if p2g_weight[I] > 0:
            smoke[I] /= p2g_weight[I]
            for j in range(4):
                if smoke[I][j] < 0:
                    smoke[I][j] = 0.0
                if smoke[I][j] > 1:
                    smoke[I][j] = 1
            # scale = 1. / p2g_weight_z[I]
            # smoke[I] *= scale
            
def many_cylinders():
    x_gap = 0.2
    for i in range(9):
        z = 0.27
        x = x_gap * (i+1)
        cylinder_func(smoke, X, x, z)
    for i in range(10):
        z = 0.5
        x = x_gap * (i+0.5)
        if i != 8:
            cylinder_func(smoke, X, x, z)
    for i in range(9):
        z = 0.73
        x = x_gap * (i+1)
        cylinder_func(smoke, X, x, z)
        
        
# @ti.kernel
# def g2p_divergence_vis(particles_pos:ti.template(), particle_init_impulse:ti.template(), gradm_x_grid:ti.template(), gradm_y_grid:ti.template(), F_x: ti.template(), F_y:ti.template(), dt:float, vis : float, dx : float):
#     for I in particles_pos:
#         if particles_active[I] >= 1:
#             vis_force = interp_MAC_divergence(gradm_x_grid, gradm_y_grid, particles_pos[I], dx)
#             F = ti.Matrix.rows([F_x[I], F_y[I], F_z[I]])
#             particle_init_impulse[I] += (F@(vis_force * vis)) * dt
        
@ti.kernel
def mask_by_boundary(field: ti.template()):
    for I in ti.grouped(field):
        if boundary_mask[I] > 0:
            field[I] *= 0

            
@ti.kernel
def add_smoke_source(field : ti.template()):
    thickness = 0.1
    for I in ti.grouped(field):
        if I[2] <= 1:
            # p = int((I + 0.5) * dx / thickness)
            # if (p.x + p.y) % 2 > 0:
                # field[I] = ti.Vector([1.0, 1.0, 1.0, 1.0])
            # else:
            #     field[I] = ti.Vector([0.0, 0.0, 0.0, 1.0])
            field[I] = ti.Vector([1, 1, 1, 1])
@ti.kernel
def get_grad_usqure(u: ti.template(), u_square: ti.template()):
    particles_half_usquare.fill(0.0)
    for I in ti.grouped(u):
        u_square[I] = 0.5 * (u[I][0]**2 + u[I][1]**2 + u[I][2]**2)
        
@ti.kernel
def add_gravity(particle_init_impulse:ti.template(), F_x: ti.template(), F_y:ti.template(), F_z:ti.template(), dt:float, g : float, p_rho_w : float):
    for I in particle_init_impulse:
        # if particles_active[I] >= 1:
            F = ti.Matrix.rows([F_x[I], F_y[I], F_z[I]])
            particle_init_impulse[I] -= (F@(ti.Vector([1.0, 0.0, 0.0]) * g)) * dt
        
# main function
def main(from_frame = 0, testing = False):
    from_frame = max(0, from_frame)
    # create some folders
    logsdir = os.path.join('logs', exp_name)
    os.makedirs(logsdir, exist_ok=True)
    if from_frame <= 0:
        remove_everything_in(logsdir)
        
    # total_t = 0.0
    # moving_paddle_boundary_mask(boundary_mask, boundary_vel, total_t)
    add_smoke_source(smoke)
    print(boundary_vel.to_numpy().max())
    
    vtkdir = "vtks"
    vtkdir = os.path.join(logsdir, vtkdir)
    os.makedirs(vtkdir, exist_ok=True)
    vort2dir = 'vort_2D'
    vort2dir = os.path.join(logsdir, vort2dir)
    os.makedirs(vort2dir, exist_ok=True)
    smoke2dir = 'smoke_2D'
    smoke2dir = os.path.join(logsdir, smoke2dir)
    os.makedirs(smoke2dir, exist_ok=True)
    ckptdir = 'ckpts'
    ckptdir = os.path.join(logsdir, ckptdir)
    os.makedirs(ckptdir, exist_ok=True)
    levelsdir = 'levels'
    levelsdir = os.path.join(logsdir, levelsdir)
    os.makedirs(levelsdir, exist_ok=True)
    modeldir = 'model' # saves the model
    modeldir = os.path.join(logsdir, modeldir)
    os.makedirs(modeldir, exist_ok=True)
    soliddir = 'solid' # saves the model
    soliddir = os.path.join(logsdir, soliddir)
    os.makedirs(soliddir, exist_ok=True)

    shutil.copyfile('./hyperparameters.py', f'{logsdir}/hyperparameters.py')

    if testing:
        testdir = 'test_buffer'
        testdir = os.path.join(logsdir, testdir)
        os.makedirs(testdir, exist_ok=True)
        remove_everything_in(testdir)
        GTdir = os.path.join(testdir, "GT")
        os.makedirs(GTdir, exist_ok=True)
        preddir = os.path.join(testdir, "pred")
        os.makedirs(preddir, exist_ok=True)

    # initial condition
    if from_frame <= 0:
        u_x.fill(0.0)
        u_y.fill(0.0)
        u_z.fill(0.0)
        solver.Poisson(u_x, u_y, u_z, 0.0)
        # initialize smoke
        # stripe_func(smoke, X, 0.25, 0.48)
        mask_by_boundary(smoke)
            
    else:
        u_x.from_numpy(np.load(os.path.join(ckptdir, "vel_x_numpy_" + str(from_frame) + ".npy")))
        u_y.from_numpy(np.load(os.path.join(ckptdir, "vel_y_numpy_" + str(from_frame) + ".npy")))
        u_z.from_numpy(np.load(os.path.join(ckptdir, "vel_z_numpy_" + str(from_frame) + ".npy")))
        smoke.from_numpy(np.load(os.path.join(ckptdir, "smoke_numpy_" + str(from_frame) + ".npy")))

    # particles_active.fill(1)
    
    # xpbd.v_v.fill(0.1)
    # xpbd.v_v_cache.fill(0.1)

    # for visualization
    get_central_vector(u_x, u_y, u_z, u)
    curl(u, w, dx)
    w_numpy = w.to_numpy()
    w_norm = np.linalg.norm(w_numpy, axis = -1)
    w_max = max(np.abs(w_norm.max()), np.abs(w_norm.min()))
    w_min = -1 * w_max
    write_field(w_norm[:,:,res_z//2], vort2dir, from_frame, vmin = w_min, vmax = w_max)
    smoke_numpy = smoke.to_numpy()
    smoke_norm = smoke_numpy[...,-1]
    write_image(smoke_numpy[:,:,res_z//2][...,:3], smoke2dir, from_frame)
    write_w_and_smoke(w_norm, smoke_norm, vtkdir, from_frame)
    Export(soliddir, 0)

    # save init 
    # np.save(os.path.join(ckptdir, "vel_x_numpy_" + str(from_frame)), u_x.to_numpy())
    # np.save(os.path.join(ckptdir, "vel_y_numpy_" + str(from_frame)), u_y.to_numpy())
    # np.save(os.path.join(ckptdir, "vel_z_numpy_" + str(from_frame)), u_z.to_numpy())
    # np.save(os.path.join(ckptdir, "w_numpy_" + str(from_frame)), w_numpy)
    # np.save(os.path.join(ckptdir, "smoke_numpy_" + str(from_frame)), smoke_numpy)
    if save_particle_pos_numpy:
        np.save(os.path.join(ckptdir, "particles_pos_numpy_" + str(from_frame)), particles_pos.to_numpy())
    # done  

    sub_t = 0. # the time since last reinit
    frame_idx = from_frame
    last_output_substep = 0
    num_reinits = 0 # number of reinitializations already performed
    i = -1
    ik = 0
#     exp_name = 'C:/Users/duowe/OneDrive/Desktop/dartmouth/3D/logs/3D_four_vorts_reinit-12-4_smoke_evolved-grad-smoke/ckpts/'
    prev_dt = 0
    frame_times = np.zeros(total_steps)
    while True:
        start_time = time.time()
        i += 1
        j = i % reinit_every
        i_next = i + 1
        j_next = i_next % reinit_every
        print("[Simulate] Running step: ", i, " / substep: ", j)
        
        # moving_paddle_boundary_mask(boundary_mask, boundary_vel, total_t)
        curr_dt_solid_inuse = 0.00005
        # determine dt
        calc_max_speed(u_x, u_y, u_z) # saved to max_speed[None]
        curr_dt = CFL * dx / max_speed[None]
        solid_dt = curr_dt / 100
        if save_frame_each_step:
            output_frame = True
            frame_idx += 1
        else:
            if sub_t+curr_dt >= visualize_dt: # if over
                curr_dt = visualize_dt-sub_t
                sub_t = 0. # empty sub_t
                if frame_idx <= total_steps - 1:
                    print(f'frame execution time: {frame_times[frame_idx]:.6f} seconds')
                frame_idx += 1
                output_frame = True
                print(f'Visualized frame {frame_idx}')
            else:
                sub_t += curr_dt
                print(f'Visualize time {sub_t}/{visualize_dt}')
                output_frame = False
        # dts[j] = curr_dt
        total_t[None] += curr_dt
        solid_iters = int((int(curr_dt / curr_dt_solid_inuse) // 2) * 2 + 1)
        curr_dt_solid_inuse = curr_dt / (solid_iters * 1.0)
        # done dt

        # reinitialize flow map if j == 0:
        if j == 0:
            print("[Simulate] Reinitializing the flow map for the: ", num_reinits, " time!")

            # if reinit_particle_pos or i == 0:
            #     init_particles_pos_uniform(particles_pos, X, res_x, res_y, particles_per_cell, dx,
            #                                particles_per_cell_axis, dist_between_neighbor)
            #     ik = i
            init_particles_pos_uniform(particles_pos, X, res_x, res_y, particles_per_cell, dx, particles_per_cell_axis, dist_between_neighbor)
            init_particles_imp(particles_init_imp, particles_pos, u_x, u_y, u_z, C_x,
                               C_y, C_z, dx)
            init_particles_smoke(particles_smoke, particles_init_grad_smoke, particles_pos, smoke, dx)

            reset_T_to_identity(T_x, T_y, T_z)
            reset_F_to_identity(F_x, F_y, F_z)
            particles_grad_lamb.fill(0.0)
            # copy_to(smoke, init_smoke)


        # start midpoint
        # pointLocation_copy.copy_from(xpbd.v_p)
        # store_cache_vel()
        # xpbd.v_p_cache.copy_from(xpbd.v_p)
        pointLocation_copy.copy_from(xpbd.v_p)
        xpbd.v_p_cache.copy_from(xpbd.v_p)
        xpbd.const_v_v.fill(0.0)
        
        advect_ibm(u_x, u_y, u_z, curr_dt_solid_inuse, 0)
        
        mesh_vp_copy.copy_from(xpbd.v_p)
        mesh_vel_copy.copy_from(xpbd.v_v)
        
        store_cache_vel()
        if use_midpoint_vel:
            copy_to(u_x, p2g_weight_x)
            copy_to(u_y, p2g_weight_y)
            copy_to(u_z, p2g_weight_z)
            advect_ibm(u_x, u_y, u_z, curr_dt_solid_inuse, 0)
            
            store_cache_vel()
            for _ in range(solid_iters//2):
                xpbd.dt = curr_dt_solid_inuse
                xpbd.make_prediction()
                solve_for_xpbd(curr_dt_solid_inuse)
            advect_u_grid(p2g_weight_x, p2g_weight_y, p2g_weight_z, u_x, u_y, u_z, dx, 0.5 * curr_dt)
            update_force(curr_dt * 0.5)
            spread_force(u_x, u_y, u_z, curr_dt * 0.5)
            solver.time = total_t[None]
            solver.Poisson(u_x, u_y, u_z, total_t[None])
            
        store_cache_vel()
        for _ in range(solid_iters//2):
                xpbd.dt = curr_dt_solid_inuse
                xpbd.make_prediction()
                solve_for_xpbd(curr_dt_solid_inuse)
        stretch_T_and_advect_particles(particles_pos, T_x, T_y, T_z, F_x, F_y, F_z, u_x, u_y, u_z, curr_dt)
        compute_particles_half_usquare_and_grad_u(particles_pos, particles_half_usquare, u_x, u_y, u_z)     
        get_central_vector(u_x, u_y, u_z, u)
        get_grad_usqure(u, u_square)

        if j != 0:
            accumulate_lamb(particles_grad_lamb, particles_pos, particles_grad_half_usquare, solver.p, u_square, curr_dt, prev_dt)
        else:
            accumulate_lamb_j(particles_grad_lamb, particles_pos, particles_grad_half_usquare, solver.p, u_square, curr_dt)
            
        update_particles_imp(particles_imp, particles_init_imp, particles_grad_lamb, particles_grad_half_usquare, T_x, T_y, T_z)
        
        P2G(particles_pos, u_x, u_y, u_z, p2g_weight_x, p2g_weight_y, p2g_weight_z, T_x, T_y, T_z, curr_dt)
        P2G_smoke(particles_smoke, particles_init_grad_smoke, particles_pos, smoke, T_x, T_y, T_z, p2g_weight)
        
        update_force(curr_dt * 0.5)

        spread_force(u_x, u_y, u_z, curr_dt * 0.5)

        
        add_smoke_source(smoke)
        solver.time = total_t[None]
        solver.Poisson(u_x, u_y, u_z, total_t[None])
        
        end_time = time.time()
        frame_time = end_time - start_time
        print(f'step execution time: {frame_time:.6f} seconds')

        if frame_idx <= total_steps - 1:
            frame_times[frame_idx] += frame_time
        prev_dt = curr_dt
        
        # store_cache_vel(curr_dt, pointLocation_copy)
        print("[Simulate] Done with step: ", i, " / substep: ", j, "\n", flush = True)

        if output_frame:
            # for visualization
            # write_image(levels_display[..., np.newaxis], levelsdir, frame_idx)
            get_central_vector(u_x, u_y, u_z, u)
            curl(u, w, dx)
            w_numpy = w.to_numpy()
            w_norm = np.linalg.norm(w_numpy, axis = -1)
            write_field(w_norm[:,:,res_z//2], vort2dir, frame_idx, vmin = w_min, vmax = w_max)
            smoke_numpy = smoke.to_numpy()
            smoke_norm = smoke_numpy[...,-1]
            write_image(smoke_numpy[:,:,res_z//2][...,:3], smoke2dir, frame_idx)
            write_w_and_smoke(w_norm, smoke_norm, vtkdir, frame_idx)
            Export(soliddir, frame_idx)
            if frame_idx % ckpt_every == 0:
                # np.save(os.path.join(ckptdir, "vel_x_numpy_" + str(frame_idx)), u_x.to_numpy())
                # np.save(os.path.join(ckptdir, "vel_y_numpy_" + str(frame_idx)), u_y.to_numpy())
                # np.save(os.path.join(ckptdir, "vel_z_numpy_" + str(frame_idx)), u_z.to_numpy())
                # np.save(os.path.join(ckptdir, "w_numpy_" + str(frame_idx)), w_numpy)
                # np.save(os.path.join(ckptdir, "smoke_numpy_" + str(frame_idx)), smoke_numpy)
                if save_particle_pos_numpy:
                    np.save(os.path.join(ckptdir, "particles_pos_numpy_" + str(frame_idx)), particles_pos.to_numpy())

            print("[Simulate] Finished frame: ", frame_idx, " in ", i-last_output_substep, "substeps \n\n")
            last_output_substep = i

            # if reached desired number of frames
            if frame_idx >= total_frames:
                frame_time_dir = 'frame_time'
                frame_time_dir = os.path.join(logsdir, frame_time_dir)
                os.makedirs(f'{frame_time_dir}', exist_ok=True)
                np.save(f'{frame_time_dir}/frame_times.npy', frame_times)
                break

        if use_total_steps and frame_idx >= total_steps - 1:
            frame_time_dir = 'frame_time'
            frame_time_dir = os.path.join(logsdir, frame_time_dir)
            os.makedirs(f'{frame_time_dir}', exist_ok=True)
            np.save(f'{frame_time_dir}/frame_times.npy', frame_times)
            break
    
    
        
if __name__ == '__main__':
    print("[Main] Begin")
    if len(sys.argv) <= 1:
        main(from_frame = from_frame)
    else:
        main(from_frame = from_frame, testing = True)
    print("[Main] Complete")
