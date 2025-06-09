#
from hyperparameters import *
from taichi_utils import *
from mgpcg_m import *
from init_conditions import *
from io_utils import *
import sys
import shutil
import time
ti.init(arch=ti.cuda, device_memory_GB=3.0, debug=False)
from mpm_utils import *
from p2g_utils_swim3 import *

#

# dx = 1. / res_y
dx = 1.0 / res_y
half_dx = 0.5 * dx
upper_boundary = 1 - half_dx
lower_boundary = half_dx
right_boundary = res_x * dx - half_dx
left_boundary = half_dx


# uniform distribute particles
particles_per_cell_axis = int(ti.sqrt(particles_per_cell))
dist_between_neighbor = dx / particles_per_cell_axis
boundary_mask = ti.field(ti.i32, shape=(res_x, res_y))
boundary_vel = ti.Vector.field(2, float, shape=(res_x, res_y))
# solver
boundary_types = ti.Matrix([[2, 2], [2, 2]], ti.i32)  # boundaries: 1 means Dirichlet, 2 means Neumann
solver = MGPCG_2(boundary_types, boundary_mask, boundary_vel, N = [res_x, res_y], levelset = levelset, p_rho = 1* p_rho, p_rho_w = 1 * p_rho_w, base_level=3)

# undeformed coordinates (cell center and faces)
X = ti.Vector.field(2, float, shape=(res_x, res_y))
X_horizontal = ti.Vector.field(2, float, shape=(res_x + 1, res_y))
X_vertical = ti.Vector.field(2, float, shape=(res_x, res_y + 1))
center_coords_func(X, dx)
horizontal_coords_func(X_horizontal, dx)
vertical_coords_func(X_vertical, dx)

# back flow map
T_x_grid = ti.Vector.field(2, float, shape=(res_x + 1, res_y))  # d_psi / d_x
T_y_grid = ti.Vector.field(2, float, shape=(res_x, res_y + 1))  # d_psi / d_y
psi_x_grid = ti.Vector.field(2, float, shape=(res_x + 1, res_y))  # x coordinate
psi_y_grid = ti.Vector.field(2, float, shape=(res_x, res_y + 1))  # y coordinate


# P2G weight storage
p2g_weight_x = ti.field(float, shape=(res_x + 1, res_y))
p2g_weight_y = ti.field(float, shape=(res_x, res_y + 1))
p2g_weight_x2 = ti.field(float, shape=(res_x + 1, res_y))
p2g_weight_y2 = ti.field(float, shape=(res_x, res_y + 1))
force_x = ti.field(float, shape=(res_x + 1, res_y))
force_y = ti.field(float, shape=(res_x, res_y + 1))

# velocity storage
u = ti.Vector.field(2, float, shape=(res_x, res_y))
pressure = ti.field(float, shape=(res_x, res_y))
w = ti.field(float, shape=(res_x, res_y))

u_x = ti.field(float, shape=(res_x + 1, res_y))
u_y = ti.field(float, shape=(res_x, res_y + 1))

rho_x = ti.field(float, shape=(res_x + 1, res_y))
rho_y = ti.field(float, shape=(res_x, res_y + 1))

u_square = ti.field(float, shape=(res_x, res_y))

initial_particle_num = (res_x) * (res_y) * particles_per_cell

particle_num = initial_particle_num * total_particles_num_ratio
current_particle_num = ti.field(int, shape=1)
particles_active = ti.field(float, shape=particle_num)
particles_active.fill(1)
particles_active_mpm_blur = ti.field(float, shape=n_particles_blur_inuse)
particles_active_mpm_blur.fill(1)
particles_pos = ti.Vector.field(2, float, shape=particle_num)
particles_pos_copy = ti.Vector.field(2, float, shape=particle_num)
particles_imp = ti.Vector.field(2, float, shape=particle_num)

particles_imp_mpm_blur = ti.Vector.field(2, float, shape=n_particles_blur_inuse)
particles_init_imp_mpm_blur = ti.Vector.field(2, float, shape=n_particles_blur_inuse)

particles_init_imp = ti.Vector.field(2, float, shape=particle_num)
particles_init_imp_grad_m = ti.Vector.field(2, float, shape=particle_num)


particles_lamb = ti.field(float, shape=particle_num)
particles_lamb_mpm_blur = ti.field(float, shape=n_particles_blur_inuse)

particles_grad_lamb = ti.Vector.field(2, float, shape=particle_num)
particles_grad_lamb_mpm_blur = ti.Vector.field(2, float, shape=n_particles_blur_inuse)

particles_half_usquare = ti.field(float, shape=particle_num)
particles_half_usquare_mpm_blur = ti.field(float, shape=n_particles_blur_inuse)

particles_grad_half_usquare = ti.Vector.field(2, float, shape=particle_num)
particles_grad_half_usquare_mpm_blur = ti.Vector.field(2, float, shape=n_particles_blur_inuse)

gradm_x_grid = ti.Vector.field(2, float, shape=(res_x + 1, res_y))  # d_psi / d_x
gradm_y_grid = ti.Vector.field(2, float, shape=(res_x, res_y + 1))  # d_psi / d_
gradm_x = ti.Vector.field(2, float, shape=particle_num)
gradm_y = ti.Vector.field(2, float, shape=particle_num)
gradm_x_mpm_blur = ti.Vector.field(2, float, shape=n_particles_blur_inuse)
gradm_y_mpm_blur = ti.Vector.field(2, float, shape=n_particles_blur_inuse)


solid_particles_cell = ti.field(int, shape=(res_x, res_y))
solid_particle_pressure = ti.Vector.field(2, float, shape=n_particles)



# reseed particle
# grid_particle_num = ti.field(int, shape=(res_x, res_y))

# back flow map
T_x = ti.Vector.field(2, float, shape=particle_num)  # d_psi / d_x
T_y = ti.Vector.field(2, float, shape=particle_num)  # d_psi / d_y

F_x = ti.Vector.field(2, float, shape=particle_num)  # d_psi / d_x
F_y = ti.Vector.field(2, float, shape=particle_num)  # d_psi / d_y

T_x_copy = ti.Vector.field(2, float, shape=particle_num)  # d_psi / d_x
T_y_copy = ti.Vector.field(2, float, shape=particle_num)  # d_psi / d_y

F_x_copy = ti.Vector.field(2, float, shape=particle_num)  # d_psi / d_x
F_y_copy = ti.Vector.field(2, float, shape=particle_num)  # d_psi / d_y


T_x_init = ti.Vector.field(2, float, shape=particle_num)  # d_psi / d_x
T_y_init = ti.Vector.field(2, float, shape=particle_num)  # d_psi / d_y
T_x_grad_m = ti.Vector.field(2, float, shape=particle_num)  # d_psi / d_x
T_y_grad_m = ti.Vector.field(2, float, shape=particle_num)  # d_psi / d_y
# T_x_backup = ti.Vector.field(2, float, shape=particle_num)  # d_psi / d_x
# T_y_backup = ti.Vector.field(2, float, shape=particle_num)  # d_psi / d_y
psi = ti.Vector.field(2, float, shape=particle_num)  # x coordinate
# grad_T_x = ti.Matrix.field(2, 2, float, shape=particle_num)
# grad_T_y = ti.Matrix.field(2, 2, float, shape=particle_num)
grad_T_init_x = ti.Matrix.field(2, 2, float, shape=particle_num)
grad_T_init_y = ti.Matrix.field(2, 2, float, shape=particle_num)

# paticles in each cell
cell_max_particle_num = int(cell_max_particle_num_ratio * particles_per_cell)
cell_particle_num = ti.field(int, shape=(res_x, res_y))
cell_particles_id = ti.field(int, shape=(res_x, res_y, cell_max_particle_num))


# APIC
C_x = ti.Vector.field(2, float, shape=particle_num)
C_y = ti.Vector.field(2, float, shape=particle_num)
C_x_mpm_blur = ti.Vector.field(2, float, shape=n_particles_blur_inuse)
C_y_mpm_blur = ti.Vector.field(2, float, shape=n_particles_blur_inuse)

init_C_x = ti.Vector.field(2, float, shape=particle_num)
init_C_y = ti.Vector.field(2, float, shape=particle_num)


tmp_u_x = ti.field(float, shape=(res_x + 1, res_y))
tmp_u_y = ti.field(float, shape=(res_x, res_y + 1))
tmp_u = ti.Vector.field(2, float, shape=(res_x, res_y))
tmp_rho = ti.field(float, shape=(res_x, res_y))

# CFL related
max_speed = ti.field(float, shape=())
max_speed_solid = ti.field(float, shape=())
total_t = ti.field(float, shape=())


# dts = torch.zeros(1)

# smoke
init_smoke = ti.Vector.field(3, float, shape=(res_x, res_y))
smoke = ti.Vector.field(3, float, shape=(res_x, res_y))
tmp_smoke = ti.Vector.field(3, float, shape=(res_x, res_y))
err_smoke = ti.Vector.field(3, float, shape=(res_x, res_y))



@ti.kernel
def calc_max_speed(u_x: ti.template(), u_y: ti.template()):
    max_speed[None] = 1.e-3  # avoid dividing by zero
    for i, j in ti.ndrange(res_x, res_y):
        u = 0.5 * (u_x[i, j] + u_x[i + 1, j])
        v = 0.5 * (u_y[i, j] + u_y[i, j + 1])
        speed = ti.sqrt(u ** 2 + v ** 2)
        ti.atomic_max(max_speed[None], speed)
        
        
@ti.kernel
def calc_max_speed_solid(mpm_v: ti.template()):
    max_speed_solid[None] = 1.e-3  # avoid dividing by zero
    for i in mpm_v:
        speed = ti.sqrt(mpm_v[i][0]**2 + mpm_v[i][1]**2)
        ti.atomic_max(max_speed_solid[None], speed)

@ti.kernel
def calc_max_imp_particles(particles_imp: ti.template()):
    max_speed[None] = 1.e-3  # avoid dividing by zero
    for i in particles_imp:
        if particles_active[i] == 1:
            imp = particles_imp[i].norm()
            ti.atomic_max(max_speed[None], imp)


# set to undeformed config
@ti.kernel
def reset_to_identity_grid(psi_x: ti.template(), psi_y: ti.template(), T_x: ti.template(), T_y: ti.template()):
    for i, j in psi_x:
        psi_x[i, j] = X_horizontal[i, j]
    for i, j in psi_y:
        psi_y[i, j] = X_vertical[i, j]
    for i, j in T_x:
        T_x[i, j] = ti.Vector.unit(2, 0)
    for i, j in T_y:
        T_y[i, j] = ti.Vector.unit(2, 1)


@ti.kernel
def reset_to_identity(psi: ti.template(), T_x: ti.template(), T_y: ti.template()):
    for i in psi:
        psi[i] = particles_pos[i]
    for i in T_x:
        T_x[i] = ti.Vector.unit(2, 0)
    for i in T_y:
        T_y[i] = ti.Vector.unit(2, 1)

@ti.kernel
def reset_T_to_identity(T_x: ti.template(), T_y: ti.template()):
    for i in T_x:
        T_x[i] = ti.Vector.unit(2, 0)
    for i in T_y:
        T_y[i] = ti.Vector.unit(2, 1)
        
@ti.kernel
def reset_F_to_identity(F_x: ti.template(), F_y: ti.template()):
    for i in F_x:
        F_x[i] = ti.Vector.unit(2, 0)
    for i in F_y:
        F_y[i] = ti.Vector.unit(2, 1)



@ti.kernel
def check_psi_and_X(curr_step: int, psi: ti.template(), particles_pos_backup: ti.template()):
    different_X_psi_num = 0
    for i in particles_pos_backup:
        if particles_active[i] == 1:
            diff = psi[i] - particles_pos_backup[i]
            if diff.norm() > 1e-3:
                different_X_psi_num += 1

    print(f'Step {curr_step}: {different_X_psi_num}/{particle_num} different psi and X')



# def stretch_T_and_advect_particles(particles_pos, T_x, T_y, F_x, F_y, u_x, u_y, dt):
#     RK4_T_forward(particles_pos, T_x, T_y, F_x, F_y, u_x, u_y, dt, 1)
#     # copy_to(particles_pos_backup, psi)
    
def stretch_T_and_advect_particles(particles_pos, particles_imp, T_x, T_y, F_x, F_y, u_x, u_y, dt):
    RK4_T_forward(particles_pos, particles_imp, T_x, T_y, F_x, F_y, u_x, u_y, dt, 1)
    # copy_to(particles_pos_backup, psi)

def stretch_T(particles_pos, T_x, T_y, u_x, u_y, dt):
    RK4_T_forward(particles_pos, T_x, T_y, u_x, u_y, dt, 0)


# curr step should be in range(reinit_every)
# def march_phi_grid(curr_step):
#     RK4(phi, F_x, F_y, u_x, u_y, -1 * dts[curr_step].item())


@ti.kernel
def RK4_grid(psi_x: ti.template(), T_x: ti.template(),
             u_x0: ti.template(), u_y0: ti.template(), dt: float):
    neg_dt = -1 * dt  # travel back in time
    for i, j in psi_x:
        # first
        u1, grad_u_at_psi, _, _ = interp_u_MAC_grad(u_x0, u_y0, psi_x[i, j], dx)
        dT_x_dt1 = grad_u_at_psi @ T_x[i, j]  # time derivative of T
        # prepare second
        psi_x1 = psi_x[i, j] + 0.5 * neg_dt * u1  # advance 0.5 steps
        T_x1 = T_x[i, j] + 0.5 * neg_dt * dT_x_dt1
        # second
        u2, grad_u_at_psi, _, _ = interp_u_MAC_grad(u_x0, u_y0, psi_x1, dx)
        dT_x_dt2 = grad_u_at_psi @ T_x1  # time derivative of T
        # prepare third
        psi_x2 = psi_x[i, j] + 0.5 * neg_dt * u2  # advance 0.5 again
        T_x2 = T_x[i, j] + 0.5 * neg_dt * dT_x_dt2
        # third
        u3, grad_u_at_psi, _, _ = interp_u_MAC_grad(u_x0, u_y0, psi_x2, dx)
        dT_x_dt3 = grad_u_at_psi @ T_x2  # time derivative of T
        # prepare fourth
        psi_x3 = psi_x[i, j] + 1.0 * neg_dt * u3
        T_x3 = T_x[i, j] + 1.0 * neg_dt * dT_x_dt3  # advance 1.0
        # fourth
        u4, grad_u_at_psi, _, _ = interp_u_MAC_grad(u_x0, u_y0, psi_x3, dx)
        dT_x_dt4 = grad_u_at_psi @ T_x3  # time derivative of T
        # final advance
        psi_x[i, j] = psi_x[i, j] + neg_dt * 1. / 6 * (u1 + 2 * u2 + 2 * u3 + u4)
        T_x[i, j] = T_x[i, j] + neg_dt * 1. / 6 * (dT_x_dt1 + 2 * dT_x_dt2 + 2 * dT_x_dt3 + dT_x_dt4)  # advance full


@ti.kernel
def RK4(psi: ti.template(), T_x: ti.template(), T_y: ti.template(),
        u_x0: ti.template(), u_y0: ti.template(), dt: float):
    neg_dt = -1 * dt  # travel back in time
    for i in psi:
        if particles_active[i] == 1:
            # first
            u1, grad_u_at_psi, _, _ = interp_u_MAC_grad(u_x0, u_y0, psi[i], dx)
            dT_x_dt1 = grad_u_at_psi @ T_x[i]  # time derivative of T
            dT_y_dt1 = grad_u_at_psi @ T_y[i]  # time derivative of T
            # prepare second
            psi_x1 = psi[i] + 0.5 * neg_dt * u1  # advance 0.5 steps
            T_x1 = T_x[i] + 0.5 * neg_dt * dT_x_dt1
            T_y1 = T_y[i] + 0.5 * neg_dt * dT_y_dt1
            # second
            u2, grad_u_at_psi, _, _ = interp_u_MAC_grad(u_x0, u_y0, psi_x1, dx)
            dT_x_dt2 = grad_u_at_psi @ T_x1  # time derivative of T
            dT_y_dt2 = grad_u_at_psi @ T_y1  # time derivative of T
            # prepare third
            psi_x2 = psi[i] + 0.5 * neg_dt * u2  # advance 0.5 again
            T_x2 = T_x[i] + 0.5 * neg_dt * dT_x_dt2
            T_y2 = T_y[i] + 0.5 * neg_dt * dT_y_dt2
            # third
            u3, grad_u_at_psi, _, _ = interp_u_MAC_grad(u_x0, u_y0, psi_x2, dx)
            dT_x_dt3 = grad_u_at_psi @ T_x2  # time derivative of T
            dT_y_dt3 = grad_u_at_psi @ T_y2  # time derivative of T
            # prepare fourth
            psi_x3 = psi[i] + 1.0 * neg_dt * u3
            T_x3 = T_x[i] + 1.0 * neg_dt * dT_x_dt3  # advance 1.0
            T_y3 = T_y[i] + 1.0 * neg_dt * dT_y_dt3  # advance 1.0
            # fourth
            u4, grad_u_at_psi, _, _ = interp_u_MAC_grad(u_x0, u_y0, psi_x3, dx)
            dT_x_dt4 = grad_u_at_psi @ T_x3  # time derivative of T
            dT_y_dt4 = grad_u_at_psi @ T_y3  # time derivative of T
            # final advance
            psi[i] = psi[i] + neg_dt * 1. / 6 * (u1 + 2 * u2 + 2 * u3 + u4)
            T_x[i] = T_x[i] + neg_dt * 1. / 6 * (dT_x_dt1 + 2 * dT_x_dt2 + 2 * dT_x_dt3 + dT_x_dt4)  # advance full
            T_y[i] = T_y[i] + neg_dt * 1. / 6 * (dT_y_dt1 + 2 * dT_y_dt2 + 2 * dT_y_dt3 + dT_y_dt4)  # advance full


@ti.kernel
def RK4_T_forward(psi: ti.template(), particles_imp: ti.template(), T_x: ti.template(), T_y: ti.template(), F_x: ti.template(), F_y: ti.template(),
                  u_x0: ti.template(), u_y0: ti.template(), dt: float, advect_psi: int):
    for i in psi:
        if particles_active[i] == 1:
            # first
            u1, grad_u_at_psi, _, _ = interp_u_MAC_grad(u_x0, u_y0, psi[i], dx)
            dT_x_dt1 = grad_u_at_psi.transpose() @ T_x[i]  # time derivative of T
            dT_y_dt1 = grad_u_at_psi.transpose() @ T_y[i]  # time derivative of T
            
            dF_x_dt1 = grad_u_at_psi @ F_x[i]  # time derivative of F
            dF_y_dt1 = grad_u_at_psi @ F_y[i]  # time derivative of F
            
            # prepare second
            psi_x1 = psi[i] + 0.5 * dt * u1  # advance 0.5 steps
            T_x1 = T_x[i] - 0.5 * dt * dT_x_dt1
            T_y1 = T_y[i] - 0.5 * dt * dT_y_dt1
            
            F_x1 = F_x[i] + 0.5 * dt * dF_x_dt1
            F_y1 = F_y[i] + 0.5 * dt * dF_y_dt1
            # second
            u2, grad_u_at_psi, _, _ = interp_u_MAC_grad(u_x0, u_y0, psi_x1, dx)
            dT_x_dt2 = grad_u_at_psi.transpose() @ T_x1  # time derivative of T
            dT_y_dt2 = grad_u_at_psi.transpose() @ T_y1  # time derivative of T
            
            dF_x_dt2 = grad_u_at_psi @ F_x1 # time derivative of F
            dF_y_dt2 = grad_u_at_psi @ F_y1  # time derivative of F
            # prepare third
            psi_x2 = psi[i] + 0.5 * dt * u2  # advance 0.5 again
            T_x2 = T_x[i] - 0.5 * dt * dT_x_dt2
            T_y2 = T_y[i] - 0.5 * dt * dT_y_dt2
            
            F_x2 = F_x[i] + 0.5 * dt * dF_x_dt2
            F_y2 = F_y[i] + 0.5 * dt * dF_y_dt2
            # third
            u3, grad_u_at_psi, _, _ = interp_u_MAC_grad(u_x0, u_y0, psi_x2, dx)
            dT_x_dt3 = grad_u_at_psi.transpose() @ T_x2  # time derivative of T
            dT_y_dt3 = grad_u_at_psi.transpose() @ T_y2  # time derivative of T
            
            dF_x_dt3 = grad_u_at_psi @ F_x2  # time derivative of T
            dF_y_dt3 = grad_u_at_psi @ F_y2  # time derivative of T
            # prepare fourth
            psi_x3 = psi[i] + 1.0 * dt * u3
            T_x3 = T_x[i] - 1.0 * dt * dT_x_dt3  # advance 1.0
            T_y3 = T_y[i] - 1.0 * dt * dT_y_dt3  # advance 1.0
            
            F_x3 = F_x[i] + 1.0 * dt * dF_x_dt3  # advance 1.0
            F_y3 = F_y[i] + 1.0 * dt * dF_y_dt3  # advance 1.0
            # fourth
            u4, grad_u_at_psi, _, _ = interp_u_MAC_grad(u_x0, u_y0, psi_x3, dx)
            dT_x_dt4 = grad_u_at_psi.transpose() @ T_x3  # time derivative of T
            dT_y_dt4 = grad_u_at_psi.transpose() @ T_y3  # time derivative of T
            
            dF_x_dt4 = grad_u_at_psi @ F_x3  # time derivative of T
            dF_y_dt4 = grad_u_at_psi @ F_y3  # time derivative of T
            # final advance
            if advect_psi:
                psi[i] = psi[i] + dt * 1. / 6 * (u1 + 2 * u2 + 2 * u3 + u4)
            T_x[i] = T_x[i] - dt * 1. / 6 * (dT_x_dt1 + 2 * dT_x_dt2 + 2 * dT_x_dt3 + dT_x_dt4)  # advance full
            T_y[i] = T_y[i] - dt * 1. / 6 * (dT_y_dt1 + 2 * dT_y_dt2 + 2 * dT_y_dt3 + dT_y_dt4)  # advance full
            
            F_x[i] = F_x[i] + dt * 1. / 6 * (dF_x_dt1 + 2 * dF_x_dt2 + 2 * dF_x_dt3 + dF_x_dt4)  # advance full
            F_y[i] = F_y[i] + dt * 1. / 6 * (dF_y_dt1 + 2 * dF_y_dt2 + 2 * dF_y_dt3 + dF_y_dt4)  # advance full
            
            particles_imp[i] = 1. / 6 * (u1 + 2 * u2 + 2 * u3 + u4)


            
@ti.kernel
def RK4_T_forward_mpm(psi: ti.template(), mpm_v: ti.template(), T_mpm: ti.template(), F_mpm: ti.template(), C_mpm:ti.template(),
                  u_x0: ti.template(), u_y0: ti.template(), u0: ti.template(), dt: float, advect_psi: int):
    for i in psi:
        # first
        T_x = T_mpm[i][0, :]
        T_y = T_mpm[i][1, :]
        F_x = F_mpm[i][:, 0]
        F_y = F_mpm[i][:, 1]
        
        u1, grad_u_at_psi, _, _ = interp_u_MAC_grad(u_x0, u_y0, psi[i], dx)
        dT_x_dt1 = grad_u_at_psi.transpose() @ T_x  # time derivative of T
        dT_y_dt1 = grad_u_at_psi.transpose() @ T_y  # time derivative of T

        dF_x_dt1 = grad_u_at_psi @ F_x  # time derivative of F
        dF_y_dt1 = grad_u_at_psi @ F_y  # time derivative of F

        nooo, dC_dt1 = interp_grad_center(u0, psi[i], dx)

        # prepare second
        psi_x1 = psi[i] + 0.5 * dt * u1  # advance 0.5 steps
        T_x1 = T_x - 0.5 * dt * dT_x_dt1
        T_y1 = T_y - 0.5 * dt * dT_y_dt1

        F_x1 = F_x + 0.5 * dt * dF_x_dt1
        F_y1 = F_y + 0.5 * dt * dF_y_dt1
        # second
        u2, grad_u_at_psi, _, _ = interp_u_MAC_grad(u_x0, u_y0, psi_x1, dx)
        dT_x_dt2 = grad_u_at_psi.transpose() @ T_x1  # time derivative of T
        dT_y_dt2 = grad_u_at_psi.transpose() @ T_y1  # time derivative of T

        dF_x_dt2 = grad_u_at_psi @ F_x1 # time derivative of F
        dF_y_dt2 = grad_u_at_psi @ F_y1  # time derivative of F

        nooo, dC_dt2 = interp_grad_center(u0, psi_x1, dx)
        # prepare third
        psi_x2 = psi[i] + 0.5 * dt * u2  # advance 0.5 again
        T_x2 = T_x - 0.5 * dt * dT_x_dt2
        T_y2 = T_y - 0.5 * dt * dT_y_dt2

        F_x2 = F_x + 0.5 * dt * dF_x_dt2
        F_y2 = F_y + 0.5 * dt * dF_y_dt2
        # third
        u3, grad_u_at_psi, _, _ = interp_u_MAC_grad(u_x0, u_y0, psi_x2, dx)
        dT_x_dt3 = grad_u_at_psi.transpose() @ T_x2  # time derivative of T
        dT_y_dt3 = grad_u_at_psi.transpose() @ T_y2  # time derivative of T

        dF_x_dt3 = grad_u_at_psi @ F_x2  # time derivative of T
        dF_y_dt3 = grad_u_at_psi @ F_y2  # time derivative of T

        nooo, dC_dt3 = interp_grad_center(u0, psi_x2, dx)
        # prepare fourth
        psi_x3 = psi[i] + 1.0 * dt * u3
        T_x3 = T_x - 1.0 * dt * dT_x_dt3  # advance 1.0
        T_y3 = T_y - 1.0 * dt * dT_y_dt3  # advance 1.0

        F_x3 = F_x + 1.0 * dt * dF_x_dt3  # advance 1.0
        F_y3 = F_y + 1.0 * dt * dF_y_dt3  # advance 1.0
        # fourth
        u4, grad_u_at_psi, _, _ = interp_u_MAC_grad(u_x0, u_y0, psi_x3, dx)
        dT_x_dt4 = grad_u_at_psi.transpose() @ T_x3  # time derivative of T
        dT_y_dt4 = grad_u_at_psi.transpose() @ T_y3  # time derivative of T

        dF_x_dt4 = grad_u_at_psi @ F_x3  # time derivative of T
        dF_y_dt4 = grad_u_at_psi @ F_y3  # time derivative of T

        nooo, dC_dt4 = interp_grad_center(u0, psi_x3, dx)
        # final advance
        if advect_psi:
            psi[i] = psi[i] + dt * 1. / 6 * (u1 + 2 * u2 + 2 * u3 + u4)
        T_x = T_x - dt * 1. / 6 * (dT_x_dt1 + 2 * dT_x_dt2 + 2 * dT_x_dt3 + dT_x_dt4)  # advance full
        T_y = T_y - dt * 1. / 6 * (dT_y_dt1 + 2 * dT_y_dt2 + 2 * dT_y_dt3 + dT_y_dt4)  # advance full

        F_x = F_x + dt * 1. / 6 * (dF_x_dt1 + 2 * dF_x_dt2 + 2 * dF_x_dt3 + dF_x_dt4)  # advance full
        F_y = F_y + dt * 1. / 6 * (dF_y_dt1 + 2 * dF_y_dt2 + 2 * dF_y_dt3 + dF_y_dt4)  # advance full

        # C_mpm[i] = 1. / 6 * (dC_dt1 + 2 * dC_dt2 + 2 * dC_dt3 + dC_dt4)
#         F_mpm[i] = ti.Matrix.rows([F_x, F_y])
#         T_mpm[i] = ti.Matrix.rows([T_x, T_y])
#         mpm_v[i] = 1. / 6 * (u1 + 2 * u2 + 2 * u3 + u4)
#         T_x = T_x - dt * dT_x_dt1
#         T_y = T_y - dt * dT_y_dt1

#         F_x = F_x + dt * dF_x_dt1
#         F_y = F_y + dt * dF_y_dt1

        C_mpm[i] = 1. / 6 * (dC_dt1 + 2 * dC_dt2 + 2 * dC_dt3 + dC_dt4)
        F_mpm[i] = ti.Matrix.cols([F_x, F_y])
        T_mpm[i] = ti.Matrix.rows([T_x, T_y])
        
        # u1, gradu, _ = interp_grad_center_2(u0, psi[i], dx)
        # F_mpm[i] += (dt * grad_u_at_psi) @ F_mpm[i] #updateF (explicitMPM way)
        # mpm_v[i] = u1
        # psi[i] = psi[i] + dt * u1
        # dC_dt1 = interp_grad_center(u0, psi[i], dx)
        mpm_v[i] = 1. / 6 * (u1 + 2 * u2 + 2 * u3 + u4)
#         # psi[i] += dt * u1


@ti.kernel
def advect_particles(dt: float):
    # different_x_phi_num = 0

    for i in particles_pos:
        if particles_active[i] == 1:
            # new_C_x, new_C_y = interp_u_MAC_grad_updated_imp(u_x, u_y, imp_x, imp_y, phi_x_grid, phi_y_grid, particles_pos[i], dx)
            # C_x[i] = new_C_x
            # C_y[i] = ne.w_C_y

            v1, _, _, _ = interp_u_MAC_grad(u_x, u_y, particles_pos[i], dx)
            p2 = particles_pos[i] + v1 * dt * 0.5
            v2, _, _, _ = interp_u_MAC_grad(u_x, u_y, p2, dx)
            p3 = particles_pos[i] + v2 * dt * 0.5
            v3, _, _, _ = interp_u_MAC_grad(u_x, u_y, p3, dx)
            p4 = particles_pos[i] + v3 * dt
            v4, _, _, _ = interp_u_MAC_grad(u_x, u_y, p4, dx)
            particles_pos[i] += (v1 + 2 * v2 + 2 * v3 + v4) / 6.0 * dt

            # diff = particles_pos[i] - phi[i]
            # if diff.norm() > 1e-5:
            #     different_x_phi_num += 1

    # print(f'{different_x_phi_num}/{particle_num} different phi and x')

@ti.kernel
def advect_u_grid(u_x0: ti.template(), u_y0: ti.template(), rho0: ti.template(),
             u_x1: ti.template(), u_y1: ti.template(), rho1: ti.template(), dx : float, dt : float):
    for I in ti.grouped(u_x1):
        p1 = X_horizontal[I]
        v1, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p1, dx)
        p2 = p1 - v1 * dt * 0.5
        v2, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p2, dx)
        p3 = p1 - v2 * dt * 0.5
        v3, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p3, dx)
        p4 = p1 - v3 * dt
        v4, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p4, dx)
        p = p1 - (v1 + 2 * v2 + 2 * v3 + v4) / 6.0 * dt
        v5, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p, dx)
        u_x1[I] = (v1 + 2 * v2 + 2 * v3 + v4)[0] / 6.0
        
    for I in ti.grouped(u_y1):
        p1 = X_vertical[I]
        v1, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p1, dx)
        p2 = p1 - v1 * dt * 0.5
        v2, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p2, dx)
        p3 = p1 - v2 * dt * 0.5
        v3, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p3, dx)
        p4 = p1 - v3 * dt
        v4, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p4, dx)
        p = p1 - (v1 + 2 * v2 + 2 * v3 + v4) / 6.0 * dt
        v5, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p, dx)
        u_y1[I] = (v1 + 2 * v2 + 2 * v3 + v4)[1] / 6.0
        
    for I in ti.grouped(rho1):
        p1 = X[I]
        v1, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p1, dx)
        p2 = p1 - v1 * dt * 0.5
        v2, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p2, dx)
        p3 = p1 - v2 * dt * 0.5
        v3, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p3, dx)
        p4 = p1 - v3 * dt
        v4, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p4, dx)
        p = p1 - (v1 + 2 * v2 + 2 * v3 + v4) / 6.0 * dt
        rho1[I] = interp_center(rho0, p, dx)
# @ti.kernel
# def get_grad_T(dt: float):
#     for i in particles_pos:
#         if particles_active[i] == 1:
#             C_grad_T_x, C_grad_T_y = interp_MAC_grad_T(T_x_grid, T_y_grid, particles_pos[i], dx)
#             grad_T_x[i] = C_grad_T_x
#             grad_T_y[i] = C_grad_T_y

@ti.kernel
def add_gravity_grid(u_x:ti.template(), u_y:ti.template(), gravity:float, curr_dt:float):
    for I in ti.grouped(u_x):
        u_x[I] += gravity * curr_dt
@ti.kernel
def limit_particles_in_boundary():
    for i in particles_pos:
        if particles_active[i] == 1:
            if particles_pos[i][0] < left_boundary:
                particles_pos[i][0] = left_boundary
            if particles_pos[i][0] > right_boundary:
                particles_pos[i][0] = right_boundary

            if particles_pos[i][1] < lower_boundary:
                particles_pos[i][1] = lower_boundary
            if particles_pos[i][1] > upper_boundary:
                particles_pos[i][1] = upper_boundary


@ti.kernel
def remove_fluid_particles_in_solid():
    solid_particles_cell.fill(0)
    # particles_active.fill(1)
    for i in range(n_particles):
        grid_idx = ti.floor((mpm_x[i]) / dx - 0.5, int)
        if 0 <= grid_idx[0] < res_x and 0 <= grid_idx[1] < res_y:
        # # loop over 16 indices
        # for a in range(-3, 4):
        #     for b in range(-3, 4):
        #         grid_idx2 = grid_idx + ti.Vector([a, b])
            solid_particles_cell[grid_idx] = 1
    for i in particles_pos:
        grid_idx = ti.floor((particles_pos[i]) / dx - 0.5, int)
        if 0 <= grid_idx[0] < res_x and 0 <= grid_idx[1] < res_y:
            if solid_particles_cell[grid_idx] >= 1:
                particles_active[i] = 0


# u_x0, u_y0 are the initial time quantities
# u_x1, u_y1 are the current time quantities (to be modified)
@ti.kernel
def advect_u(u_x0: ti.template(), u_y0: ti.template(),
             u_x1: ti.template(), u_y1: ti.template(),
             T_x: ti.template(), T_y: ti.template(),
             psi_x: ti.template(), psi_y: ti.template(), dx: float):
    # horizontal velocity
    for i, j in u_x1:
        u_at_psi, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, psi_x[i, j], dx)
        u_x1[i, j] = T_x[i, j].dot(u_at_psi)
    # vertical velocity
    for i, j in u_y1:
        u_at_psi, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, psi_y[i, j], dx)
        u_y1[i, j] = T_y[i, j].dot(u_at_psi)


@ti.kernel
def advect_smoke(smoke0: ti.template(), smoke1: ti.template(),
                 psi_x: ti.template(), psi_y: ti.template(), dx: float):
    # horizontal velocity
    for i, j in ti.ndrange(res_x, res_y):
        psi_c = 0.25 * (psi_x[i, j] + psi_x[i + 1, j] + psi_y[i, j] + psi_y[i, j + 1])
        smoke1[i, j] = interp_1(smoke0, psi_c, dx)


# def diffuse_sizing():
#     for _ in range(1024):
#         diffuse_grid(sizing, tmp_sizing)


@ti.kernel
def get_particles_id_in_every_cell(cell_particles_id: ti.template(), cell_particle_num: ti.template(),
                                   particles_pos: ti.template()):
    cell_particles_id.fill(-1)
    cell_particle_num.fill(0)
    for i in particles_pos:
        cell_id = int(particles_pos[i] / dx)
        particles_index_in_cell = ti.atomic_add(cell_particle_num[cell_id], 1)
        if particles_index_in_cell < cell_max_particle_num:
            cell_particles_id[cell_id[0], cell_id[1], particles_index_in_cell] = i


@ti.kernel
def compute_dT_dx(grad_T_init_x: ti.template(), grad_T_init_y: ti.template(), T_x: ti.template(),
                          T_y: ti.template(), particles_init_pos: ti.template(),
                          cell_particles_id: ti.template(), cell_particle_num: ti.template()):
    for i in grad_T_init_x:
        # weight = 0.
        # weight_x = 0.
        # weight_y = 0.
        if particles_active[i] == 1:
            dT_dx = ti.Matrix.zero(float, 2, 2)
            dT_dy = ti.Matrix.zero(float, 2, 2)
            grad_T_init_x[i] = ti.Matrix.rows([dT_dx[0, :], dT_dy[0, :]])
            grad_T_init_y[i] = ti.Matrix.rows([dT_dx[1, :], dT_dy[1, :]])
            
            


@ti.kernel
def update_particles_imp(particles_imp: ti.template(), particles_init_imp: ti.template(), grad_lamb: ti.template(), grad_half_usquare: ti.template(),
                         T_x: ti.template(), T_y: ti.template(), curr_dt: ti.template(), particles_active:ti.template()):
    for i in particles_imp:
        if particles_active[i] == 1:
            T = ti.Matrix.cols([T_x[i], T_y[i]])
            # particles_imp[i] = ti.Vector([T_x[i] @ particles_init_imp[i], T_y[i] @ particles_init_imp[i]])
            particles_imp[i] = T @ (particles_init_imp[i] - grad_lamb[i]) + grad_half_usquare[i]
            # particles_imp[i] = T @ particles_init_imp[i]
@ti.kernel
def update_particles_imp_add_visc(particles_imp: ti.template(), particles_init_imp: ti.template(), grad_lamb: ti.template(), grad_half_usquare: ti.template(),
                         T_x: ti.template(), T_y: ti.template(), curr_dt: ti.template(), particles_active:ti.template()):
    for i in particles_imp:
        if particles_active[i] == 1:
            T = ti.Matrix.cols([T_x[i], T_y[i]])
            # particles_imp[i] = ti.Vector([T_x[i] @ particles_init_imp[i], T_y[i] @ particles_init_imp[i]])
            particles_imp[i] = T @ particles_init_imp[i]
            
@ti.kernel
def update_particles_grad_m(C_x: ti.template(), C_y: ti.template(), init_C_x: ti.template(), init_C_y: ti.template(),
                            T_x: ti.template(), T_y: ti.template(), grad_T_init_x: ti.template(), grad_T_init_y: ti.template(),
                            particles_init_imp: ti.template()):
    for i in C_x:
        if particles_active[i] == 1:
            T = ti.Matrix.rows([T_x[i], T_y[i]])
            T_transpose = ti.Matrix.cols([T_x[i], T_y[i]])
            init_C = ti.Matrix.rows([init_C_x[i], init_C_y[i]])
            T_init_C_T = T_transpose @ (init_C @ T)
            C_x[i] = grad_T_init_x[i] @ particles_init_imp[i] + T_init_C_T[0, :]
            C_y[i] = grad_T_init_y[i] @ particles_init_imp[i] + T_init_C_T[1, :]

@ti.kernel
def update_T(T_x: ti.template(), T_y: ti.template(), T_x_init: ti.template(), T_y_init: ti.template(),
             T_x_grad_m: ti.template(), T_y_grad_m: ti.template()):
    for i in T_x:
        if particles_active[i] == 1:
            T_grad_m = ti.Matrix.cols([T_x_grad_m[i], T_y_grad_m[i]])
            T_init = ti.Matrix.cols([T_x_init[i], T_y_init[i]])
            T = T_grad_m @ T_init
            T_x[i] = T[:, 0]
            T_y[i] = T[:, 1]
        
        
@ti.kernel
def compute_particles_half_usquare_and_grad_u(particles_pos: ti.template(), particles_half_usquare: ti.template(), u_x: ti.template(), u_y: ti.template(), C_x: ti.template(), C_y: ti.template(), particles_active:ti.template()):
    C_x.fill(0.0)
    C_y.fill(0.0)
    for i in particles_half_usquare:
        if particles_active[i] == 1:
            p = particles_pos[i]
            u_x_p, grad_u_x_p, _ = interp_grad_2(u_x, p, dx, BL_x=0.0, BL_y=0.5, is_y=False)
            u_y_p, grad_u_y_p, _ = interp_grad_2(u_y, p, dx, BL_x=0.5, BL_y=0.0, is_y=True)
            C_x[i] = grad_u_x_p
            C_y[i] = grad_u_y_p
        
@ti.kernel
def mpm_compute_C_fluid(particles_pos: ti.template(), particles_half_usquare: ti.template(), u0:ti.template(), C_mpm_blur:ti.template(), particles_active:ti.template(), dx:float):
    C_mpm_blur.fill(0.0)
    for i in particles_half_usquare:
        if particles_active[i] == 1:
            p = particles_pos[i]
            nooo, dC_dt4 = interp_grad_center(u0, p, dx)
            C_mpm_blur[i] = dC_dt4
            
@ti.kernel
def accumulate_lamb(particles_grad_lamb: ti.template(), particles_pos: ti.template(), particles_grad_half_u: ti.template(), pressure: ti.template(), u: ti.template(), curr_dt: ti.template(), prev_dt: ti.template(), particles_active:ti.template(), rho: float):
    for i in particles_grad_half_u:
        if particles_active[i] == 1:
            p = particles_pos[i]
            nouse, grad_u, _ = interp_grad_2(u, p, dx, BL_x=0.5, BL_y=0.5)
            particles_grad_half_u[i] = grad_u * curr_dt
        
    for i in particles_grad_lamb:
        if particles_active[i] == 1:
            p = particles_pos[i]
            nouse, grad_p, _ = interp_grad_2(pressure, p, dx, BL_x=0.5, BL_y=0.5)
            nouse, grad_u, _ = interp_grad_2(u, p, dx, BL_x=0.5, BL_y=0.5)
            F = ti.Matrix.rows([F_x[i], F_y[i]])
            particles_grad_lamb[i] += F @ (grad_p / rho - grad_u) * prev_dt
            
            
# @ti.kernel
# def compute_solid_gradP(particles_pos:ti.template(), pressure:ti.template()):
#     for I in

@ti.kernel
def accumulate_lamb_j(particles_grad_lamb: ti.template(), particles_pos: ti.template(), particles_grad_half_u: ti.template(), pressure: ti.template(), u: ti.template(), curr_dt: ti.template(),particles_active:ti.template()):
    for i in particles_grad_half_u:
        p = particles_pos[i]
        nouse, grad_u, _ = interp_grad_2(u, p, dx, BL_x=0.5, BL_y=0.5)
        particles_grad_half_u[i] = grad_u * curr_dt
          
            
            
@ti.kernel
def get_grad_usquare(u: ti.template(), u_square: ti.template()):
    particles_half_usquare.fill(0.0)
    particles_half_usquare_mpm_blur.fill(0.0)
    for I in ti.grouped(u):
        u_square[I] = 0.5 * (u[I][0]**2 + u[I][1]**2)

        
def copy_to_FT():
    F_x_copy.copy_from(F_x)
    F_y_copy.copy_from(F_y)
    T_x_copy.copy_from(T_x)
    T_y_copy.copy_from(T_y)
    particles_pos_copy.copy_from(particles_pos)
    
def copy_from_FT():
    F_x.copy_from(F_x_copy)
    F_y.copy_from(F_y_copy)
    T_x.copy_from(T_x_copy)
    T_y.copy_from(T_y_copy)
    particles_pos.copy_from(particles_pos_copy)
    
    
def copy_to_mpm():
    mpm_x_temp.copy_from(mpm_x)
    mpm_v_temp.copy_from(mpm_v)
    F_temp.copy_from(F_mpm)
    F_x_mpm_blur_copy.copy_from(F_x_mpm_blur)
    F_y_mpm_blur_copy.copy_from(F_y_mpm_blur)
    T_x_mpm_blur_copy.copy_from(T_x_mpm_blur)
    T_y_mpm_blur_copy.copy_from(T_y_mpm_blur)
    mpm_blur_x_temp.copy_from(mpm_blur_x)
def copy_from_mpm():
    mpm_x.copy_from(mpm_x_temp)
    mpm_v.copy_from(mpm_v_temp)
    F_mpm.copy_from(F_temp)
    F_x_mpm_blur.copy_from(F_x_mpm_blur_copy)
    F_y_mpm_blur.copy_from(F_y_mpm_blur_copy)
    T_x_mpm_blur.copy_from(T_x_mpm_blur_copy)
    T_y_mpm_blur.copy_from(T_y_mpm_blur_copy)
    mpm_blur_x.copy_from(mpm_blur_x_temp)
    
# main function
def main(from_frame=0, testing=False):
    from_frame = max(0, from_frame)
    # create some folders
    logsdir = os.path.join('logs', exp_name)
    os.makedirs(logsdir, exist_ok=True)
    if from_frame <= 0:
        remove_everything_in(logsdir)

    vortdir = 'vorticity'
    vortdir = os.path.join(logsdir, vortdir)
    os.makedirs(vortdir, exist_ok=True)
    smokedir = 'smoke'
    smokedir = os.path.join(logsdir, smokedir)
    os.makedirs(smokedir, exist_ok=True)
    ckptdir = 'ckpts'
    ckptdir = os.path.join(logsdir, ckptdir)
    os.makedirs(ckptdir, exist_ok=True)
    levelsdir = 'levels'
    levelsdir = os.path.join(logsdir, levelsdir)
    os.makedirs(levelsdir, exist_ok=True)
    modeldir = 'model'  # saves the model
    modeldir = os.path.join(logsdir, modeldir)
    os.makedirs(modeldir, exist_ok=True)
    velocity_buffer_dir = 'velocity_buffer'
    velocity_buffer_dir = os.path.join(logsdir, velocity_buffer_dir)
    os.makedirs(velocity_buffer_dir, exist_ok=True)
    particles_dir = 'particles'
    particles_dir = os.path.join(logsdir, particles_dir)
    os.makedirs(particles_dir, exist_ok=True)

    shutil.copyfile('./hyperparameters.py', f'{logsdir}/hyperparameters.py')
    
    init_particles_pos_uniform(particles_pos, X, res_x, particles_per_cell, dx,
                               particles_per_cell_axis, dist_between_neighbor)
    remove_fluid_particles_in_solid()
    P2G_init(particles_pos, mpm_blur_x, mpm_x, rho_x, rho_y, p2g_weight_x, p2g_weight_y, particles_active, levelset, p_rho_w, p_rho)
    levelset.phi_temp.copy_from(levelset.phi)
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
        solver.Poisson(u_x, u_y, 1)
    else:
        u_x.from_numpy(np.load(os.path.join(ckptdir, "vel_x_numpy_" + str(from_frame) + ".npy")))
        u_y.from_numpy(np.load(os.path.join(ckptdir, "vel_y_numpy_" + str(from_frame) + ".npy")))
        smoke.from_numpy(np.load(os.path.join(ckptdir, "smoke_numpy_" + str(from_frame) + ".npy")))

    reset_T_to_identity(T_x_grad_m, T_y_grad_m)
    reset_F_to_identity(T_x_mpm_blur, T_y_mpm_blur)
    reset_F_to_identity(F_x, F_y)
    reset_F_to_identity(F_x_mpm_blur, F_y_mpm_blur)
    # backup_particles_pos(particles_pos, particles_pos_backup)
    init_particles_imp(particles_imp, particles_init_imp, particles_pos, u_x, u_y, init_C_x, init_C_y, dx)
    init_particles_imp(particles_imp_mpm_blur, particles_init_imp_mpm_blur, mpm_blur_x, u_x, u_y, init_C_x, init_C_y, dx)
    current_particle_num[0] = initial_particle_num

    # for visualization
    get_central_vector(u_x, u_y, u)
    curl(u, w, dx)
    w_numpy = w.to_numpy()
    w_max = max(np.abs(w_numpy.max()), np.abs(w_numpy.min()))
    w_min = -1 * w_max
    write_field(w_numpy, vortdir, from_frame, particles_pos.to_numpy() / dx, vmin=w_min,
                vmax=w_max,
                plot_particles=plot_particles, dpi=dpi_vor)
    # write_particles(w_numpy, particles_dir, from_frame, particles_pos.to_numpy() / dx, vmin=w_min, vmax=w_max)
    write_image(smoke.to_numpy(), smokedir, from_frame)

    if save_ckpt:
        np.save(os.path.join(ckptdir, "vel_x_numpy_" + str(from_frame)), u_x.to_numpy())
        np.save(os.path.join(ckptdir, "vel_y_numpy_" + str(from_frame)), u_y.to_numpy())
        np.save(os.path.join(ckptdir, "smoke_numpy_" + str(from_frame)), smoke.to_numpy())
        np.save(os.path.join(ckptdir, "mpm_x_numpy_" + str(from_frame)), mpm_x.to_numpy())
        np.save(os.path.join(ckptdir, "mpm_blur_x_numpy_" + str(from_frame)), mpm_blur_x.to_numpy())
        # np.save(os.path.join(ckptdir, "smoke_numpy_" + str(frame_idx)), smoke.to_numpy())
        np.save(os.path.join(ckptdir, "w_numpy_" + str(from_frame)), w.to_numpy())

    sub_t = 0.  # the time since last reinit
    frame_idx = from_frame
    last_output_substep = 0
    num_reinits = 0  # number of reinitializations already performed
    i = -1
    frame_times = np.zeros(total_steps)
    prev_dt = 0
    prev_total_t = 0
    while True:
        start_time = time.time()
        i += 1
        j = i % reinit_every
        k = i % reinit_every_grad_m
        i_next = i + 1
        j_next = i_next % reinit_every
        print("[Simulate] Running step: ", i, " / substep: ", j)

        # determine dt
        calc_max_speed(u_x, u_y)  # saved to max_speed[None]
        calc_max_speed_solid(mpm_v)  # saved to max_speed_solid[None]
        curr_dt = CFL * dx / max_speed[None]

        curr_dt_solid_inuse = 0.00002
        # curr_dt = curr_dt_solid_inuse
        if save_frame_each_step:
            output_frame = True
            frame_idx += 1
        else:
            if sub_t + curr_dt >= visualize_dt:  # if over
                curr_dt = visualize_dt - sub_t
                sub_t = 0.  # empty sub_t
                frame_idx += 1
                print(f'Visualized frame {frame_idx}')
                output_frame = True
            else:
                sub_t += curr_dt
                print(f'Visualize time {sub_t}/{visualize_dt}')
                output_frame = False

        solid_iters = int((int(curr_dt / curr_dt_solid_inuse) // 2) * 2 + 1)
        curr_dt_solid_inuse = curr_dt / (solid_iters * 1.0)

        prev_total_t = total_t[None]

        if j == 0:
            print("[Simulate] Reinitializing the flow map for the: ", num_reinits, " time!")

            if use_reseed_particles:
                # reseed_particles()
                pass
            else:
                if reinit_particle_pos:
                    init_particles_pos_uniform(particles_pos, X, res_x, particles_per_cell, dx,
                                               particles_per_cell_axis, dist_between_neighbor)
                    init_particles_imp_grad_m(particles_init_imp_grad_m, particles_pos, u_x, u_y,
                                              init_C_x, init_C_y, dx)
                    reset_T_to_identity(T_x_grad_m, T_y_grad_m)

            init_particles_imp(particles_imp, particles_init_imp, particles_pos, u_x, u_y, init_C_x, init_C_y, dx)

            reset_to_identity(psi, T_x, T_y)
            reset_T_to_identity(T_x_init, T_y_init)
            reset_F_to_identity(F_x, F_y)
            
            # particles_lamb.fill(0.0)
            particles_grad_lamb.fill(0.0)
            particles_grad_lamb_mpm_blur.fill(0.0)
            
            resample_mpm_blur()
            reset_T_to_identity(T_x_mpm_blur, T_y_mpm_blur)
            reset_F_to_identity(F_x_mpm_blur, F_y_mpm_blur)
            init_particles_imp(particles_imp_mpm_blur, particles_init_imp_mpm_blur, mpm_blur_x, u_x, u_y, init_C_x, init_C_y, dx)
            copy_to(smoke, init_smoke)

            get_central_vector(u_x, u_y, u)
            particles_active.fill(1)
            remove_fluid_particles_in_solid()
            P2G_init(particles_pos, mpm_blur_x, mpm_x, rho_x, rho_y, p2g_weight_x, p2g_weight_y, particles_active, levelset, p_rho_w, p_rho)
            num_reinits += 1

        if k == 0:
            init_particles_imp_grad_m(particles_init_imp_grad_m, particles_pos, u_x, u_y,
                                      init_C_x, init_C_y, dx)
            reset_T_to_identity(T_x_grad_m, T_y_grad_m)
            copy_to(T_x, T_x_init)
            copy_to(T_y, T_y_init)
            
            particles_grad_lamb_mpm_blur.fill(0.0)
            
            resample_mpm_blur()
            reset_T_to_identity(T_x_mpm_blur, T_y_mpm_blur)
            reset_F_to_identity(F_x_mpm_blur, F_y_mpm_blur)
            init_particles_imp(particles_imp_mpm_blur, particles_init_imp_mpm_blur, mpm_blur_x, u_x, u_y, init_C_x, init_C_y, dx)

        if use_midpoint_vel:
            copy_to_FT()
            # copy_to_mpm()
            get_central_vector(u_x, u_y, u)
            tmp_u_x.copy_from(u_x)
            tmp_u_y.copy_from(u_y)
            get_central_vector(tmp_u_x, tmp_u_y, tmp_u)
            for itr in range(int(solid_iters // 2) - 1):
                get_central_vector(tmp_u_x, tmp_u_y, tmp_u)
                stretch_T_and_advect_particles(mpm_blur_x, particles_imp_mpm_blur, T_x_mpm_blur, T_y_mpm_blur, F_x_mpm_blur, F_y_mpm_blur, tmp_u_x, tmp_u_y, curr_dt_solid_inuse)
                RK4_T_forward_mpm(mpm_x, mpm_v, T_mpm, F_mpm, C_mpm, tmp_u_x, tmp_u_y, tmp_u, curr_dt_solid_inuse, 1)
                mpm_compute_C_fluid(mpm_blur_x, particles_half_usquare_mpm_blur, tmp_u, C_mpm_blur, particles_active_mpm_blur, dx)
                compute_particles_half_usquare_and_grad_u(mpm_blur_x, particles_half_usquare_mpm_blur, tmp_u_x, tmp_u_y, C_x_mpm_blur, C_y_mpm_blur, particles_active_mpm_blur)
                P2G_solid(particles_imp, particles_imp_mpm_blur, mpm_v, particles_pos, mpm_blur_x, mpm_x, tmp_u_x, tmp_u_y, rho_x, rho_y, p2g_weight_x, p2g_weight_y, C_x, C_y, C_x_mpm_blur, C_y_mpm_blur, particles_active, F_mpm, C_mpm, mpm_act, levelset, p_rho_w, p_rho, curr_dt_solid_inuse, total_t[None], force_x, force_y, F_x, F_y, F_x_mpm_blur, F_y_mpm_blur, p2g_weight_x2, p2g_weight_y2, C_mpm_blur)
                total_t[None] += curr_dt_solid_inuse
            get_central_vector(tmp_u_x, tmp_u_y, tmp_u)
            stretch_T_and_advect_particles(mpm_blur_x, particles_imp_mpm_blur, T_x_mpm_blur, T_y_mpm_blur, F_x_mpm_blur, F_y_mpm_blur, tmp_u_x, tmp_u_y, curr_dt_solid_inuse)
            RK4_T_forward_mpm(mpm_x, mpm_v, T_mpm, F_mpm, C_mpm, tmp_u_x, tmp_u_y, tmp_u, curr_dt_solid_inuse, 1)
            
            stretch_T_and_advect_particles(particles_pos, particles_imp, T_x, T_y, F_x, F_y, u_x, u_y, curr_dt * 0.5)
            compute_particles_half_usquare_and_grad_u(particles_pos, particles_half_usquare, u_x, u_y, C_x, C_y, particles_active)
            compute_particles_half_usquare_and_grad_u(mpm_blur_x, particles_half_usquare_mpm_blur, u_x, u_y, C_x_mpm_blur, C_y_mpm_blur, particles_active_mpm_blur)
            # compute_particles_half_usquare_and_grad_u(mpm_blur_x, particles_half_usquare_mpm_blur, tmp_u_x, tmp_u_y, C_x_mpm_blur, C_y_mpm_blur, particles_active_mpm_blur)
            mpm_compute_C_fluid(mpm_blur_x, particles_half_usquare_mpm_blur, tmp_u, C_mpm_blur, particles_active_mpm_blur, dx)
            P2G(particles_imp, particles_imp_mpm_blur, mpm_v, particles_pos, mpm_blur_x, mpm_x, u_x, u_y,rho_x, rho_y, p2g_weight_x, p2g_weight_y, C_x, C_y, C_x_mpm_blur, C_y_mpm_blur, particles_active, F_mpm, C_mpm, mpm_act, levelset, p_rho_w, p_rho, curr_dt_solid_inuse, total_t[None], force_x, force_y, F_x, F_y, F_x_mpm_blur, F_y_mpm_blur, p2g_weight_x2, p2g_weight_y2, C_mpm_blur)
            total_t[None] += curr_dt_solid_inuse
            P2G_init(particles_pos, mpm_blur_x, mpm_x, rho_x, rho_y, p2g_weight_x, p2g_weight_y, particles_active, levelset, p_rho_w, p_rho)
            solver.Poisson(u_x, u_y, 0.5 * curr_dt)
            copy_from_FT()

        get_central_vector(u_x, u_y, u)


        for itr in range(int(solid_iters//2) - 1):
            get_central_vector(tmp_u_x, tmp_u_y, tmp_u)
            # stretch_T_and_advect_particles(particles_pos, particles_imp, T_x, T_y, F_x, F_y, tmp_u_x, tmp_u_y, curr_dt_solid_inuse)
            stretch_T_and_advect_particles(mpm_blur_x, particles_imp_mpm_blur, T_x_mpm_blur, T_y_mpm_blur, F_x_mpm_blur, F_y_mpm_blur, tmp_u_x, tmp_u_y, curr_dt_solid_inuse)
            RK4_T_forward_mpm(mpm_x, mpm_v, T_mpm, F_mpm, C_mpm, tmp_u_x, tmp_u_y, tmp_u, curr_dt_solid_inuse, 1)
            mpm_compute_C_fluid(mpm_blur_x, particles_half_usquare_mpm_blur, tmp_u, C_mpm_blur, particles_active_mpm_blur, dx)
            compute_particles_half_usquare_and_grad_u(mpm_blur_x, particles_half_usquare_mpm_blur, tmp_u_x, tmp_u_y, C_x_mpm_blur, C_y_mpm_blur, particles_active_mpm_blur)
            P2G_solid(particles_imp, particles_imp_mpm_blur, mpm_v, particles_pos, mpm_blur_x, mpm_x, tmp_u_x, tmp_u_y, rho_x, rho_y, p2g_weight_x, p2g_weight_y, C_x, C_y, C_x_mpm_blur, C_y_mpm_blur, particles_active, F_mpm, C_mpm, mpm_act, levelset, p_rho_w, p_rho, curr_dt_solid_inuse, total_t[None], force_x, force_y, F_x, F_y, F_x_mpm_blur, F_y_mpm_blur, p2g_weight_x2, p2g_weight_y2, C_mpm_blur)
            total_t[None] += curr_dt_solid_inuse
        get_central_vector(tmp_u_x, tmp_u_y, tmp_u)
        stretch_T_and_advect_particles(mpm_blur_x, particles_imp_mpm_blur, T_x_mpm_blur, T_y_mpm_blur, F_x_mpm_blur, F_y_mpm_blur, tmp_u_x, tmp_u_y, curr_dt_solid_inuse)
        RK4_T_forward_mpm(mpm_x, mpm_v, T_mpm, F_mpm, C_mpm, tmp_u_x, tmp_u_y, tmp_u, curr_dt_solid_inuse, 1)

        stretch_T_and_advect_particles(particles_pos, particles_imp, T_x, T_y, F_x, F_y, u_x, u_y, curr_dt)

        remove_fluid_particles_in_solid()
        compute_particles_half_usquare_and_grad_u(particles_pos, particles_half_usquare, u_x, u_y, C_x, C_y, particles_active)
        mpm_compute_C_fluid(mpm_blur_x, particles_half_usquare_mpm_blur, tmp_u, C_mpm_blur, particles_active_mpm_blur, dx)
        compute_particles_half_usquare_and_grad_u(mpm_blur_x, particles_half_usquare_mpm_blur, u_x, u_y, C_x_mpm_blur, C_y_mpm_blur, particles_active_mpm_blur)
        
        get_grad_usquare(u, u_square)
        if j != 0:
            accumulate_lamb(particles_grad_lamb, particles_pos, particles_grad_half_usquare, solver.p, u_square, curr_dt, prev_dt, particles_active, p_rho_w)
            accumulate_lamb(particles_grad_lamb_mpm_blur, mpm_blur_x, particles_grad_half_usquare_mpm_blur, solver.p, u_square, curr_dt, prev_dt, particles_active_mpm_blur, p_rho_w)
        else:
            accumulate_lamb_j(particles_grad_lamb, particles_pos, particles_grad_half_usquare, solver.p, u_square, curr_dt, particles_active)
            accumulate_lamb_j(particles_grad_lamb_mpm_blur, mpm_blur_x, particles_grad_half_usquare_mpm_blur, solver.p, u_square, curr_dt, particles_active_mpm_blur)
        
        if use_viscosity:
            update_particles_imp_add_visc(particles_imp, particles_init_imp, particles_grad_lamb, particles_grad_half_usquare, T_x, T_y, curr_dt, particles_active)
            update_particles_imp_add_visc(particles_imp_mpm_blur, particles_init_imp_mpm_blur, particles_grad_lamb_mpm_blur, particles_grad_half_usquare_mpm_blur, T_x_mpm_blur, T_y_mpm_blur, curr_dt, particles_active_mpm_blur)

            P2G_visc(particles_imp, particles_imp_mpm_blur, particles_pos, mpm_blur_x, u_x, u_y, p2g_weight_x, p2g_weight_y, C_x, C_y, C_x_mpm_blur, C_y_mpm_blur, particles_active)
            sample_gradm_and_p2g(gradm_x, gradm_y, gradm_x_mpm_blur, gradm_y_mpm_blur, gradm_x_grid, gradm_y_grid, particles_pos, mpm_blur_x, u_x, u_y, dx, p2g_weight_x, p2g_weight_y, particles_active)
            g2p_divergence_vis(particles_pos, particles_init_imp, gradm_x_grid, gradm_y_grid, F_x, F_y, particles_active, curr_dt, viscosity, dx)
            g2p_divergence_vis(mpm_blur_x, particles_init_imp_mpm_blur, gradm_x_grid, gradm_y_grid, F_x_mpm_blur, F_y_mpm_blur, particles_active_mpm_blur, curr_dt, viscosity, dx)
        


        update_particles_imp(particles_imp, particles_init_imp, particles_grad_lamb, particles_grad_half_usquare, T_x, T_y, curr_dt, particles_active)
        update_particles_imp(particles_imp_mpm_blur, particles_init_imp_mpm_blur, particles_grad_lamb_mpm_blur, particles_grad_half_usquare_mpm_blur, T_x_mpm_blur, T_y_mpm_blur, curr_dt, particles_active_mpm_blur)
        
        P2G(particles_imp, particles_imp_mpm_blur, mpm_v, particles_pos, mpm_blur_x, mpm_x, u_x, u_y,rho_x, rho_y, p2g_weight_x, p2g_weight_y, C_x, C_y, C_x_mpm_blur, C_y_mpm_blur, particles_active, F_mpm, C_mpm, mpm_act, levelset, p_rho_w, p_rho, curr_dt_solid_inuse, total_t[None], force_x, force_y, F_x, F_y, F_x_mpm_blur, F_y_mpm_blur, p2g_weight_x2, p2g_weight_y2, C_mpm_blur)

        P2G_init(particles_pos, mpm_blur_x, mpm_x, rho_x, rho_y, p2g_weight_x, p2g_weight_y, particles_active, levelset, p_rho_w, p_rho)

        solver.Poisson(u_x, u_y, curr_dt)

        pressure.copy_from(solver.p)
        levelset.phi_temp.copy_from(levelset.phi)
        total_t[None] = prev_total_t + curr_dt
        # sample_p_and_p2g(solid_particle_pressure, mpm_x, solver.p, u_x, u_y, p2g_weight_x, p2g_weight_y, force_x, force_y)
        if use_gravity:
            add_gravity(particles_init_imp, F_x, F_y, particles_active, curr_dt, gravity, 1)
            add_gravity(particles_init_imp_mpm_blur, F_x_mpm_blur, F_y_mpm_blur, particles_active_mpm_blur, curr_dt, gravity, 1)

        advect_smoke(init_smoke, smoke, psi_x_grid, psi_y_grid, dx)

        end_time = time.time()
        frame_time = end_time - start_time
        print(f'frame execution time: {frame_time:.6f} seconds')
        prev_dt = curr_dt
        if use_total_steps:
            frame_times[i] = frame_time


        print("[Simulate] Done with step: ", i, " / substep: ", j, "\n", flush=True)

        if output_frame:
            # for visualization
            get_central_vector(u_x, u_y, u)
            curl(u, w, dx)
            w_numpy = w.to_numpy()
            write_field(w_numpy, vortdir, frame_idx, particles_pos.to_numpy() / dx, vmin=w_min, vmax=w_max,
                        plot_particles=plot_particles, dpi=dpi_vor)
            write_image(smoke.to_numpy(), smokedir, frame_idx)
            if frame_idx % ckpt_every == 0 and save_ckpt:
                np.save(os.path.join(ckptdir, "vel_x_numpy_" + str(frame_idx)), u_x.to_numpy())
                np.save(os.path.join(ckptdir, "vel_y_numpy_" + str(frame_idx)), u_y.to_numpy())
                np.save(os.path.join(ckptdir, "mpm_x_numpy_" + str(frame_idx)), mpm_x.to_numpy())
                np.save(os.path.join(ckptdir, "mpm_blur_x_numpy_" + str(frame_idx)), mpm_blur_x.to_numpy())
                # np.save(os.path.join(ckptdir, "smoke_numpy_" + str(frame_idx)), smoke.to_numpy())
                np.save(os.path.join(ckptdir, "w_numpy_" + str(frame_idx)), w.to_numpy())

            print("\n[Simulate] Finished frame: ", frame_idx, " in ", i - last_output_substep, "substeps \n\n")
            last_output_substep = i

            # if reached desired number of frames
            if frame_idx >= total_frames:
                break

        if use_total_steps and i >= total_steps - 1:
            frame_time_dir = 'frame_time'
            frame_time_dir = os.path.join(logsdir, frame_time_dir)
            os.makedirs(f'{frame_time_dir}', exist_ok=True)
            np.save(f'{frame_time_dir}/frame_times.npy', frame_times)
            break


if __name__ == '__main__':
    print("[Main] Begin")
    if len(sys.argv) <= 1:
        main(from_frame=from_frame)
    else:
        main(from_frame=from_frame, testing=testing)
    print("[Main] Complete")
