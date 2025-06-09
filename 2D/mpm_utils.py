import taichi as ti
import numpy as np
from math import pi
from levelset import *
import matplotlib.pyplot as plt
import sys
from taichi_utils import *
from hyperparameters import *


print(f"n_particles: {n_particles} n_particles_blur: {n_particles_blur_inuse}")

sound_cfl_mpm_dt = mpm_dx / ti.sqrt(E/p_rho)


strength = ti.field(dtype=float, shape=())
strength[None] = 150

mpm_x = ti.Vector.field(2, dtype=float, shape=n_particles + n_particles_blur_inuse) # position
mpm_mass = ti.field(float, shape=n_particles + n_particles_blur_inuse) # position
mpm_blur_mass = ti.field(float, shape=n_particles_blur_inuse) # position
mpm_blur_x = ti.Vector.field(2, dtype=float, shape=n_particles_blur_inuse) # position
mpm_blur_normal = ti.Vector.field(2, dtype=float, shape=n_particles_blur_inuse) # position
mpm_force = ti.Vector.field(2, dtype=float, shape=n_particles) # position
mpm_blur_force = ti.Vector.field(2, dtype=float, shape=n_particles_blur_inuse) # position


mpm_v = ti.Vector.field(2, dtype=float, shape=n_particles + n_particles_blur_inuse) # velocity
mpm_blur_v = ti.Vector.field(2, dtype=float, shape=n_particles_blur_inuse) # velocity
C_mpm = ti.Matrix.field(2, 2, dtype=float, shape=n_particles + n_particles_blur_inuse) # affine velocity field
F_mpm = ti.Matrix.field(2, 2, dtype=float, shape=n_particles + n_particles_blur_inuse) # deformation gradient
T_mpm = ti.Matrix.field(2, 2, dtype=float, shape=n_particles + n_particles_blur_inuse) # deformation gradient

### 尝试用T把blur zone送回去然后看能不能correct一下

T_x_mpm_blur = ti.Vector.field(2, float, shape=n_particles_blur_inuse)  # d_psi / d_x
T_y_mpm_blur = ti.Vector.field(2, float, shape=n_particles_blur_inuse)  # d_psi / d_y

F_x_mpm_blur = ti.Vector.field(2, float, shape=n_particles_blur_inuse)  # d_psi / d_x
F_y_mpm_blur = ti.Vector.field(2, float, shape=n_particles_blur_inuse)  # d_psi / d_y


C_mpm_blur = ti.Matrix.field(2, 2, dtype=float, shape=n_particles_blur_inuse) # affine velocity field

T_x_mpm_blur_copy = ti.Vector.field(2, float, shape=n_particles_blur_inuse)  # d_psi / d_x
T_y_mpm_blur_copy = ti.Vector.field(2, float, shape=n_particles_blur_inuse)  # d_psi / d_y

F_x_mpm_blur_copy = ti.Vector.field(2, float, shape=n_particles_blur_inuse)  # d_psi / d_x
F_y_mpm_blur_copy = ti.Vector.field(2, float, shape=n_particles_blur_inuse)  # d_psi / d_y

mpm_x_temp = ti.Vector.field(2, dtype=float, shape=n_particles + n_particles_blur_inuse) # position
mpm_blur_x_temp = ti.Vector.field(2, dtype=float, shape=n_particles_blur_inuse) # position
mpm_v_temp = ti.Vector.field(2, dtype=float, shape=n_particles + n_particles_blur_inuse) # velocity
mpm_blur_v_temp = ti.Vector.field(2, dtype=float, shape=n_particles_blur_inuse) # velocity
C_temp = ti.Matrix.field(2, 2, dtype=float, shape=n_particles + n_particles_blur_inuse) # affine velocity field
F_temp = ti.Matrix.field(2, 2, dtype=float, shape=n_particles + n_particles_blur_inuse) # deformation gradient


mpm_blur_dist = ti.field(float, shape=n_particles_blur_inuse)

mpm_blur_near = ti.field(ti.int32, shape=n_particles_blur_inuse)


material = ti.field(dtype=float, shape=n_particles + n_particles_blur_inuse) # material id
material_blur = ti.field(dtype=float, shape=n_particles_blur_inuse) # material id

# grid_material = ti.Vector.field(2, dtype=float, shape=n_particles) # position
grid_v = ti.Vector.field(2, dtype=float, shape=(n_grid_x, n_grid_y)) # grid node momentum/velocity
grid_force = ti.Vector.field(2, dtype=float, shape=(n_grid_x, n_grid_y)) # grid node momentum/velocity
grid_force_copy = ti.Vector.field(2, dtype=float, shape=(n_grid_x, n_grid_y)) # grid node momentum/velocity
grid_F = ti.Matrix.field(2, 2, dtype=float, shape=(n_grid_x, n_grid_y)) # grid node momentum/velocity
grid_m2 = ti.field(dtype=float, shape=(n_grid_x, n_grid_y))
mpm_u_x = ti.field(float, shape=(n_grid_x + 1, n_grid_y)) # grid node momentum/velocity
mpm_u_y = ti.field(float, shape=(n_grid_x, n_grid_y + 1)) # grid node momentum/velocity

grid_m = ti.field(dtype=float, shape=(n_grid_x, n_grid_y)) # grid node mass
grid_material = ti.field(float, shape=(n_grid_x, n_grid_y))
grid_node = ti.field(float, shape=(n_grid_x, n_grid_y))
mpm_act = ti.field(dtype=float, shape=n_particles + n_particles_blur_inuse)

mpm_alpha = ti.field(dtype=float, shape=n_particles + n_particles_blur_inuse)
mpm_act_blur = ti.field(dtype=float, shape=n_particles_blur_inuse)
total_t = ti.field(dtype=float, shape=())

# levelset
# levelset = LevelSet([n_grid_x, n_grid_y], mpm_dx, n_particles + n_particles_blur, epsilon)
# levelset.redistance(mpm_x)
total_momentum = ti.Vector.field(2, dtype=float, shape=())
total_momentum_blur = ti.Vector.field(2, dtype=float, shape=())

@ti.kernel
def mpm_reset():
    group_size = n_particles
    width_x = x_width
    width_y = y_width
    # offset_x = 1.675
    offset_x = 0.3 - x_width / 2
    offset_y = 0.5 - y_width / 2
    for i in range(group_size):
        mpm_x[i] = [
                ti.random() * width_x + offset_x,
                ti.random() * width_y + offset_y,
        ]
        material[i] = 1
        F_mpm[i] = ti.Matrix([[1, 0], [0, 1]])
        T_mpm[i] = ti.Matrix([[1, 0], [0, 1]])
        C_mpm[i] = ti.Matrix.zero(float, 2, 2)
        
    group_size = (n_particles_blur - corner)
    area = (width_x + 2 * sample_epsilon) * (width_y + 2 * sample_epsilon)
    a_size = group_size * sample_epsilon* width_x * 2 / (area - width_x * width_y)
    a_size = a_size // 2 * 2
    a_size1 = int(a_size // 2)
    a_size2 = int(a_size - a_size1)
    b_size = group_size - a_size
    b_size = b_size // 2 * 2
    b_size1 = int(b_size // 2)
    b_size2 = int(group_size - a_size1 - a_size2 - b_size1)
    for i in range(b_size1): 
        mpm_blur_dist[i] = ti.random() * sample_epsilon
        mpm_blur_x[i] = [
                offset_x - mpm_blur_dist[i],
                ti.random() * width_y + offset_y,
        ]
        mpm_x[n_particles + i] = [
                offset_x,
                mpm_blur_x[i][1],
        ]
        
        mpm_blur_normal[i] = [-1.0, 0.0]
        F_mpm[n_particles + i] = ti.Matrix([[1, 0], [0, 1]])
        T_mpm[n_particles + i] = ti.Matrix([[1, 0], [0, 1]])
        C_mpm[n_particles + i] = ti.Matrix.zero(float, 2, 2)
    for i in range(b_size2): 
        mpm_blur_dist[i + b_size1] = ti.random() * sample_epsilon
        mpm_blur_x[i + b_size1] = [
                offset_x + width_x + mpm_blur_dist[i + b_size1],
                ti.random() * width_y + offset_y,
        ]
        mpm_x[n_particles + i + b_size1] = [
                offset_x + width_x,
                mpm_blur_x[i + b_size1][1],
        ]
        mpm_blur_normal[i + b_size1] = [1.0, 0.0]
        
        F_mpm[n_particles + b_size1 + i] = ti.Matrix([[1, 0], [0, 1]])
        T_mpm[n_particles + b_size1 + i] = ti.Matrix([[1, 0], [0, 1]])
        C_mpm[n_particles + b_size1 + i] = ti.Matrix.zero(float, 2, 2)
    
    offset = b_size1 + b_size2
    for i in range(a_size1): 
        mpm_blur_dist[i + offset] = ti.random() * sample_epsilon
        mpm_blur_x[i + offset] = [
                ti.random() * width_x + offset_x,
                offset_y + width_y + mpm_blur_dist[i + offset],
        ]
        mpm_x[n_particles + i + offset] = [
                mpm_blur_x[i + offset][0],
                offset_y + width_y,
        ]
        mpm_blur_normal[i + offset] = [0.0, 1.0]

        F_mpm[n_particles + offset + i] = ti.Matrix([[1, 0], [0, 1]])
        T_mpm[n_particles + offset + i] = ti.Matrix([[1, 0], [0, 1]])
        C_mpm[n_particles + offset + i] = ti.Matrix.zero(float, 2, 2)
        
    offset = b_size1 + b_size2 + a_size1
    for i in range(a_size2):
        mpm_blur_dist[i + offset] = ti.random() * sample_epsilon
        mpm_blur_x[i + offset] = [
                ti.random() * width_x + offset_x,
                offset_y - mpm_blur_dist[i + offset],
        ]
        mpm_x[n_particles + i + offset] = [
                mpm_blur_x[i + offset][0],
                offset_y,
        ]
        mpm_blur_normal[i + offset] = [0.0, -1.0]
        # material_blur[i + offset] = 2

        F_mpm[n_particles + offset + i] = ti.Matrix([[1, 0], [0, 1]])
        T_mpm[n_particles + offset + i] = ti.Matrix([[1, 0], [0, 1]])
        C_mpm[n_particles + offset + i] = ti.Matrix.zero(float, 2, 2)
        
    for i in range(corner // 4): 
        mpm_blur_x[group_size + i * 4] = [
                ti.random() * sample_epsilon + offset_x - sample_epsilon,
                ti.random() * sample_epsilon + offset_y - sample_epsilon,
        ]
        mpm_x[n_particles + group_size + i * 4] = [
                offset_x,
                offset_y,
        ]
        mpm_blur_dist[group_size + i * 4] = ti.math.length(mpm_blur_x[group_size + i * 4] - mpm_x[n_particles + group_size + i * 4])
        mpm_blur_normal[group_size + i * 4] = ti.math.normalize(mpm_blur_x[group_size + i * 4] - mpm_x[n_particles + group_size + i * 4])

        F_mpm[n_particles + group_size + i * 4] = ti.Matrix([[1, 0], [0, 1]])
        T_mpm[n_particles + group_size + i * 4] = ti.Matrix([[1, 0], [0, 1]])
        C_mpm[n_particles + group_size + i * 4] = ti.Matrix.zero(float, 2, 2)
        
        mpm_blur_x[group_size + i * 4 + 1] = [
                ti.random() * sample_epsilon + offset_x + width_x,
                ti.random() * sample_epsilon + offset_y - sample_epsilon,
        ]
        mpm_x[n_particles + group_size + i * 4 + 1] = [
                offset_x + width_x,
                offset_y,
        ]
        mpm_blur_dist[group_size + i * 4 + 1] = ti.math.length(mpm_blur_x[group_size + i * 4 + 1] - mpm_x[n_particles + group_size + i * 4 + 1])
        mpm_blur_normal[group_size + i * 4 + 1] = ti.math.normalize(mpm_blur_x[group_size + i * 4 + 1] - mpm_x[n_particles + group_size + i * 4 + 1])
        
        F_mpm[n_particles + group_size + i * 4 + 1] = ti.Matrix([[1, 0], [0, 1]])
        T_mpm[n_particles + group_size + i * 4 + 1] = ti.Matrix([[1, 0], [0, 1]])
        C_mpm[n_particles + group_size + i * 4 + 1] = ti.Matrix.zero(float, 2, 2)
        
        mpm_blur_x[group_size + i * 4 + 2] = [
                ti.random() * sample_epsilon + offset_x - sample_epsilon,
                ti.random() * sample_epsilon + offset_y + width_y,
        ]
        mpm_x[n_particles + group_size + i * 4 + 2] = [
                offset_x,
                offset_y + width_y,
        ]
        mpm_blur_dist[group_size + i * 4 + 2] = ti.math.length(mpm_blur_x[group_size + i * 4 + 2] - mpm_x[n_particles + group_size + i * 4 + 2])
        mpm_blur_normal[group_size + i * 4 + 2] = ti.math.normalize(mpm_blur_x[group_size + i * 4 + 2] - mpm_x[n_particles + group_size + i * 4 + 2])
        
        F_mpm[n_particles + group_size + i * 4 + 2] = ti.Matrix([[1, 0], [0, 1]])
        T_mpm[n_particles + group_size + i * 4 + 2] = ti.Matrix([[1, 0], [0, 1]])
        C_mpm[n_particles + group_size + i * 4 + 2] = ti.Matrix.zero(float, 2, 2)
        
        mpm_blur_x[group_size + i * 4 + 3] = [
                ti.random() * sample_epsilon + offset_x + width_x,
                ti.random() * sample_epsilon + offset_y + width_y,
        ]
        
        mpm_x[n_particles + group_size + i * 4 + 3] = [
                offset_x + width_x,
                offset_y + width_y,
        ]
        mpm_blur_dist[group_size + i * 4 + 3] = ti.math.length(mpm_blur_x[group_size + i * 4 + 3] - mpm_x[n_particles + group_size + i * 4 + 3])
        mpm_blur_normal[group_size + i * 4 + 3] = ti.math.normalize(mpm_blur_x[group_size + i * 4 + 3] - mpm_x[n_particles + group_size + i * 4 + 3])
        
        
        F_mpm[n_particles + group_size + i * 4 + 3] = ti.Matrix([[1, 0], [0, 1]])
        T_mpm[n_particles + group_size + i * 4 + 3] = ti.Matrix([[1, 0], [0, 1]])
        C_mpm[n_particles + group_size + i * 4 + 3] = ti.Matrix.zero(float, 2, 2)
        
    
    for p in mpm_x:

        this_offset = ((mpm_x[p][0] - offset_x))
        if (this_offset >= x_width * 0.6 and this_offset < x_width * 0.9):
            mpm_act[p] = ((y_width) - (mpm_x[p][1] - offset_y))

        else:
            mpm_act[p] = -1000
        
        
@ti.kernel
def resample_mpm_blur():
    for I in range(n_particles_blur_inuse):
        mpm_blur_x[I] = mpm_x[n_particles + I] + ti.math.normalize((T_mpm[n_particles + I].transpose() @ mpm_blur_normal[I])) * mpm_blur_dist[I]
        T_x_mpm_blur[I] = ti.Vector.unit(2, 0)
        T_y_mpm_blur[I] = ti.Vector.unit(2, 1)
        F_x_mpm_blur[I] = ti.Vector.unit(2, 0)
        F_y_mpm_blur[I] = ti.Vector.unit(2, 1)
        
@ti.kernel
def set_particle_mass():
    for I in ti.grouped(mpm_mass):
        mpm_mass[I] = p_rho
        
        
@ti.kernel
def set_particle_blur_mass():
    for I in ti.grouped(mpm_blur_mass):
        mpm_blur_mass[I] = p_rho_w
        
@ti.kernel
def set_fixed_heaviside():
    for I in mpm_x:
        fixed_heaviside_value[I] = levelset.heaviside(mpm_x[I])
mpm_reset()        
levelset = LevelSet([n_grid_x, n_grid_y], mpm_dx, n_particles, epsilon)
# levelset.redistance()

set_particle_mass()
set_particle_blur_mass()