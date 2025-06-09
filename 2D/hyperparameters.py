from math import pi
# some hyperparameters
dim = 2
testing = False
use_neural = False
save_ckpt = True
save_frame_each_step = False
use_BFECC = False
use_midpoint_vel = True
use_APIC = True
use_gravity = True
use_viscosity = True

forward_update_T = True
plot_particles = False
use_reseed_particles = False
reinit_particle_pos = True
dpi_vor = 512 if plot_particles else 512 // 8

# encoder hyperparameters
min_res = (128, 32)
num_levels = 4
feat_dim = 2
activate_threshold = 0.03
# neural buffer hyperparameters
N_iters = 2000
N_batch = 40000 #25000
success_threshold = 3.e-8
# simulation hyperparameters
res_x = 512
res_y = 128
visualize_dt = 0.04
reinit_every = 12
reinit_every_grad_m = 1
ckpt_every = 1
CFL = 0.5
gravity = 0
viscosity = 8e-6
from_frame = 0
total_frames = 1500
use_total_steps = False
total_steps = 500
exp_name = "2D_swim"

particles_per_cell = 16
total_particles_num_ratio = 1
cell_max_particle_num_ratio = 1.6

min_particles_per_cell_ratio = 1
min_particles_per_cell = int(particles_per_cell * min_particles_per_cell_ratio)
max_particles_per_cell_ratio = 1.6
max_particles_per_cell = int(particles_per_cell * max_particles_per_cell_ratio)

max_delete_particle_try_num = 100000


bound = 1
bunny_nodes = 0 
quality = 1
n_particles, n_grid_y = 10000 * quality**2, 128 * quality

n_grid_x = 2 * n_grid_y

mpm_dx, inv_dx = 1.0 / n_grid_y, float(n_grid_y)
epsilon = mpm_dx * 1
sample_epsilon = epsilon
y_width = 0.01
x_width = 0.25

n_particles_blur = int((1.0 - ((y_width * x_width) / ((y_width + 2*sample_epsilon) * (x_width + 2*sample_epsilon)))) * (n_particles / ((y_width * x_width) / ((y_width + 2*sample_epsilon)* (x_width + 2*sample_epsilon)))))

corner = int(sample_epsilon**2 / (((y_width + 2*sample_epsilon) * (x_width + 2*sample_epsilon)) - (y_width * x_width)) * n_particles_blur) * 4
n_particles_blur = ((n_particles_blur - corner) // 4 * 4 + corner)
n_particles_blur_inuse = n_particles_blur

mpm_dt = 1e-3 / quality
max_mpm_dt = 1e-3 / quality
p_vol, p_rho, p_rho_w = 1.0, 1.0, 1.0

E, nu = 3e3, 0.3 
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1+nu) * (1 - 2 * nu)) # Lame parameters - may change these later to model other materials
