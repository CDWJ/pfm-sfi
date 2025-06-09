
dim = 3

use_midpoint_vel = True
forward_update_T = True
use_APIC = True
use_APIC_smoke = True
save_particle_pos_numpy = False
save_particle_imp_numpy = False
save_frame_each_step = False
reinit_particle_pos = True
use_gravity = True

# encoder hyperparameters
min_res = (32, 16, 16)
num_levels = 4
feat_dim = 4
activate_threshold = 0.032
# neural buffer hyperparameters
N_iters = 1500
N_batch = 240000
success_threshold = 2.e-7  # 5.e-8
# simulation hyperparameters
res_x = 256
res_y = 128
res_z = 128
visualize_dt = 0.05

reinit_every = 12  # 20
reinit_every_grad_m = 4
ckpt_every = 1
CFL = 0.25
gravity = 0
from_frame = 0
total_frames = 1500
exp_name = "3D_cloth4"  # "3D_four_vorts_CF" #"3D_four_vorts_CF_MCM"
use_total_steps = False
total_steps = 1

particles_per_cell = 8
total_particles_num_ratio = 1
cell_max_particle_num_ratio = 1.6