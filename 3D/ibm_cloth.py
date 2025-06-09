


import taichi as ti
import numpy as np
import os
from gmesh import *
from framework import *
from length import *
import meshio
from bend import *
from math import pi
from hyperparameters import *

ibm_dx = 1.0 / res_y
# from utils import renderer, parser

# ti.init(arch=ti.gpu)

def obj_parser(filepath):
    mesh = meshio.read(filepath)
    v, f = mesh.points, mesh.cells_dict['triangle']
    return v, f.flatten()

filepath = os.path.join(os.getcwd(), 'assets', 'mesh', 'silk2.obj')
verts, faces = obj_parser(filepath)
rho = 2
# mesh = TrianMesh(verts, faces, dim=3, rho=1.0, scale=0.4, repose=(0.2, 0.3, 0.5))
mesh = TrianMesh(verts, faces, dim=3, rho=2.0, scale=0.6, repose=(0.2, 0.5, 0.35))

# fps = 60
# substep = 15
solve_iters = 50
dt = 0.0005
# dt = 1.0 / (fps * substep)


# xpbd = pbd_framework(g=g, n_vert=mesh.n_vert, v_p=mesh.v_p, dt=dt)
# length_cons = LengthCons(mesh.v_p,
#                                 mesh.v_p_ref,
#                                 mesh.e_i,
#                                 mesh.v_invm,
#                                 dt=dt,
#                                 alpha=0.1)
# bend_cons = Bend3D(mesh.v_p,
#                         mesh.v_p_ref,
#                         mesh.e_i,
#                         mesh.e_sidei,
#                         mesh.v_invm,
#                         dt=dt,
#                         alpha=10000)

g = ti.Vector([0.0, 5, -3])
xpbd = pbd_framework(g=g, n_vert=mesh.n_vert, v_p=mesh.v_p, dt=dt)
length_cons = LengthCons(mesh.v_p,
                                mesh.v_p_ref,
                                mesh.e_i,
                                mesh.v_invm,
                                dt=dt,
                                alpha=0.015)
bend_cons = Bend3D(mesh.v_p,
                        mesh.v_p_ref,
                        mesh.e_i,
                        mesh.e_sidei,
                        mesh.v_invm,
                        dt=dt,
                        alpha=6000)

xpbd.add_cons(length_cons)
xpbd.add_cons(bend_cons)
xpbd.init_rest_status()

indices = np.where(mesh.v_p.to_numpy()[:, 0] < 0.205)[0]

cons_vert_i = ti.field(dtype=ti.i32, shape=indices.shape[0])
cons_vert_i.from_numpy(indices)
cons_vert_p = ti.Vector.field(3, dtype=ti.f32, shape=indices.shape[0])
mesh.get_pos_by_index(n=indices.shape[0], index=cons_vert_i, pos=cons_vert_p)
mesh.set_fixed_point(n=indices.shape[0], index=cons_vert_i)
cons_pos = cons_vert_p.to_numpy()
cons_pos_init = np.copy(cons_pos)

pointForce = ti.Vector.field(3, dtype=ti.f32, shape=mesh.n_vert)
pointForce_copy = ti.Vector.field(3, dtype=ti.f32, shape=mesh.n_vert)
pointLocation_copy = ti.Vector.field(3, dtype=ti.f32, shape=mesh.n_vert)

mesh_vp_copy = ti.Vector.field(3, dtype=ti.f32, shape=mesh.n_vert)
mesh_vel_copy = ti.Vector.field(3, dtype=ti.f32, shape=mesh.n_vert)

@ti.func
def ibm_kernel(dis):
    weight = 0.0
    r = ti.abs(dis)
    if ti.abs(dis) <= 2:
        weight = 0.25 * (1 + ti.cos(pi * 0.5 * r))
    return weight

def Export(path, i: int):
  npL = mesh.v_p.to_numpy()
  npI = mesh.f_i.to_numpy()

  mesh_writer = ti.tools.PLYWriter(num_vertices=mesh.n_vert, num_faces=mesh.n_face, face_type="tri")
  mesh_writer.add_vertex_pos(npL[:, 0] * res_y - 0.5, npL[:, 1] * res_y - 0.5, npL[:, 2] * res_y - 0.5)
  mesh_writer.add_faces(npI)

  mesh_writer.export_frame_ascii(i, f'{path}/S.ply')
    
    
    
@ti.kernel
def spread_force(u_x:ti.template(), u_y:ti.template(), u_z:ti.template(), dt:float):
    for i in range(mesh.n_vert):
        # horizontal impulse
        pos = mesh.v_p[i] / ibm_dx
                
        base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1) - 0.5 * ti.Vector.unit(dim, 2))
        for offset in ti.grouped(ti.ndrange(*((-3, 4),) * dim)):
            face_id = base_face_id + offset
            if 0 <= face_id[0] <= res_x and 0 <= face_id[1] < res_y and 0 <= face_id[2] < res_z:
                weight = ibm_kernel(pos[0] - face_id[0]) * ibm_kernel(pos[1] - face_id[1] - 0.5) * ibm_kernel(pos[2] - face_id[2] - 0.5)
                u_x[face_id] += pointForce[i][0] * weight * dt

        # vertical impulse
        # base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1)) + ti.Vector.unit(dim, 1)
        base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0) - 0.5 * ti.Vector.unit(dim, 2))
        for offset in ti.grouped(ti.ndrange(*((-3, 4),) * dim)):
            face_id = base_face_id + offset
            if 0 <= face_id[0] < res_x and 0 <= face_id[1] <= res_y and 0 <= face_id[2] < res_z:
                weight = ibm_kernel(pos[0] - face_id[0] - 0.5) * ibm_kernel(pos[1] - face_id[1]) * ibm_kernel(pos[2] - face_id[2] - 0.5)
                u_y[face_id] += pointForce[i][1] * weight * dt

                # psi_y_grid[face_id] += (psi[i] + T.transpose() @ dpos) * weight

        base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0) - 0.5 * ti.Vector.unit(dim, 1))
        for offset in ti.grouped(ti.ndrange(*((-3, 4),) * dim)):
            face_id = base_face_id + offset
            if 0 <= face_id[0] < res_x and 0 <= face_id[1] < res_y and 0 <= face_id[2] <= res_z:
                weight = ibm_kernel(pos[0] - face_id[0] - 0.5) * ibm_kernel(pos[1] - face_id[1] - 0.5) * ibm_kernel(pos[2] - face_id[2])
                u_z[face_id] += pointForce[i][2] * weight * dt
                
                
@ti.kernel
def spread_rho(u_x:ti.template(), u_y:ti.template(), u_z:ti.template(), dt:float):
    for i in range(mesh.n_vert):
        # horizontal impulse
        pos = mesh.v_p[i] / ibm_dx
                
        base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0) - 0.5 * ti.Vector.unit(dim, 1) - 0.5 * ti.Vector.unit(dim, 2))
        for offset in ti.grouped(ti.ndrange(*((-3, 4),) * dim)):
            face_id = base_face_id + offset
            if 0 <= face_id[0] < res_x and 0 <= face_id[1] < res_y and 0 <= face_id[2] < res_z:
                weight = ibm_kernel(pos[0] - face_id[0] - 0.5) * ibm_kernel(pos[1] - face_id[1] - 0.5) * ibm_kernel(pos[2] - face_id[2] - 0.5)
                rho[face_id] += p_rho * weight
                
                
@ti.func
def sample_ibm_u(u_x, u_y, u_z, p, dx):
    vel = ti.Vector([0.0, 0.0, 0.0])
    pos = p / dx
    base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1) - 0.5 * ti.Vector.unit(dim, 2))
    for offset in ti.grouped(ti.ndrange(*((-3, 4),) * dim)):
        face_id = base_face_id + offset
        if 0 <= face_id[0] <= res_x and 0 <= face_id[1] < res_y and 0 <= face_id[2] < res_z:
            weight = ibm_kernel(pos[0] - face_id[0]) * ibm_kernel(pos[1] - face_id[1] - 0.5) * ibm_kernel(pos[2] - face_id[2] - 0.5)
            vel[0] += u_x[face_id] * weight

    base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0) - 0.5 * ti.Vector.unit(dim, 2))
    for offset in ti.grouped(ti.ndrange(*((-3, 4),) * dim)):
        face_id = base_face_id + offset
        if 0 <= face_id[0] < res_x and 0 <= face_id[1] <= res_y and 0 <= face_id[2] < res_z:
            weight = ibm_kernel(pos[0] - face_id[0] - 0.5) * ibm_kernel(pos[1] - face_id[1]) * ibm_kernel(pos[2] - face_id[2] - 0.5)
            vel[1] += u_y[face_id] * weight

    base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0) - 0.5 * ti.Vector.unit(dim, 1))
    for offset in ti.grouped(ti.ndrange(*((-3, 4),) * dim)):
        face_id = base_face_id + offset
        if 0 <= face_id[0] < res_x and 0 <= face_id[1] < res_y and 0 <= face_id[2] <= res_z:
            weight = ibm_kernel(pos[0] - face_id[0] - 0.5) * ibm_kernel(pos[1] - face_id[1] - 0.5) * ibm_kernel(pos[2] - face_id[2])
            vel[2] += u_z[face_id] * weight
    return vel
    
@ti.kernel
def advect_ibm(u_x:ti.template(), u_y:ti.template(), u_z:ti.template(), dt:float, change_p: float):
    for i in range(mesh.n_vert):
        # first
        u1 = sample_ibm_u(u_x, u_y, u_z, mesh.v_p[i], ibm_dx)
        psi_x1 = mesh.v_p[i] + 0.5 * dt * u1  # advance 0.5 steps
        # second
        u2 = sample_ibm_u(u_x, u_y, u_z, psi_x1, ibm_dx)
        psi_x2 = mesh.v_p[i] + 0.5 * dt * u2  # advance 0.5 again
        # third
        u3 = sample_ibm_u(u_x, u_y, u_z, psi_x2, ibm_dx)
        psi_x3 = mesh.v_p[i] + 1.0 * dt * u3
        # fourth
        u4 = sample_ibm_u(u_x, u_y, u_z, psi_x3, ibm_dx)
        
        if change_p == 1:
            mesh.v_p[i] = mesh.v_p[i] + dt * 1. / 6 * (u1 + 2 * u2 + 2 * u3 + u4)
        xpbd.v_v[i] = sample_ibm_u(u_x, u_y, u_z, mesh.v_p[i], ibm_dx) + g*dt
        
def solve_for_xpbd(dt):
    # xpbd.make_prediction()
    cons_vert_p.from_numpy(cons_pos)
    mesh.set_pos_by_index(n=indices.shape[0], index=cons_vert_i, pos=cons_vert_p)
    length_cons.update_alpha(dt)
    xpbd.dt= dt
    bend_cons.update_alpha(dt)
    xpbd.preupdate_cons()
    for _ in range(solve_iters):
        xpbd.update_cons()
    xpbd.update_vel()
    pointForce_copy.copy_from(pointForce)

# @ti.kernel
# def store_cache_vel(dt:float, pos:ti.template()):
#     # xpbd.v_v_cache.copy_from(xpbd.v_v)
#     for i in range(mesh.n_vert):
#         xpbd.v_v_cache[i] = (xpbd.v_p[i] - pos[i]) / dt


def store_cache_vel():
    xpbd.v_v_cache.copy_from(xpbd.v_v)

@ti.kernel
def update_force(dt:float):
    for i in range(mesh.n_vert):
        pointForce[i] = (xpbd.v_v[i] - xpbd.v_v_cache[i]) / dt
        # pointForce[i] = (xpbd.v_v[i] - mesh_vel_copy[i]) / dt
        
        # if ti.math.length(pointForce[i] - pointForce_copy[i]) > 0.2 * ti.math.length(pointForce_copy[i]):
        #     pointForce[i] = pointForce[i] * 0.2
    # for _ in range(4):
    #     for k in range(mesh.n_face):
    #       p1 = mesh.f_i[k * 3]
    #       p2 = mesh.f_i[k * 3 + 1]
    #       p3 = mesh.f_i[k * 3 + 2]
    #       pointForcet = (pointForce[p1] + pointForce[p2] + pointForce[p3]) / 3
    #       pointForce[p1] = pointForcet
    #       pointForce[p2] = pointForcet
    #       pointForce[p3] = pointForcet
        
      
        
@ti.kernel
def update_vel_explicit(dt:float):
    for i in range(mesh.n_vert):
        xpbd.v_v[i] = (xpbd.v_p[i] - pointLocation_copy[i]) / dt
    
# Frame = 0
# while True:
#     for sub in range(substep):
#         # init XPBD solver
#         xpbd.make_prediction()

#         # set fixed points
#         # cons_pos[1, 0] = cons_pos_init[1, 0] - np.math.sin(
#         #     2 * np.math.pi * tirender.time / 5.0)**2 * 0.6
#         cons_vert_p.from_numpy(cons_pos)
#         mesh.set_pos_by_index(n=2, index=cons_vert_i, pos=cons_vert_p)

#         # solve constraints
#         xpbd.preupdate_cons()
#         for _ in range(solve_iters):
#             xpbd.update_cons()
#         xpbd.update_vel()

#     if Frame % 3 == 0:
#         Export(Frame // 3)
#     Frame += 1
    


