import taichi as ti
from hyperparameters import *
from taichi_utils import *
dx = 1. / res_y
half_dx = 0.5 * dx
p_vol = 1
@ti.kernel
def P2G_init(particles_pos: ti.template(), mpm_blur_x: ti.template(), mpm_x: ti.template(), rho_x: ti.template(), rho_y: ti.template(), p2g_weight_x: ti.template(), p2g_weight_y: ti.template(), particles_active: ti.template(), levelset: ti.template(), p_rho_w:float, p_rho:float):
    rho_x.fill(0.0)
    rho_y.fill(0.0)
    levelset.phi.fill(0.0)
    p2g_weight_x.fill(0.0)
    p2g_weight_y.fill(0.0)

    for i in particles_pos:
        if particles_active[i] == 1:
            # horizontal impulse
            pos = particles_pos[i] / dx
            # base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0)) + ti.Vector.unit(dim, 0)
            base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1))
            for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
                face_id = base_face_id + offset
                if 0 <= face_id[0] <= res_x and 0 <= face_id[1] < res_y:
                    weight = N_2(pos[0] - face_id[0]) * N_2(pos[1] - face_id[1] - 0.5)
                    rho_x[face_id] += weight * (p_rho_w) * p_vol
                    p2g_weight_x[face_id] += weight


            # vertical impulse
            # base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1)) + ti.Vector.unit(dim, 1)
            base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0))
            for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
                face_id = base_face_id + offset
                if 0 <= face_id[0] < res_x and 0 <= face_id[1] <= res_y:
                    weight = N_2(pos[0] - face_id[0] - 0.5) * N_2(pos[1] - face_id[1])
                    rho_y[face_id] += weight * (p_rho_w) * p_vol
                    p2g_weight_y[face_id] += weight

                    
    for i in mpm_blur_x:
        # horizontal impulse
        pos = mpm_blur_x[i] / dx
        # base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0)) + ti.Vector.unit(dim, 0)
        base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1))
        for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
            face_id = base_face_id + offset
            if 0 <= face_id[0] <= res_x and 0 <= face_id[1] < res_y:
                weight = N_2(pos[0] - face_id[0]) * N_2(pos[1] - face_id[1] - 0.5)
                rho_x[face_id] += weight * (p_rho_w) * p_vol
                p2g_weight_x[face_id] += weight


        # vertical impulse
        # base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1)) + ti.Vector.unit(dim, 1)
        base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0))
        for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
            face_id = base_face_id + offset
            if 0 <= face_id[0] < res_x and 0 <= face_id[1] <= res_y:
                weight = N_2(pos[0] - face_id[0] - 0.5) * N_2(pos[1] - face_id[1])
                rho_y[face_id] += weight * (p_rho_w) * p_vol
                p2g_weight_y[face_id] += weight

                    
                    
    for i in mpm_x:
        # horizontal impulse
        pos = mpm_x[i] / dx
        # base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0)) + ti.Vector.unit(dim, 0)
        base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1))
        for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
            face_id = base_face_id + offset
            if 0 <= face_id[0] <= res_x and 0 <= face_id[1] < res_y:
                weight = N_2(pos[0] - face_id[0]) * N_2(pos[1] - face_id[1] - 0.5)
                rho_x[face_id] += weight * (p_rho) * p_vol
                p2g_weight_x[face_id] += weight

        # vertical impulse
        # base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1)) + ti.Vector.unit(dim, 1)
        base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0))
        for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
            face_id = base_face_id + offset
            if 0 <= face_id[0] < res_x and 0 <= face_id[1] <= res_y:
                weight = N_2(pos[0] - face_id[0] - 0.5) * N_2(pos[1] - face_id[1])
                rho_y[face_id] += weight * (p_rho) * p_vol
                p2g_weight_y[face_id] += weight
    

    for I in ti.grouped(p2g_weight_x):
        if p2g_weight_x[I] > 0:
            rho_x[I] /= p2g_weight_x[I]
            # imp_x[I] = u_x[I]

    for I in ti.grouped(p2g_weight_y):
        if p2g_weight_y[I] > 0:
            rho_y[I] /= p2g_weight_y[I]
            # imp_y[I] = u_y[I]
            
            
    for I in ti.grouped(levelset.phi):
        i = I[0]
        j = I[1]
        levelset.phi[I] = 0.25 * (rho_x[i, j] + rho_x[i + 1, j] + rho_y[i, j] + rho_y[i, j + 1])
        

@ti.kernel
def P2G_visc(particles_imp: ti.template(), particles_imp_mpm_blur: ti.template(), particles_pos: ti.template(), mpm_blur_x: ti.template(), u_x: ti.template(), u_y: ti.template(), p2g_weight_x: ti.template(), p2g_weight_y: ti.template(), C_x: ti.template(), C_y: ti.template(), C_x_mpm_blur: ti.template(), C_y_mpm_blur: ti.template(), particles_active: ti.template()):
    u_x.fill(0.0)
    u_y.fill(0.0)
    p2g_weight_x.fill(0.0)
    p2g_weight_y.fill(0.0)

    for i in particles_imp:
        if particles_active[i] == 1:
            # horizontal impulse
            pos = particles_pos[i] / dx
            # base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0)) + ti.Vector.unit(dim, 0)
            base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1))
            for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
                face_id = base_face_id + offset
                if 0 <= face_id[0] <= res_x and 0 <= face_id[1] < res_y:
                    weight = N_2(pos[0] - face_id[0]) * N_2(pos[1] - face_id[1] - 0.5)
                    dpos = ti.Vector([face_id[0] - pos[0], face_id[1] + 0.5 - pos[1]]) * dx
                    p2g_weight_x[face_id] += weight
                    delta = C_x[i].dot(dpos)
                    # print(particles_imp[i][0], weight, delta)
                    if use_APIC:
                        u_x[face_id] += (particles_imp[i][0] + delta) * weight
                    else:
                        u_x[face_id] += (particles_imp[i][0]) * weight

            # vertical impulse
            # base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1)) + ti.Vector.unit(dim, 1)
            base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0))
            for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
                face_id = base_face_id + offset
                if 0 <= face_id[0] < res_x and 0 <= face_id[1] <= res_y:
                    weight = N_2(pos[0] - face_id[0] - 0.5) * N_2(pos[1] - face_id[1])
                    dpos = ti.Vector([face_id[0] + 0.5 - pos[0], face_id[1] - pos[1]]) * dx
                    p2g_weight_y[face_id] += weight
                    delta = C_y[i].dot(dpos)
                    if use_APIC:
                        u_y[face_id] += (particles_imp[i][1] + delta) * weight
                    else:
                        u_y[face_id] += (particles_imp[i][1]) * weight
                    
    for i in mpm_blur_x:
        # horizontal impulse
        pos = mpm_blur_x[i] / dx
        # base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0)) + ti.Vector.unit(dim, 0)
        base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1))
        for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
            face_id = base_face_id + offset
            if 0 <= face_id[0] <= res_x and 0 <= face_id[1] < res_y:
                weight = N_2(pos[0] - face_id[0]) * N_2(pos[1] - face_id[1] - 0.5)
                dpos = ti.Vector([face_id[0] - pos[0], face_id[1] + 0.5 - pos[1]]) * dx
                p2g_weight_x[face_id] += weight
                delta = C_x_mpm_blur[i].dot(dpos)
                # print(particles_imp[i][0], weight, delta)
                if use_APIC:
                    u_x[face_id] += (particles_imp_mpm_blur[i][0] + delta) * weight
                else:
                    u_x[face_id] += (particles_imp_mpm_blur[i][0]) * weight

        # vertical impulse
        # base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1)) + ti.Vector.unit(dim, 1)
        base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0))
        for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
            face_id = base_face_id + offset
            if 0 <= face_id[0] < res_x and 0 <= face_id[1] <= res_y:
                weight = N_2(pos[0] - face_id[0] - 0.5) * N_2(pos[1] - face_id[1])
                dpos = ti.Vector([face_id[0] + 0.5 - pos[0], face_id[1] - pos[1]]) * dx
                p2g_weight_y[face_id] += weight
                delta = C_y_mpm_blur[i].dot(dpos)
                if use_APIC:
                    u_y[face_id] += (particles_imp_mpm_blur[i][1] + delta) * weight
                else:
                    u_y[face_id] += (particles_imp_mpm_blur[i][1]) * weight
                    
    

    for I in ti.grouped(p2g_weight_x):
        if p2g_weight_x[I] > 0:
            scale = 1. / p2g_weight_x[I]
            u_x[I] *= scale


    for I in ti.grouped(p2g_weight_y):
        if p2g_weight_y[I] > 0:
            scale = 1. / p2g_weight_y[I]
            u_y[I] *= scale


        
@ti.kernel
def sample_p_and_p2g(particle_pressure:ti.template(), particle_pos:ti.template(), pressure:ti.template(), u_x:ti.template(), u_y:ti.template(), p2g_weight_x:ti.template(), p2g_weight_y:ti.template(), force_x:ti.template(), force_y:ti.template()):
    force_x.fill(0.0)
    force_y.fill(0.0)
    p2g_weight_x.fill(0.0)
    p2g_weight_y.fill(0.0)
    
    for I in range(n_particles):
        p = particle_pos[I]
        nouse, grad_p, _ = interp_grad_2(pressure, p, dx, BL_x=0.5, BL_y=0.5)
        particle_pressure[I] = grad_p / p_rho
        
    for i in range(n_particles):
        # horizontal impulse
        pos = particle_pos[i] / dx
        # base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0)) + ti.Vector.unit(dim, 0)
        base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1))
        for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
            face_id = base_face_id + offset
            if 0 <= face_id[0] <= res_x and 0 <= face_id[1] < res_y:
                weight = N_2(pos[0] - face_id[0]) * N_2(pos[1] - face_id[1] - 0.5)
                dpos = ti.Vector([face_id[0] - pos[0], face_id[1] + 0.5 - pos[1]]) * dx

                force_x[face_id] += (p_rho) * p_vol * weight * ((particle_pressure[i][0]))
                p2g_weight_x[face_id] += weight

        # vertical impulse
        # base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1)) + ti.Vector.unit(dim, 1)
        base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0))
        for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
            face_id = base_face_id + offset
            if 0 <= face_id[0] < res_x and 0 <= face_id[1] <= res_y:
                weight = N_2(pos[0] - face_id[0] - 0.5) * N_2(pos[1] - face_id[1])
                dpos = ti.Vector([face_id[0] + 0.5 - pos[0], face_id[1] - pos[1]]) * dx
                force_y[face_id] += (p_rho) * p_vol * weight * ((particle_pressure[i][1]))

                p2g_weight_y[face_id] += weight

            
    for I in ti.grouped(p2g_weight_x):
        if p2g_weight_x[I] > 0:
            force_x[I] /= p2g_weight_x[I]
            u_x[I] += force_x[I]

    for I in ti.grouped(p2g_weight_y):
        if p2g_weight_y[I] > 0:
            force_y[I] /= p2g_weight_y[I]
            u_y[I] += force_y[I]
            
    

    
@ti.kernel
def sample_gradm_and_p2g(gradm_x:ti.template(), gradm_y:ti.template(), gradm_x_mpm_blur:ti.template(), gradm_y_mpm_blur:ti.template(), gradm_x_grid:ti.template(), gradm_y_grid:ti.template(), particles_pos:ti.template(), mpm_blur_x:ti.template(), u_x:ti.template(), u_y:ti.template(), dx:ti.template(), p2g_weight_x: ti.template(), p2g_weight_y: ti.template(), particles_active: ti.template()):
    gradm_x_grid.fill(0.0)
    gradm_y_grid.fill(0.0)
    p2g_weight_x.fill(0.0)
    p2g_weight_y.fill(0.0)
    
    for I in gradm_x:
        nouse, _, new_C_x, new_C_y = interp_u_MAC_grad_imp(u_x, u_y, particles_pos[I], dx)
        gradm_x[I] = new_C_x
        gradm_y[I] = new_C_y
        
    for I in gradm_x_mpm_blur:
        nouse, _, new_C_x, new_C_y = interp_u_MAC_grad_imp(u_x, u_y, mpm_blur_x[I], dx)
        gradm_x_mpm_blur[I] = new_C_x
        gradm_y_mpm_blur[I] = new_C_y
        
    for i in gradm_x:
        if particles_active[i] >= 1:
            # horizontal impulse grad
            pos = particles_pos[i] / dx
            base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1))
            for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
                face_id = base_face_id + offset
                if 0 <= face_id[0] <= res_x and 0 <= face_id[1] < res_y:
                    weight = N_2(pos[0] - face_id[0]) * N_2(pos[1] - face_id[1] - 0.5)
                    p2g_weight_x[face_id] += weight
                    gradm_x_grid[face_id] += gradm_x[i] * weight

            # vertical impulse
            # base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1)) + ti.Vector.unit(dim, 1)
            base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0))
            for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
                face_id = base_face_id + offset
                if 0 <= face_id[0] < res_x and 0 <= face_id[1] <= res_y:
                    weight = N_2(pos[0] - face_id[0] - 0.5) * N_2(pos[1] - face_id[1])
                    p2g_weight_y[face_id] += weight
                    gradm_y_grid[face_id] += gradm_y[i] * weight
                    
    for i in gradm_x_mpm_blur:
        # horizontal impulse grad
        pos = mpm_blur_x[i] / dx
        base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1))
        for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
            face_id = base_face_id + offset
            if 0 <= face_id[0] <= res_x and 0 <= face_id[1] < res_y:
                weight = N_2(pos[0] - face_id[0]) * N_2(pos[1] - face_id[1] - 0.5)
                p2g_weight_x[face_id] += weight
                gradm_x_grid[face_id] += gradm_x_mpm_blur[i] * weight

        # vertical impulse
        # base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1)) + ti.Vector.unit(dim, 1)
        base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0))
        for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
            face_id = base_face_id + offset
            if 0 <= face_id[0] < res_x and 0 <= face_id[1] <= res_y:
                weight = N_2(pos[0] - face_id[0] - 0.5) * N_2(pos[1] - face_id[1])
                p2g_weight_y[face_id] += weight
                gradm_y_grid[face_id] += gradm_y_mpm_blur[i] * weight

    for I in ti.grouped(p2g_weight_x):
        if p2g_weight_x[I] > 0:
            scale = 1. / p2g_weight_x[I]
            gradm_x_grid[I] *= scale
            
    for I in ti.grouped(p2g_weight_y):
        if p2g_weight_y[I] > 0:
            scale = 1. / p2g_weight_y[I]
            gradm_y_grid[I] *= scale
            
@ti.kernel
def P2G_solid(particles_imp: ti.template(), particles_imp_mpm_blur: ti.template(), mpm_v: ti.template(), particles_pos: ti.template(), mpm_blur_x: ti.template(), mpm_x: ti.template(), u_x: ti.template(), u_y: ti.template(), rho_x: ti.template(), rho_y: ti.template(), p2g_weight_x: ti.template(), p2g_weight_y: ti.template(), C_x: ti.template(), C_y: ti.template(), C_x_mpm_blur: ti.template(), C_y_mpm_blur: ti.template(), particles_active: ti.template(), F_mpm: ti.template(), C_mpm: ti.template(), mpm_act: ti.template(), levelset: ti.template(), p_rho_w:float, p_rho:float, curr_dt: float, curr_dt_solid:float, force_x: ti.template(), force_y:ti.template(), F_x:ti.template(), F_y:ti.template(), F_x_mpm_blur:ti.template(), F_y_mpm_blur:ti.template(), p2g_weight_x2:ti.template(), p2g_weight_y2:ti.template(), C_mpm_blur:ti.template()):
    u_x.fill(0.0)
    u_y.fill(0.0)
    rho_x.fill(0.0)
    rho_y.fill(0.0)
    levelset.phi.fill(0.0)
    p2g_weight_x.fill(0.0)
    p2g_weight_y.fill(0.0)
    p2g_weight_x2.fill(0.0)
    p2g_weight_y2.fill(0.0)
    force_x.fill(0.0)
    force_y.fill(0.0)
    for i in mpm_blur_x:
        mu, la = mu_0, lambda_0
        mu = 0.0
        # horizontal impulse
        pos = mpm_blur_x[i] / dx
        # base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0)) + ti.Vector.unit(dim, 0)
        base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1))
        
        F_inuse = ti.Matrix.cols([F_x_mpm_blur[i], F_y_mpm_blur[i]])
        U, sig, V = ti.svd(F_inuse)
        J = 1.0
        for d in ti.static(range(2)):
            new_sig = sig[d, d]
            # Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        F_inuse = ti.Matrix.identity(float, 2) * ti.sqrt(J)
        kirchoff = kirchoff_FCR(F_inuse, U@V.transpose(), J, mu, la) #eq 52 
            
        # F_v = ti.Matrix.rows([gradm_x_mpm_blur[i], gradm_y_mpm_blur[i]])
        for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
            face_id = base_face_id + offset
            if 0 <= face_id[0] <= res_x and 0 <= face_id[1] < res_y:
                weight = N_2(pos[0] - face_id[0]) * N_2(pos[1] - face_id[1] - 0.5)
                dpos = ti.Vector([face_id[0] - pos[0], face_id[1] + 0.5 - pos[1]]) * dx
                rho_x[face_id] += weight * (p_rho_w) * p_vol
                p2g_weight_x[face_id] += p_vol * weight
                p2g_weight_x2[face_id] += (p_rho_w) *weight
                # rho_x[face_id] += weight
                delta = C_x_mpm_blur[i].dot(dpos)
                dw_x = 1. / dx * dN_2(pos[0] - face_id[0]) * N_2(pos[1] - face_id[1] - 0.5)
                dw_y = 1. / dx * N_2(pos[0] - face_id[0]) * dN_2(pos[1] - face_id[1] - 0.5)
                dweight = ti.Vector([dw_x, dw_y])
                # print(particles_imp[i][0], weight, delta)
                # force = F_v @ dweight
                ela_force = -kirchoff @ dweight
                if use_APIC:
                    u_x[face_id] += (p_rho_w) * p_vol * (particles_imp_mpm_blur[i] + C_mpm_blur[i] @ dpos)[0] * weight
                    # u_x[face_id] += (p_rho_w) * p_vol * ((particles_imp_mpm_blur[i])[0] + delta) * weight
                else:
                    u_x[face_id] += (particles_imp_mpm_blur[i][0]) * weight
                # vis_force_x[face_id] += curr_dt * viscosity * force[0]
                force_x[face_id] += curr_dt *  ela_force[0] * 0

        # vertical impulse
        # base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1)) + ti.Vector.unit(dim, 1)
        base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0))
        for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
            face_id = base_face_id + offset
            if 0 <= face_id[0] < res_x and 0 <= face_id[1] <= res_y:
                weight = N_2(pos[0] - face_id[0] - 0.5) * N_2(pos[1] - face_id[1])
                dpos = ti.Vector([face_id[0] + 0.5 - pos[0], face_id[1] - pos[1]]) * dx
                rho_y[face_id] += weight * (p_rho_w) * p_vol
                p2g_weight_y[face_id] +=  p_vol * weight
                p2g_weight_y2[face_id] += (p_rho_w) *weight
                # rho_y[face_id] += weight
                delta = C_y_mpm_blur[i].dot(dpos)
                dw_x = 1. / dx * dN_2(pos[0] - face_id[0] - 0.5) * N_2(pos[1] - face_id[1])
                dw_y = 1. / dx * N_2(pos[0] - face_id[0] - 0.5) * dN_2(pos[1] - face_id[1])
                dweight = ti.Vector([dw_x, dw_y])
                ela_force = -kirchoff @ dweight
                # force = F_v @ dweight
                if use_APIC:
                    u_y[face_id] += (p_rho_w) * p_vol * (particles_imp_mpm_blur[i] + C_mpm_blur[i] @ dpos)[1] * weight
                    # u_y[face_id] += (p_rho_w) * p_vol * ((particles_imp_mpm_blur[i])[1] + delta) * weight
                else:
                    u_y[face_id] += (particles_imp_mpm_blur[i][1]) * weight
                # vis_force_y[face_id] += curr_dt * viscosity * force[1]
                force_y[face_id] += curr_dt *  ela_force[1] * 0
                    
                    
    for i in mpm_x:


        lamb = ti.math.log(2.2/0.021)
        F_inuse = F_mpm[i]
        # H = levelset.heaviside(mpm_x[p])
        Xy = mpm_act[i]
        # alpha = -lamb * Xy * 8 * ti.sin(0.25 * pi * curr_dt_solid)**8
        # F_a = ti.Matrix([[ti.exp(-alpha), 0.0], [0.0, ti.exp(alpha)]])
        # # F_a = ti.Matrix([[0.0, ti.exp(-alpha)], [ti.exp(alpha), 0.0]])
        # F_inuse = F_inuse @ F_a.inverse()
        mu, la = mu_0, lambda_0
                
        T = 2
        time = (curr_dt_solid % T)
        md = (curr_dt_solid // T) % 2
        if Xy != -1000:
            alpha = 1.0
            if time < T / 2:
            # # #     # act = -(act - x_width)
            # # # #     if (total_t[None] // T) % 2 == 0:
            # # # # # alpha = 1.0 - 0.05 * ti.sin(2 * pi * time / T) * ti.exp(-act / (x_width / 3) * (1 - H))
            # # # #         alpha = 1.0 - 0.1 * ti.exp(-act / (x_width / 3))
            # # # #     else:
            # # # #         alpha = 1.0 - 0.025 * ti.exp(-act / (x_width / 3))
            #     # if md == 0:
            #     if Xy == -2000:
            #         # alpha = 1.0 - 0.15 * ti.abs(ti.sin(2 * pi * time / T)) * ti.exp(-(Xy) / (y_width / 3))
            #         mu *= 0.95
            #         la *= 0.95
            #     else:
                alpha = 1.0 - 0.25 * ti.abs(ti.sin(2 * pi * time / T)) * ti.exp(-(Xy) / (y_width / 3))
                # else:
                    # alpha = 1.0 - 0.2 * ti.sin(2 * pi * time / T) * ti.exp((Xy - y_width) / (y_width / 3))
                    # alpha = 1
            else:
                # alpha = 1.0 - 0.2 * (ti.sin(2 * pi * time / T)) * ti.exp(-(mpm_act[p]) / (x_width))
                # if md == 0:
                # if Xy == -2000:
                #     mu *= 0.95
                #     la *= 0.95
                #     # alpha = 1.0 - 0.15 * ti.abs(ti.sin(2 * pi * time / T)) * ti.exp(-(Xy) / (y_width / 3))
                # else:
                alpha = 1.0 - 0.25 * ti.abs(ti.sin(2 * pi * time / T)) * ti.exp((Xy - y_width) / (y_width / 3))
                    # alpha = 1
                # else:
                    # alpha = 1.0 - 0.2 * ti.sin(2 * pi * time / T) * ti.exp(-(Xy) / (y_width / 3))
                    # alpha = 1
            # alpha = 1.0 - 0.15 * (ti.sin(2 * pi * time / T)) * ti.exp((Xy - y_width) / (y_width / 3))
            F_a = ti.Matrix([[1.0 / alpha, 0.0], [0.0, alpha]])
            # F_a = ti.Matrix([[alpha, 0.0], [0.0, 1.0/alpha]])
            
            # F_a = ti.Matrix([[ti.exp(-alpha), 0.0], [0.0, ti.exp(alpha)]])
            F_inuse = F_inuse @ (F_a.inverse())


        # mu, la = mu_0, lambda_0

        U, sig, V = ti.svd(F_inuse)
        J = 1.0

        for d in ti.static(range(2)):
            J *= sig[d, d]

        # Compute Kirchoff Stress
        # kirchoff = kirchoff_FCR(F_inuse, U@V.transpose(), J, mu, la) #eq 52 
        FFt = F_inuse @ F_inuse.transpose()
        
        tr = 0.0
        for d in ti.static(range(2)):
            tr += FFt[d, d]
        
        kirchoff = kirchoff_FCR(F_inuse, U@V.transpose(), J, mu, la) #eq 52 
        # kirchoff = 2 * (FFt - ti.Matrix([[1.0, 0.0], [0.0, 1.0]]) * (tr + 1.0) / 3.0)

        # horizontal impulse
        pos = mpm_x[i] / dx
        # base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0)) + ti.Vector.unit(dim, 0)
        base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1))
        for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
            face_id = base_face_id + offset
            if 0 <= face_id[0] <= res_x and 0 <= face_id[1] < res_y:
                weight = N_2(pos[0] - face_id[0]) * N_2(pos[1] - face_id[1] - 0.5)
                dpos = ti.Vector([face_id[0] - pos[0], face_id[1] + 0.5 - pos[1]]) * dx
                
                dw_x = 1. / dx * dN_2(pos[0] - face_id[0]) * N_2(pos[1] - face_id[1] - 0.5)
                dw_y = 1. / dx * N_2(pos[0] - face_id[0]) * dN_2(pos[1] - face_id[1] - 0.5)
                dweight = ti.Vector([dw_x, dw_y])
                
                force = -p_vol * kirchoff @ dweight
                # delta = C_x_mpm[i].dot(dpos)
                # u_x[face_id] += (p_rho) * p_vol * weight * ((mpm_v[i] + C_mpm[i] @ dpos)[0])
                u_x[face_id] += (p_rho) * p_vol * weight * ((mpm_v[i][0]))
                # u_x[face_id] += (p_rho) * p_vol * weight * ((mpm_v[i])[0])
                # u_x[face_id] += curr_dt * force[0]
                # u_x[face_id] += curr_dt_solid * gravity
                rho_x[face_id] += p_rho * p_vol * weight
                p2g_weight_x[face_id] += p_vol * weight
                p2g_weight_x2[face_id] += (p_rho) *weight
                force_x[face_id] += curr_dt * force[0]

        # vertical impulse
        # base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1)) + ti.Vector.unit(dim, 1)
        base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0))
        for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
            face_id = base_face_id + offset
            if 0 <= face_id[0] < res_x and 0 <= face_id[1] <= res_y:
                weight = N_2(pos[0] - face_id[0] - 0.5) * N_2(pos[1] - face_id[1])
                dpos = ti.Vector([face_id[0] + 0.5 - pos[0], face_id[1] - pos[1]]) * dx
                
                dw_x = 1. / dx * dN_2(pos[0] - face_id[0] - 0.5) * N_2(pos[1] - face_id[1])
                dw_y = 1. / dx * N_2(pos[0] - face_id[0] - 0.5) * dN_2(pos[1] - face_id[1])
                dweight = ti.Vector([dw_x, dw_y])
                
                force = -p_vol * kirchoff @ dweight
                # delta = C_y_mpm[i].dot(dpos)
                u_y[face_id] += (p_rho) * p_vol * weight * ((mpm_v[i][1]))
                # u_y[face_id] += (p_rho) * p_vol * weight * ((mpm_v[i]  + C_mpm[i] @ dpos)[1])
                # u_y[face_id] += (p_rho) * p_vol * weight * ((mpm_v[i])[1])
                # u_y[face_id] += curr_dt * force[1]
                rho_y[face_id] += p_rho * p_vol * weight
                p2g_weight_y2[face_id] += (p_rho) *weight

                p2g_weight_y[face_id] += p_vol * weight
                force_y[face_id] += curr_dt * force[1]

    

    for I in ti.grouped(p2g_weight_x):
        if p2g_weight_x[I] > 0:
            scale = 1. / rho_x[I]
            u_x[I] *= scale
            u_x[I] += curr_dt * gravity
            p2g_weight_x2[I] /= p2g_weight_x[I]
            # rho_x[I] /= p2g_weight_x[I]
            # imp_x[I] = u_x[I]
            
    for I in ti.grouped(rho_x):
        if rho_x[I] > 0:
            force_x[I] /= rho_x[I]
#             # u_x[I] += force_x[I]
    for I in ti.grouped(p2g_weight_x):
        prev_u = u_x[I]
        # if rho_x[I] > 0 and p2g_weight_x2[I] >= (p_rho):
            # u_x[I] += force_x[I]
        if rho_x[I] > 0 and I[0] > 0 and I[0] <= res_x - 1:
            u_x[I] += force_x[I]

    for I in ti.grouped(p2g_weight_y):
        if p2g_weight_y[I] > 0:
            scale = 1. / rho_y[I]
            u_y[I] *= scale
            # u_y[I] += curr_dt * gravity
            p2g_weight_y2[I] /= p2g_weight_y[I]
            # rho_y[I] /= p2g_weight_y[I]
            # imp_y[I] = u_y[I]
            
    for I in ti.grouped(rho_y):
        if rho_y[I] > 0:
            force_y[I] /= rho_y[I]
            # u_y[I] += force_y[I]
            
    for I in ti.grouped(p2g_weight_y):
        prev_u = u_y[I]
        # if rho_y[I] > 0 and p2g_weight_y2[I] >= (p_rho):
            # u_y[I] += force_y[I]
        if rho_y[I] > 0 and I[1] > 0 and I[1] <= res_y - 1:
            u_y[I] += force_y[I]
            
    for I in ti.grouped(p2g_weight_x):   
        if I[0] < 1 and u_x[I] < 0:
            u_x[I] = 0  # Boundary conditions
        if I[0] > res_x - 1 and u_x[I] > 0:
            u_x[I] = 0
    for I in ti.grouped(p2g_weight_y):
        if I[1] < 1 and u_y[I] < 0:
            u_y[I] = 0  # Boundary conditions
        if I[1] > res_y - 1 and u_y[I] > 0:
            u_y[I] = 0
       
            
@ti.kernel
def P2G(particles_imp: ti.template(), particles_imp_mpm_blur: ti.template(), mpm_v: ti.template(), particles_pos: ti.template(), mpm_blur_x: ti.template(), mpm_x: ti.template(), u_x: ti.template(), u_y: ti.template(), rho_x: ti.template(), rho_y: ti.template(), p2g_weight_x: ti.template(), p2g_weight_y: ti.template(), C_x: ti.template(), C_y: ti.template(), C_x_mpm_blur: ti.template(), C_y_mpm_blur: ti.template(), particles_active: ti.template(), F_mpm: ti.template(), C_mpm: ti.template(), mpm_act: ti.template(), levelset: ti.template(), p_rho_w:float, p_rho:float, curr_dt: float, curr_dt_solid:float, force_x: ti.template(), force_y:ti.template(), F_x:ti.template(), F_y:ti.template(), F_x_mpm_blur:ti.template(), F_y_mpm_blur:ti.template(), p2g_weight_x2:ti.template(), p2g_weight_y2:ti.template(), C_mpm_blur:ti.template()):
    u_x.fill(0.0)
    u_y.fill(0.0)
    rho_x.fill(0.0)
    rho_y.fill(0.0)
    levelset.phi.fill(0.0)
    p2g_weight_x.fill(0.0)
    p2g_weight_y.fill(0.0)
    p2g_weight_x2.fill(0.0)
    p2g_weight_y2.fill(0.0)
    force_x.fill(0.0)
    force_y.fill(0.0)
    
    for i in particles_imp:
        if particles_active[i] == 1:
            mu, la = mu_0, lambda_0
            mu = 0.0
            F_inuse = ti.Matrix.cols([F_x[i], F_y[i]])
            U, sig, V = ti.svd(F_inuse)
            J = 1.0
            for d in ti.static(range(2)):
                new_sig = sig[d, d]
                # Jp[p] *= sig[d, d] / new_sig
                sig[d, d] = new_sig
                J *= new_sig
            # F_inuse = ti.Matrix.identity(float, 2) * ti.sqrt(J)
            kirchoff = kirchoff_FCR(F_inuse, U@V.transpose(), J, mu, la) #eq 52 
            
            # horizontal impulse
            pos = particles_pos[i] / dx
            # F_v = ti.Matrix.rows([gradm_x[i], gradm_y[i]])
            # base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0)) + ti.Vector.unit(dim, 0)
            base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1))
            for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
                face_id = base_face_id + offset
                if 0 <= face_id[0] <= res_x and 0 <= face_id[1] < res_y:
                    weight = N_2(pos[0] - face_id[0]) * N_2(pos[1] - face_id[1] - 0.5)
                    dpos = ti.Vector([face_id[0] - pos[0], face_id[1] + 0.5 - pos[1]]) * dx
                    rho_x[face_id] += weight * (p_rho_w) * p_vol
                    p2g_weight_x[face_id] += (p_rho_w) * p_vol * weight
                    p2g_weight_x2[face_id] += weight
                    # rho_x[face_id] += weight
                    delta = C_x[i].dot(dpos)
                    # print(particles_imp[i][0], weight, delta)
                    dw_x = 1. / dx * dN_2(pos[0] - face_id[0]) * N_2(pos[1] - face_id[1] - 0.5)
                    dw_y = 1. / dx * N_2(pos[0] - face_id[0]) * dN_2(pos[1] - face_id[1] - 0.5)
                    dweight = ti.Vector([dw_x, dw_y])
                    
                    ela_force = -kirchoff @ dweight
                    # force = F_v @ dweight
                    if use_APIC:
                        u_x[face_id] += (p_rho_w) * p_vol * (particles_imp[i][0] + delta) * weight
                    else:
                        u_x[face_id] += (particles_imp[i][0]) * weight
                        
                    # vis_force_x[face_id] += curr_dt * viscosity * force[0]
                    force_x[face_id] += curr_dt * ela_force[0] * 0


            # vertical impulse
            # base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1)) + ti.Vector.unit(dim, 1)
            base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0))
            for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
                face_id = base_face_id + offset
                if 0 <= face_id[0] < res_x and 0 <= face_id[1] <= res_y:
                    weight = N_2(pos[0] - face_id[0] - 0.5) * N_2(pos[1] - face_id[1])
                    dpos = ti.Vector([face_id[0] + 0.5 - pos[0], face_id[1] - pos[1]]) * dx
                    rho_y[face_id] += weight * (p_rho_w) * p_vol
                    p2g_weight_y[face_id] += (p_rho_w) * p_vol * weight
                    p2g_weight_y2[face_id] += weight
                    # rho_y[face_id] += weight
                    delta = C_y[i].dot(dpos)
                    dw_x = 1. / dx * dN_2(pos[0] - face_id[0] - 0.5) * N_2(pos[1] - face_id[1])
                    dw_y = 1. / dx * N_2(pos[0] - face_id[0] - 0.5) * dN_2(pos[1] - face_id[1])
                    dweight = ti.Vector([dw_x, dw_y])
                    
                    ela_force = -kirchoff @ dweight
                    # force = F_v @ dweight
                    if use_APIC:
                        u_y[face_id] += (p_rho_w) * p_vol * (particles_imp[i][1] + delta) * weight
                    else:
                        u_y[face_id] += (particles_imp[i][1]) * weight
                    # vis_force_y[face_id] += curr_dt * viscosity * force[1]
                    force_y[face_id] += curr_dt *  ela_force[1] * 0
                    
    for i in mpm_blur_x:
        mu, la = mu_0, lambda_0
        mu = 0.0
        # horizontal impulse
        pos = mpm_blur_x[i] / dx
        # base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0)) + ti.Vector.unit(dim, 0)
        base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1))
        
        F_inuse = ti.Matrix.cols([F_x_mpm_blur[i], F_y_mpm_blur[i]])
        U, sig, V = ti.svd(F_inuse)
        J = 1.0
        for d in ti.static(range(2)):
            new_sig = sig[d, d]
            # Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        F_inuse = ti.Matrix.identity(float, 2) * ti.sqrt(J)
        kirchoff = kirchoff_FCR(F_inuse, U@V.transpose(), J, mu, la) #eq 52 
            
        # F_v = ti.Matrix.rows([gradm_x_mpm_blur[i], gradm_y_mpm_blur[i]])
        for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
            face_id = base_face_id + offset
            if 0 <= face_id[0] <= res_x and 0 <= face_id[1] < res_y:
                weight = N_2(pos[0] - face_id[0]) * N_2(pos[1] - face_id[1] - 0.5)
                dpos = ti.Vector([face_id[0] - pos[0], face_id[1] + 0.5 - pos[1]]) * dx
                rho_x[face_id] += weight * (p_rho_w) * p_vol
                p2g_weight_x[face_id] += (p_rho_w) * p_vol * weight
                p2g_weight_x2[face_id] += weight
                # rho_x[face_id] += weight
                delta = C_x_mpm_blur[i].dot(dpos)
                dw_x = 1. / dx * dN_2(pos[0] - face_id[0]) * N_2(pos[1] - face_id[1] - 0.5)
                dw_y = 1. / dx * N_2(pos[0] - face_id[0]) * dN_2(pos[1] - face_id[1] - 0.5)
                dweight = ti.Vector([dw_x, dw_y])
                # print(particles_imp[i][0], weight, delta)
                # force = F_v @ dweight
                ela_force = -kirchoff @ dweight
                if use_APIC:
                    u_x[face_id] += (p_rho_w) * p_vol * (particles_imp_mpm_blur[i][0] + delta) * weight
                    # u_x[face_id] += (p_rho_w) * p_vol * (particles_imp_mpm_blur[i] + C_mpm_blur[i] @ dpos)[0] * weight
                else:
                    u_x[face_id] += (particles_imp_mpm_blur[i][0]) * weight
                # vis_force_x[face_id] += curr_dt * viscosity * force[0]
                force_x[face_id] += curr_dt *  ela_force[0] * 0

        # vertical impulse
        # base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1)) + ti.Vector.unit(dim, 1)
        base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0))
        for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
            face_id = base_face_id + offset
            if 0 <= face_id[0] < res_x and 0 <= face_id[1] <= res_y:
                weight = N_2(pos[0] - face_id[0] - 0.5) * N_2(pos[1] - face_id[1])
                dpos = ti.Vector([face_id[0] + 0.5 - pos[0], face_id[1] - pos[1]]) * dx
                rho_y[face_id] += weight * (p_rho_w) * p_vol
                p2g_weight_y[face_id] += (p_rho_w) * p_vol * weight
                p2g_weight_y2[face_id] += weight
                # rho_y[face_id] += weight
                delta = C_y_mpm_blur[i].dot(dpos)
                dw_x = 1. / dx * dN_2(pos[0] - face_id[0] - 0.5) * N_2(pos[1] - face_id[1])
                dw_y = 1. / dx * N_2(pos[0] - face_id[0] - 0.5) * dN_2(pos[1] - face_id[1])
                dweight = ti.Vector([dw_x, dw_y])
                ela_force = -kirchoff @ dweight
                # force = F_v @ dweight
                if use_APIC:
                    u_y[face_id] += (p_rho_w) * p_vol * (particles_imp_mpm_blur[i][1] + delta) * weight
                    # u_y[face_id] += (p_rho_w) * p_vol * (particles_imp_mpm_blur[i] + C_mpm_blur[i] @ dpos)[1] * weight
                else:
                    u_y[face_id] += (particles_imp_mpm_blur[i][1]) * weight
                # vis_force_y[face_id] += curr_dt * viscosity * force[1]
                force_y[face_id] += curr_dt *  ela_force[1] * 0
                    
                    
    for i in mpm_x:


        # lamb = ti.math.log(2.2/0.021)
        # F_inuse = F_mpm[i]
        # # H = levelset.heaviside(mpm_x[p])
        # Xy = mpm_act[i]
        # alpha = -lamb * Xy * 8 * ti.sin(0.25 * pi * curr_dt_solid)**8
        # F_a = ti.Matrix([[ti.exp(-alpha), 0.0], [0.0, ti.exp(alpha)]])
        # # F_a = ti.Matrix([[0.0, ti.exp(-alpha)], [ti.exp(alpha), 0.0]])
        # F_inuse = F_inuse @ F_a.inverse()
        
        lamb = ti.math.log(2.2/0.021)
        F_inuse = F_mpm[i]
        # H = levelset.heaviside(mpm_x[p])
        Xy = mpm_act[i]
        # alpha = -lamb * Xy * 8 * ti.sin(0.25 * pi * curr_dt_solid)**8
        # F_a = ti.Matrix([[ti.exp(-alpha), 0.0], [0.0, ti.exp(alpha)]])
        # # F_a = ti.Matrix([[0.0, ti.exp(-alpha)], [ti.exp(alpha), 0.0]])
        # F_inuse = F_inuse @ F_a.inverse()
        
        mu, la = mu_0, lambda_0
        
        T = 2
        time = (curr_dt_solid % T)
        md = (curr_dt_solid // T) % 2
        if Xy != -1000:
            alpha = 1.0
            if time < T / 2:
            # # #     # act = -(act - x_width)
            # # # #     if (total_t[None] // T) % 2 == 0:
            # # # # # alpha = 1.0 - 0.05 * ti.sin(2 * pi * time / T) * ti.exp(-act / (x_width / 3) * (1 - H))
            # # # #         alpha = 1.0 - 0.1 * ti.exp(-act / (x_width / 3))
            # # # #     else:
            # # # #         alpha = 1.0 - 0.025 * ti.exp(-act / (x_width / 3))
            #     # if md == 0:
            #     if Xy == -2000:
            #         # alpha = 1.0 - 0.15 * ti.abs(ti.sin(2 * pi * time / T)) * ti.exp(-(Xy) / (y_width / 3))
            #         mu *= 0.95
            #         la *= 0.95
            #     else:
                alpha = 1.0 - 0.25 * ti.abs(ti.sin(2 * pi * time / T)) * ti.exp(-(Xy) / (y_width / 3))
                # else:
                    # alpha = 1.0 - 0.2 * ti.sin(2 * pi * time / T) * ti.exp((Xy - y_width) / (y_width / 3))
                    # alpha = 1
            else:
                # alpha = 1.0 - 0.2 * (ti.sin(2 * pi * time / T)) * ti.exp(-(mpm_act[p]) / (x_width))
                # if md == 0:
                # if Xy == -2000:
                #     mu *= 0.95
                #     la *= 0.95
                #     # alpha = 1.0 - 0.15 * ti.abs(ti.sin(2 * pi * time / T)) * ti.exp(-(Xy) / (y_width / 3))
                # else:
                alpha = 1.0 - 0.25 * ti.abs(ti.sin(2 * pi * time / T)) * ti.exp((Xy - y_width) / (y_width / 3))
                    # alpha = 1
                # else:
                    # alpha = 1.0 - 0.2 * ti.sin(2 * pi * time / T) * ti.exp(-(Xy) / (y_width / 3))
                    # alpha = 1
            # alpha = 1.0 - 0.15 * (ti.sin(2 * pi * time / T)) * ti.exp((Xy - y_width) / (y_width / 3))
            F_a = ti.Matrix([[1.0 / alpha, 0.0], [0.0, alpha]])
            # F_a = ti.Matrix([[alpha, 0.0], [0.0, 1.0/alpha]])
            
            # F_a = ti.Matrix([[ti.exp(-alpha), 0.0], [0.0, ti.exp(alpha)]])
            F_inuse = F_inuse @ (F_a.inverse())


        # mu, la = mu_0, lambda_0

        U, sig, V = ti.svd(F_inuse)
        J = 1.0

        for d in ti.static(range(2)):
            J *= sig[d, d]

        # Compute Kirchoff Stress
        # kirchoff = kirchoff_FCR(F_inuse, U@V.transpose(), J, mu, la) #eq 52 
        FFt = F_inuse @ F_inuse.transpose()
        
        tr = 0.0
        for d in ti.static(range(2)):
            tr += FFt[d, d]
        
        kirchoff = kirchoff_FCR(F_inuse, U@V.transpose(), J, mu, la) #eq 52 
        # kirchoff = 2 * (FFt - ti.Matrix([[1.0, 0.0], [0.0, 1.0]]) * (tr + 1.0) / 3.0)

        # horizontal impulse
        pos = mpm_x[i] / dx
        # base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0)) + ti.Vector.unit(dim, 0)
        base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1))
        for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
            face_id = base_face_id + offset
            if 0 <= face_id[0] <= res_x and 0 <= face_id[1] < res_y:
                weight = N_2(pos[0] - face_id[0]) * N_2(pos[1] - face_id[1] - 0.5)
                dpos = ti.Vector([face_id[0] - pos[0], face_id[1] + 0.5 - pos[1]]) * dx
                
                dw_x = 1. / dx * dN_2(pos[0] - face_id[0]) * N_2(pos[1] - face_id[1] - 0.5)
                dw_y = 1. / dx * N_2(pos[0] - face_id[0]) * dN_2(pos[1] - face_id[1] - 0.5)
                dweight = ti.Vector([dw_x, dw_y])
                
                force = -p_vol * kirchoff @ dweight
                # delta = C_x_mpm[i].dot(dpos)
                # u_x[face_id] += (p_rho) * p_vol * weight * ((mpm_v[i] + C_mpm[i] @ dpos)[0])
                u_x[face_id] += (p_rho) * p_vol * weight * ((mpm_v[i][0]))
                # u_x[face_id] += (p_rho) * p_vol * weight * ((mpm_v[i])[0])
                # u_x[face_id] += curr_dt * force[0]
                # u_x[face_id] += curr_dt_solid * gravity
                rho_x[face_id] += p_rho * p_vol * weight
                p2g_weight_x[face_id] += (p_rho) * p_vol * weight
                force_x[face_id] += curr_dt * force[0]

        # vertical impulse
        # base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1)) + ti.Vector.unit(dim, 1)
        base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0))
        for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
            face_id = base_face_id + offset
            if 0 <= face_id[0] < res_x and 0 <= face_id[1] <= res_y:
                weight = N_2(pos[0] - face_id[0] - 0.5) * N_2(pos[1] - face_id[1])
                dpos = ti.Vector([face_id[0] + 0.5 - pos[0], face_id[1] - pos[1]]) * dx
                
                dw_x = 1. / dx * dN_2(pos[0] - face_id[0] - 0.5) * N_2(pos[1] - face_id[1])
                dw_y = 1. / dx * N_2(pos[0] - face_id[0] - 0.5) * dN_2(pos[1] - face_id[1])
                dweight = ti.Vector([dw_x, dw_y])
                
                force = -p_vol * kirchoff @ dweight
                # delta = C_y_mpm[i].dot(dpos)
                u_y[face_id] += (p_rho) * p_vol * weight * ((mpm_v[i][1]))
                # u_y[face_id] += (p_rho) * p_vol * weight * ((mpm_v[i] + C_mpm[i] @ dpos)[1])
                # u_y[face_id] += (p_rho) * p_vol * weight * ((mpm_v[i])[1])
                # u_y[face_id] += curr_dt * force[1]
                rho_y[face_id] += p_rho * p_vol * weight
                p2g_weight_y[face_id] += (p_rho) * p_vol * weight
                force_y[face_id] += curr_dt * force[1]

    

    for I in ti.grouped(p2g_weight_x):
        if p2g_weight_x[I] > 0:
            scale = 1. / p2g_weight_x[I]
            u_x[I] *= scale
            u_x[I] += curr_dt * gravity
            # rho_x[I] /= p2g_weight_x[I]
            # imp_x[I] = u_x[I]
            
    for I in ti.grouped(rho_x):
        if rho_x[I] > 0:
            force_x[I] /= rho_x[I]
#             # u_x[I] += force_x[I]
    for I in ti.grouped(p2g_weight_x):
        prev_u = u_x[I]
        # if rho_x[I] > 0 and p2g_weight_x2[I] == 0:
            # u_x[I] += force_x[I]
        if rho_x[I] > 0 and I[0] > 0 and I[0] <= res_x - 1:
            u_x[I] += force_x[I]

    for I in ti.grouped(p2g_weight_y):
        if p2g_weight_y[I] > 0:
            scale = 1. / p2g_weight_y[I]
            u_y[I] *= scale
            # u_y[I] += curr_dt * gravity
            # rho_y[I] /= p2g_weight_y[I]
            # imp_y[I] = u_y[I]
            
    for I in ti.grouped(rho_y):
        if rho_y[I] > 0:
            force_y[I] /= rho_y[I]
            # u_y[I] += force_y[I]
            
    for I in ti.grouped(p2g_weight_y):
        prev_u = u_y[I]
        # if rho_y[I] > 0 and p2g_weight_y2[I] == 0:
            # u_y[I] += force_y[I]
        if rho_y[I] > 0 and I[1] > 0 and I[1] <= res_y - 1:
            u_y[I] += force_y[I]
            
    for I in ti.grouped(p2g_weight_x):   
        if I[0] < 1 and u_x[I] < 0:
            u_x[I] = 0  # Boundary conditions
        if I[0] > res_x - 1 and u_x[I] > 0:
            u_x[I] = 0
    for I in ti.grouped(p2g_weight_y):
        if I[1] < 1 and u_y[I] < 0:
            u_y[I] = 0  # Boundary conditions
        if I[1] > res_y - 1 and u_y[I] > 0:
            u_y[I] = 0
        
        
        
@ti.kernel
def g2p_divergence_vis(particles_pos:ti.template(), particle_init_impulse:ti.template(), gradm_x_grid:ti.template(), gradm_y_grid:ti.template(), F_x: ti.template(), F_y:ti.template(), particles_active: ti.template(), dt:float, vis : float, dx : float):
    for I in particles_pos:
        if particles_active[I] >= 1:
            vis_force = interp_MAC_divergence(gradm_x_grid, gradm_y_grid, particles_pos[I], dx)
            F = ti.Matrix.rows([F_x[I], F_y[I]])
            particle_init_impulse[I] += (F@(vis_force * vis)) * dt
            
@ti.kernel
def add_gravity(particle_init_impulse:ti.template(), F_x: ti.template(), F_y:ti.template(), particles_active: ti.template(), dt:float, g : float, p_rho_w : float):
    for I in particle_init_impulse:
        if particles_active[I] >= 1:
            F = ti.Matrix.rows([F_x[I], F_y[I]])
            particle_init_impulse[I] -= (F@(ti.Vector([1.0, 0.0]) * g)) * dt
            # particle_init_impulse[I] -= (F@(ti.Vector([0.0, 1.0]) * g)) * dt
            
@ti.kernel
def add_fluid_force(particles_pos:ti.template(), particle_init_impulse:ti.template(), F_x: ti.template(), F_y:ti.template(), particles_active: ti.template(), force_x:ti.template(), force_y:ti.template(), dt:float):
    for I in particle_init_impulse:
        if particles_active[I] >= 1:
            vis_force, _, _, _ = interp_u_MAC_grad(force_x, force_y, particles_pos[I], dx)
            F = ti.Matrix.rows([F_x[I], F_y[I]])
            particle_init_impulse[I] += (F@(vis_force)) * dt
            
            
@ti.kernel
def add_gravity_solid(mpm_v:ti.template(), curr_dt:ti.template(), g:ti.template()):
    for I in mpm_v:
        mpm_v[I] += ti.Vector([1.0, 0.0]) * g * curr_dt
