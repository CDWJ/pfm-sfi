import taichi as ti
import taichi.math as ts
from math import pi
from taichi_utils import *
@ti.data_oriented
class LevelSet:
    def __init__(self, res, dx, total_mk, epsilon, dim=2):
        self.grid_n = int(res[0] / 4)
        self.grid_size = 1.0 / self.grid_n
        self.list_head = ti.field(dtype=ti.i32, shape=self.grid_n * self.grid_n * 2)
        self.list_cur = ti.field(dtype=ti.i32, shape=self.grid_n * self.grid_n * 2)
        self.list_tail = ti.field(dtype=ti.i32, shape=self.grid_n * self.grid_n * 2)

        self.grain_count = ti.field(dtype=ti.i32,
                               shape=(self.grid_n * 2, self.grid_n),
                               name="grain_count")
        self.column_sum = ti.field(dtype=ti.i32, shape=self.grid_n, name="column_sum")
        self.prefix_sum = ti.field(dtype=ti.i32, shape=(self.grid_n * 2, self.grid_n), name="prefix_sum")
        
        self.valid = ti.field(int, shape=(res[0], res[1]))
        self.particle_valid = ti.field(int, shape=(total_mk))
        self.phi = ti.field(float, shape=(res[0], res[1]))
        self.phi_temp = ti.field(float, shape=(res[0], res[1]))
        self.dx = dx
        self.total_mk = ti.field(int, shape=())
        self.count = ti.field(int, shape=())
        self.total_mk[None] = total_mk
        self.count[None] = 0
        self.particle_id = ti.field(ti.i32, shape=(total_mk,))
        self.dim = dim
        self.repeat_times = 4
        self.res = res
        self.epsilon = epsilon
        
    @ti.func
    def sample(self, data, pos):
        tot = data.shape
        # static unfold for efficiency
        if ti.static(len(data.shape) == 2):
            i, j = ts.clamp(int(pos[0]), 0, tot[0] - 1), ts.clamp(int(pos[1]), 0, tot[1] - 1)
            ip, jp = ts.clamp(i + 1, 0, tot[0] - 1), ts.clamp(j + 1, 0, tot[1] - 1)
            s, t = ts.clamp(pos[0] - i, 0.0, 1.0), ts.clamp(pos[1] - j, 0.0, 1.0)
            return \
                (data[i, j] * (1 - s) + data[ip, j] * s) * (1 - t) + \
                (data[i, jp] * (1 - s) + data[ip, jp] * s) * t

        else:
            i, j, k = ts.clamp(int(pos[0]), 0, tot[0] - 1), ts.clamp(int(pos[1]), 0, tot[1] - 1), ts.clamp(int(pos[2]), 0, tot[2] - 1)
            ip, jp, kp = ts.clamp(i + 1, 0, tot[0] - 1), ts.clamp(j + 1, 0, tot[1] - 1), ts.clamp(k + 1, 0, tot[2] - 1)
            s, t, u = ts.clamp(pos[0] - i, 0.0, 1.0), ts.clamp(pos[1] - j, 0.0, 1.0), ts.clamp(pos[2] - k, 0.0, 1.0)
            return \
                ((data[i, j, k] * (1 - s) + data[ip, j, k] * s) * (1 - t) + \
                (data[i, jp, k] * (1 - s) + data[ip, jp, k] * s) * t) * (1 - u) + \
                ((data[i, j, kp] * (1 - s) + data[ip, j, kp] * s) * (1 - t) + \
                (data[i, jp, kp] * (1 - s) + data[ip, jp, kp] * s) * t) * u    
        
    @ti.func
    def heaviside2(self, x : ti.template()):
        sdf = interp_center(self.phi, x, self.dx, BL_x=0.5, BL_y=0.5)
        # sdf = self.phi[ti.floor(x * self.res[0], int)]
        result = 1.0
        epsilon_inuse = self.epsilon
        if sdf <= -epsilon_inuse:
            result = 0.0
        elif ti.abs(sdf) < epsilon_inuse:
            result = 0.5 * (1 + sdf / epsilon_inuse + 1.0 / pi * ti.sin(pi * sdf / epsilon_inuse))
        # return result
        
        # sdf = self.sample(self.phi, x * self.res[0] - 0.5)
        # result = 1.0
        # if sdf <= 0:
        #     result = 0.0
        # elif ti.abs(sdf) < epsilon_inuse:
        #     result = 0.5 * (1 + (sdf - epsilon_inuse / 2) / epsilon_inuse + 1.0 / pi * ti.sin(pi * (sdf - epsilon_inuse / 2) / epsilon_inuse))
        return result

    @ti.func
    def heaviside(self, x : ti.template()):
        sdf = interp_center(self.phi, x, self.dx, BL_x=0.5, BL_y=0.5)
        return sdf
    
    
    # @ti.func
    # def heaviside2(self, x : ti.template(), sdf: ti.template()):
    #     result = 1.0
    #     epsilon_inuse = self.epsilon
    #     if sdf <= 0:
    #         result = 0.0
    #     elif sdf < epsilon_inuse:
    #         result = 0.5 * (1 + (sdf - epsilon_inuse / 2) / epsilon_inuse + 1.0 / pi * ti.sin(pi * (sdf - epsilon_inuse / 2) / epsilon_inuse))
        # return result
    
    @ti.func
    def sample_levelset(self, x : ti.template()):
        sdf = self.sample(self.phi, x * self.res[0] - 0.5)
        result = 0.0
        if sdf <= 0:
            result = 1.0
        return result


    @ti.func
    def cal_normal(self, temp_p):
        n = ti.Vector.zero(float, self.dim)
        for k in ti.static(range(self.dim)):
            I1 = temp_p * self.res[0] - ti.Vector.unit(self.dim, k)
            phi_m = self.sample(self.phi, I1 - 0.5)
            I1 = temp_p * self.res[0] + ti.Vector.unit(self.dim, k)
            phi_p = self.sample(self.phi, I1 - 0.5)
            n[k] = (phi_p - phi_m)
        return ti.math.normalize(n)
    
    
    @ti.kernel
    def sample_narrowband(self, p_x_blur : ti.template()):
        phi_dim_x = (p_x_blur.shape)[0]
        self.count[None] = 0
        for I in ti.grouped(self.phi):
            self.valid[I] = -1
            if self.phi[I] <= self.epsilon and self.phi[I] >= -self.epsilon:
                self.count[None] += 1
                self.valid[I] = 1
        per_cell = ti.cast(phi_dim_x // self.count[None], ti.int32)
        curr_count = ti.cast(per_cell * self.count[None], ti.int32)
        offset = phi_dim_x - curr_count
        self.count[None] = 0
        for I in ti.grouped(self.phi):
            if self.valid[I] == 1:
                for j in range(per_cell):
                    num = ti.atomic_add(self.count[None], 1)
                    p_x_blur[num] = ((I + 0.5) + ti.Vector([ti.random() - 0.5, ti.random() - 0.5])) * self.dx
        
        for I in range(offset):
            p_x_blur[curr_count + I] = p_x_blur[curr_count - 1]
            
        # for I in range(phi_dim_x):
        #     I1 = p_x_blur[I] * self.res[0]
        #     phi = self.sample(self.phi, I1 - 0.5)
        #     if phi < 0:
        #         p_x_blur[I] = p_x_blur[I] - self.cal_normal(p_x_blur[I]) * phi


    

#     @ti.kernel
#     def sample_narrowband(self, p_x : ti.template(), p_x_blur : ti.template()):
#         phi_dim_x = (p_x_blur.shape)[0]
#         self.count[None] = 0
#         for I in ti.grouped(p_x):
#             self.particle_valid[I] = -1
#             coord = ti.floor((p_x[I]) / self.dx, int)
#             if self.phi[coord] <= self.epsilon and self.phi[coord] >= -self.epsilon:
#                 self.count[None] += 1
#                 self.particle_valid[I] = 1
        
#         per_cell = ti.cast(phi_dim_x // self.count[None], ti.int32)
#         curr_count = ti.cast(per_cell * self.count[None], ti.int32)
#         offset = phi_dim_x - curr_count
#         self.count[None] = 0
#         for I in ti.grouped(p_x):
#             if self.particle_valid[I] == 1:
#                 n_x = self.cal_normal(p_x[I])
#                 for j in range(per_cell):
#                     num = ti.atomic_add(self.count[None], 1)
#                     p_x_blur[num] = p_x[I] + n_x * ti.Vector([(ti.random() + 0.5) * self.epsilon, (ti.random() + 0.5) * self.epsilon])
#                     # p_x_blur[num] = ((I + 0.5) + ti.Vector([ti.random() - 0.5, ti.random() - 0.5])) * self.dx
        
#         for I in range(offset):
#             p_x_blur[curr_count + I] = p_x_blur[curr_count - 1]
            
            
#     @ti.kernel
#     def find_nearest(self, p_x : ti.template(), p_x_blur : ti.template(), nearest_p : ti.template(), nearest_d : ti.template()):
#         self.total_mk[None] = p_x.shape[0]

#         for I in ti.grouped(p_x_blur):
#             blur_coord = p_x_blur[I]
#             for j in range(self.total_mk[None]):
#                 curr_length = ti.math.length(p_x[j] - blur_coord)
#                 if nearest_d[I] == -1:
#                     nearest_p[I] = j
#                     nearest_d[I] = curr_length
#                 else:
#                     if (nearest_d[I]) > (curr_length):
#                         nearest_d[I] = curr_length
#                         nearest_p[I] = j

    @ti.kernel
    def find_nearest(self, p_x : ti.template(), p_x_blur : ti.template(), nearest_p : ti.template(), nearest_d : ti.template()):
        self.total_mk[None] = p_x.shape[0]

        for I in ti.grouped(p_x_blur):
            blur_coord = p_x_blur[I]
            nearest = 1e20
            nearest_pp = -1
            for j in range(self.total_mk[None]):
                curr_length = ti.math.length(p_x[j] - blur_coord)
                # if nearest_d[I] == -1:
                #     nearest_p[I] = j
                #     nearest_d[I] = curr_length
                # else:
                #     if (nearest_d[I]) > (curr_length):
                #         nearest_d[I] = curr_length
                #         nearest_p[I] = j
                if nearest > curr_length:
                    nearest = curr_length
                    nearest_pp = j
            nearest_p[I] = nearest_pp
            nearest_d[I] = nearest
            
#             I1 = p_x_blur[I] * self.res[0]
#             phi = self.sample(self.phi, I1 - 0.5)
#             p_x_blur[I] = p_x[nearest_pp] + self.cal_normal(p_x_blur[I]) * (ti.random()) * self.epsilon * 1.25
            
#         for I in ti.grouped(p_x_blur):
#             blur_coord = p_x_blur[I]
#             nearest = 1e20
#             nearest_pp = -1
#             for j in range(self.total_mk[None]):
#                 curr_length = ti.math.length(p_x[j] - blur_coord)
#                 # if nearest_d[I] == -1:
#                 #     nearest_p[I] = j
#                 #     nearest_d[I] = curr_length
#                 # else:
#                 #     if (nearest_d[I]) > (curr_length):
#                 #         nearest_d[I] = curr_length
#                 #         nearest_p[I] = j
#                 if nearest > curr_length:
#                     nearest = curr_length
#                     nearest_pp = j
#             nearest_p[I] = nearest_pp
#             nearest_d[I] = nearest
            

    @ti.kernel
    def contact(self, p_x : ti.template()):
        self.total_mk[None] = p_x.shape[0]

        phi_dim_x, phi_dim_y = self.phi.shape
        for I in ti.grouped(self.phi):
            self.valid[I] = -1
            grid_coord = (I + 0.5) * self.dx
            for j in range(self.total_mk[None]):
                this_phi = ti.math.length(p_x[j] - grid_coord) - (0.0 * self.dx)
                # this_phi = ti.math.length(p_x[j] - grid_coord)
                if self.valid[I] == -1:
                    self.phi[I] = this_phi
                    self.valid[I] = 0
                else:
                    if (self.phi[I]) > (this_phi):
                        self.phi[I] = this_phi

                                
            if self.phi[I] < 0:
                self.valid[I] = -1
                self.phi[I] = -1e20
                
        for I in ti.grouped(p_x):
            grid_idx = ti.floor((p_x[I]) / self.dx - 0.5, int)
            self.valid[grid_idx] = -1
            self.phi[grid_idx] = -1e20
     
    @ti.func
    def update_from_neighbor(self, I):
        # solve the Eikonal equation
        nb = ti.Vector.zero(float, self.dim)
        for k in ti.static(range(self.dim)):
            o = ti.Vector.unit(self.dim, k)
            if I[k] == 0 or (I[k] < self.res[k] - 1 and ti.abs(self.phi_temp[I + o]) < ti.abs(self.phi_temp[I - o])): nb[k] = ti.abs(self.phi_temp[I + o])
            else: nb[k] = ti.abs(self.phi_temp[I - o])

        # sort
        for i in ti.static(range(self.dim-1)):
            for j in ti.static(range(self.dim-1-i)):
                if nb[j] > nb[j + 1]: nb[j], nb[j + 1] = nb[j + 1], nb[j]
        # (Try just the closest neighbor)
        d = nb[0] + self.dx
        if d > nb[1]:
            # (Try the two closest neighbors)
            d = (1/2) * (nb[0] + nb[1] + ti.sqrt(2 * (self.dx ** 2) - (nb[1] - nb[0]) ** 2))
            if ti.static(self.dim == 3):
                if d > nb[2]:
                    # (Use all three neighbors)
                    d = (1/3) * (nb[0] + nb[1] + nb[2] + ti.sqrt(ti.max(0, (nb[0] + nb[1] + nb[2]) ** 2 - 3 * (nb[0] ** 2 + nb[1] ** 2 + nb[2] ** 2 - self.dx ** 2))))

        return d
    @ti.func
    def propagate_update(self, I, s):
        if self.valid[I] == -1:
            d = self.update_from_neighbor(I)
            if ti.abs(d) < ti.abs(self.phi_temp[I]): self.phi_temp[I] = d * ti.math.sign(self.phi[I])
        return s

    @ti.kernel
    def propagate(self):
        if ti.static(self.dim == 2):
            for t in ti.static(range(self.repeat_times)):
                for i in range(self.res[0]):
                    j = 0
                    while j < self.res[1]: j += self.propagate_update([i, j], 1)

                for i in range(self.res[0]):
                    j = self.res[1] - 1
                    while j >= 0: j += self.propagate_update([i, j], -1)
            
                for j in range(self.res[1]):
                    i = 0
                    while i < self.res[1]: i += self.propagate_update([i, j], 1)

                for j in range(self.res[1]):
                    i = self.res[1] - 1
                    while i >= 0: i += self.propagate_update([i, j], -1)

        if ti.static(self.dim == 3):
            for t in ti.static(range(self.repeat_times)):
                for i, j in ti.ndrange(self.res[0], self.res[1]):
                    k = 0
                    while k < self.res[2]: k += self.propagate_update([i, j, k], 1)

                for i, j in ti.ndrange(self.res[0], self.res[1]):
                    k = self.res[2] - 1
                    while k >= 0: k += self.propagate_update([i, j, k], -1)

                for i, k in ti.ndrange(self.res[0], self.res[2]):
                    j = 0
                    while j < self.res[1]: j += self.propagate_update([i, j, k], 1)

                for i, k in ti.ndrange(self.res[0], self.res[2]):
                    j = self.res[1] - 1
                    while j >= 0: j += self.propagate_update([i, j, k], -1)

                for j, k in ti.ndrange(self.res[1], self.res[2]):
                    i = 0
                    while i < self.res[1]: i += self.propagate_update([i, j, k], 1)

                for j, k in ti.ndrange(self.res[1], self.res[2]):
                    i = self.res[0] - 1
                    while i >= 0: i += self.propagate_update([i, j, k], -1)
                            
                        
    # def redistance(self, mpm_x):
    #     self.phi.fill(1e20)
    #     self.contact(mpm_x)
    #     self.phi_temp.copy_from(self.phi)
    #     self.propagate()
    #     self.phi.copy_from(self.phi_temp)
        
        
        
    @ti.func
    def markers_propagate_update(self, markers, lI, o, s):
        I, offset = ti.Vector(lI), ti.Vector(o)
        if all(I + offset >= 0) and all(I + offset < self.res):
            d = (markers[self.valid[I + offset]] - (I + 0.5) * self.dx).norm()
            if d < self.phi[I]:
                self.phi[I] = d
                self.valid[I] = self.valid[I + o]
        return s
    
    @ti.kernel
    def markers_propagate(self, markers : ti.template(), total_mk : ti.template()):
        if ti.static(self.dim == 2):
            for t in ti.static(range(self.repeat_times)):
                for i in range(self.res[0]):
                    j = 0
                    while j < self.res[1]: j += self.markers_propagate_update(markers, [i, j], [0, 1], 1)

                for i in range(self.res[0]):
                    j = self.res[1] - 1
                    while j >= 0: j += self.markers_propagate_update(markers, [i, j], [0, -1], -1)
            
                for j in range(self.res[1]):
                    i = 0
                    while i < self.res[1]: i += self.markers_propagate_update(markers, [i, j], [1, 0], 1)

                for j in range(self.res[1]):
                    i = self.res[1] - 1
                    while i >= 0: i += self.markers_propagate_update(markers, [i, j], [-1, 0], -1)

        if ti.static(self.dim == 3):
            for t in ti.static(range(self.repeat_times)):
                for i, j in ti.ndrange(self.res[0], self.res[1]):
                    k = 0
                    while k < self.res[2]: k += self.markers_propagate_update(markers, [i, j, k], [0, 0, 1], 1)

                for i, j in ti.ndrange(self.res[0], self.res[1]):
                    k = self.res[2] - 1
                    while k >= 0: k += self.markers_propagate_update(markers, [i, j, k], [0, 0, -1], -1)

                for i, k in ti.ndrange(self.res[0], self.res[2]):
                    j = 0
                    while j < self.res[1]: j += self.markers_propagate_update(markers, [i, j, k], [0, 1, 0], 1)

                for i, k in ti.ndrange(self.res[0], self.res[2]):
                    j = self.res[1] - 1
                    while j >= 0: j += self.markers_propagate_update(markers, [i, j, k], [0, -1, 0], -1)

                for j, k in ti.ndrange(self.res[1], self.res[2]):
                    i = 0
                    while i < self.res[1]: i += self.markers_propagate_update(markers, [i, j, k], [1, 0, 0], 1)

                for j, k in ti.ndrange(self.res[1], self.res[2]):
                    i = self.res[0] - 1
                    while i >= 0: i += self.markers_propagate_update(markers, [i, j, k], [-1, 0, 0], -1)
                    
                    
    @ti.kernel
    def distance_to_markers(self, markers : ti.template(), total_mk : ti.template()):
        # (Initialize the arrays near the input geometry)
        for p in range(total_mk[None]):
            I = (markers[p] / self.dx).cast(int)
            d = (markers[p] - (I + 0.5) * self.dx).norm()
            if all(I >= 0) and all(I < self.res) and d < self.phi[I]:
                self.phi[I] = d
                self.valid[I] = p
                
    @ti.kernel
    def target_minus(self):
        for I in ti.grouped(self.phi):
            self.phi[I] -= (0.99 * self.dx) # the particle radius r (typically just a little less than the grid cell size dx)

        for I in ti.grouped(self.phi):
            sign_change = False
            for k in ti.static(range(self.dim)):
                for s in ti.static((-1, 1)):
                    offset = ti.Vector.unit(self.dim, k) * s
                    I1 = I + offset
                    if I1[k] >= 0 and I1[k] < self.res[k] and \
                    ts.sign(self.phi[I]) != ts.sign(self.phi[I1]):
                        sign_change = True

            if sign_change and self.phi[I] <= 0:
                self.valid[I] = 0
                self.phi_temp[I] = self.phi[I]
            elif self.phi[I] <= 0:
                self.phi_temp[I] = ti.cast(-1, float)
            else:
                self.phi_temp[I] = self.phi[I]
                self.valid[I] = 0
                
                
    @ti.kernel
    def smoothing(self, phi : ti.template(), phi_temp : ti.template()):
        for I in ti.grouped(phi_temp):
            phi_avg = ti.cast(0, float)
            tot = ti.cast(0, int)
            for k in ti.static(range(self.dim)):
                for s in ti.static((-1, 1)):
                    offset = ti.Vector.unit(self.dim, k) * s
                    I1 = I + offset
                    if I1[k] >= 0 and I1[k] < self.res[k]:
                        phi_avg += phi_temp[I1]
                        tot += 1

            phi_avg /= tot
            phi[I] = phi_avg if phi_avg < phi_temp[I] else phi_temp[I]
            
    @ti.kernel
    def make_sym(self, phi : ti.template()):
        for i, j in ti.ndrange(self.res[0]//2, self.res[1]):
            self.phi[self.res[0] - i - 1, j] = self.phi[i, j]
            
            
    def build_from_markers(self, markers):
        self.phi_temp.copy_from(self.phi)
        self.phi.fill(1e20)
        self.valid.fill(-1)
        self.distance_to_markers(markers, self.total_mk)
        self.markers_propagate(markers, self.total_mk)
        self.valid.fill(-1)
        self.target_minus()
        self.propagate()
        self.smoothing(self.phi, self.phi_temp)
        self.smoothing(self.phi_temp, self.phi)
        self.smoothing(self.phi, self.phi_temp)
        
        
        
    def redistance(self, mpm_x):
        self.phi.fill(1e20)
        self.contact(mpm_x)
        self.phi_temp.copy_from(self.phi)
        self.propagate()
        self.phi.copy_from(self.phi_temp)
        
        self.smoothing(self.phi, self.phi_temp)
        self.smoothing(self.phi_temp, self.phi)
        self.smoothing(self.phi, self.phi_temp)
        
        # self.make_sym(self.phi)