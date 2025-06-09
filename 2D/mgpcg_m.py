import math
import time
import matplotlib.pyplot as plt
from taichi_utils import *
import numpy as np

@ti.data_oriented
class MGPCG:
    '''
Grid-based MGPCG solver for the possion equation.

.. note::

    This solver only runs on CPU and CUDA backends since it requires the
    ``pointer`` SNode.
    '''
    def __init__(self, boundary_types, boundary_mask, N, levelset, p_rho, p_rho_w, dim=2, base_level=4, real=float):
        '''
        :parameter dim: Dimensionality of the fields.
        :parameter N: Grid resolutions.
        :parameter n_mg_levels: Number of multigrid levels.
        '''

        # grid parameters
        self.use_multigrid = True
        base_level = 4
        self.N = N
        res = N
        self.n_mg_levels = int(math.log2(min(N))) - base_level + 1
        self.pre_and_post_smoothing = 2
        # self.bottom_smoothing = 50
        self.bottom_smoothing = 20
        self.dim = dim
        self.real = real
        self.levelset = levelset
        self.p_rho = p_rho
        self.p_rho_w = p_rho_w
        self.dt = 0
        self.edge_length = 1.0

        self.dx = self.edge_length / N[1]
        
        # self.dx = [1.0 / (res[0] // 2**l) for l in range(self.n_mg_levels)]
        # setup sparse simulation data arrays
        # setup sparse simulation data arrays
        self.r = [ti.field(dtype=self.real)
                  for _ in range(self.n_mg_levels)]  # residual
        self.bm = [boundary_mask] + [ti.field(dtype=ti.i32)
                  for _ in range(self.n_mg_levels - 1)]  # boundary_mask
        self.z = [ti.field(dtype=self.real)
                  for _ in range(self.n_mg_levels)]  # M^-1 self.r
        self.x = ti.field(dtype=self.real)  # solution
        self.p = ti.field(dtype=self.real)  # conjugate gradient
        self.Ap = ti.field(dtype=self.real)  # matrix-vector product
        self.alpha = ti.field(dtype=self.real)  # step size
        self.beta = ti.field(dtype=self.real)  # step size
        self.sum = ti.field(dtype=self.real)  # storage for reductions
        self.r_mean = ti.field(dtype=self.real)  # storage for avg of r
        self.num_entries = math.prod(self.N)

        indices = ti.ijk if self.dim == 3 else ti.ij
        self.grid = ti.root.pointer(indices, [n // 4 for n in self.N]).dense(
            indices, 4).place(self.x, self.p, self.Ap)

        for l in range(self.n_mg_levels):
            self.grid = ti.root.pointer(indices,
                                        [n // (4 * 2**l) for n in self.N]).dense(
                                            indices,
                                            4).place(self.r[l], self.z[l])
        for l in range(1, self.n_mg_levels):
            ti.root.dense(indices,
                            [n // (4 * 2**l) for n in self.N]).dense(
                                indices,
                                4).place(self.bm[l])

        ti.root.place(self.alpha, self.beta, self.sum, self.r_mean)

        self.boundary_types = boundary_types



    @ti.func
    def init_r(self, I, r_I):
        self.r[0][I] = r_I
        self.z[0][I] = 0
        self.Ap[I] = 0
        self.p[I] = 0
        self.x[I] = 0

    @ti.kernel
    def init(self, r: ti.template(), k: ti.template()):
        '''
        Set up the solver for $\nabla^2 x = k r$, a scaled Poisson problem.
        :parameter k: (scalar) A scaling factor of the right-hand side.
        :parameter r: (ti.field) Unscaled right-hand side.
        '''
        for I in ti.grouped(ti.ndrange(*self.N)):
            self.init_r(I, r[I] * k)


    @ti.kernel
    def get_result(self, x: ti.template()):
        '''
        Get the solution field.

        :parameter x: (ti.field) The field to store the solution
        '''
        for I in ti.grouped(ti.ndrange(*self.N)):
            x[I] = self.x[I]

    @ti.kernel
    def downsample_bm(self, bm_fine: ti.template(), bm_coarse: ti.template()):
        # if self.dim == 2:
        #     range_d = (2, 2)
        # else:
        #     range_d = (2, 2, 2)

        for I in ti.grouped(bm_coarse):
            I2 = I * 2
            all_solid = 1
            range_d = 2 * ti.Vector.one(ti.i32, self.dim)
            for J in ti.grouped(ti.ndrange(*range_d)):
                if bm_fine[I2 + J] <= 0:
                    all_solid = 0
            
            bm_coarse[I] = all_solid

    @ti.func
    def neighbor_sum2(self, x, I, bm):
        dims = x.shape
        # print(dims)
        dx = self.edge_length / dims[1]
        ret = ti.cast(0.0, self.real)
        for i in ti.static(range(self.dim)):
            offset = ti.Vector.unit(self.dim, i)
            # add right if has right (and also not boundary)
            if (I[i] < dims[i] - 1) and bm[I+offset] <= 0:
                rho = self.levelset.heaviside(((I + offset / 2.0 + 0.5) * (dx * 1.0)))
                ret += x[I + offset] * self.dt / rho
            # add left if has left (and also not boundary)
            if (I[i] > 0) and bm[I-offset] <= 0:
                rho = self.levelset.heaviside(((I - offset / 2.0 + 0.5) * (dx * 1.0)))
                # H = self.levelset.heaviside(((I - offset / 2.0) * (dx * 1.0)))
                ret += x[I - offset] * self.dt / rho
        # H = self.levelset.heaviside(((I + 0.5) * (dx * 1.0)))
        return ret
    
    @ti.func
    def num_fluid_neighbors2(self, x, I, bm): # l is the level
        dims = x.shape
        num = 0.0
        # H = ti.cast(0.0, self.real)
        dx = self.edge_length / dims[1]
        for i in ti.static(range(self.dim)):
            offset = ti.Vector.unit(self.dim, i)
            # check low
            if not(I[i] <= 0 and self.boundary_types[i,0] == 2): # if on lower boundary
                if bm[I-offset] <= 0:
                    rho = self.levelset.heaviside(((I - offset/2.0 + 0.5) * (dx * 1.0)))
                    # H = self.levelset.heaviside(((I - offset/2.0) * (dx * 1.0)))
                    num += self.dt / rho
            # check high
            if not(I[i] >= dims[i] - 1 and self.boundary_types[i,1] == 2): # if on upper boundary
                if bm[I+offset] <= 0:
                    rho = self.levelset.heaviside(((I + offset/2.0 + 0.5) * (dx * 1.0)))
                    # H = self.levelset.heaviside(((I + offset/2.0) * (dx * 1.0)))
                    num += self.dt / rho
        return num

                
    @ti.kernel
    def compute_Ap(self):
        for I in ti.grouped(self.Ap):
            if self.bm[0][I] <= 0: # only if a cell is not a solid
                multiplier = self.num_fluid_neighbors2(self.p, I, self.bm[0])
                self.Ap[I] = multiplier * self.p[I] - self.neighbor_sum2(
                    self.p, I, self.bm[0])

    @ti.kernel
    def get_Ap(self, p: ti.template(), Ap: ti.template()):
        for I in ti.grouped(Ap):
            if self.bm[0][I] <= 0: # only if a cell is not a solid
                multiplier = self.num_fluid_neighbors2(p, I, self.bm[0])
                Ap[I] = multiplier * p[I] - self.neighbor_sum2(
                    p, I, self.bm[0])

    @ti.kernel
    def reduce(self, p: ti.template(), q: ti.template()):
        self.sum[None] = 0
        for I in ti.grouped(p):
            if self.bm[0][I] <= 0: # only if a cell is not a solid
                self.sum[None] += p[I] * q[I]

    @ti.kernel
    def update_x(self):
        for I in ti.grouped(self.p):
            if self.bm[0][I] <= 0: # only if a cell is not a solid
                self.x[I] += self.alpha[None] * self.p[I]

    @ti.kernel
    def update_r(self):
        for I in ti.grouped(self.p):
            if self.bm[0][I] <= 0: # only if a cell is not a solid
                self.r[0][I] -= self.alpha[None] * self.Ap[I]

    @ti.kernel
    def update_p(self):
        for I in ti.grouped(self.p):
            if self.bm[0][I] <= 0: # only if a cell is not a solid
                self.p[I] = self.z[0][I] + self.beta[None] * self.p[I]

    @ti.kernel
    def restrict(self, l: ti.template()):
        for I in ti.grouped(self.r[l]):
            if self.bm[l][I] <= 0: # only if a cell is not a solid, on level = l
                multiplier = self.num_fluid_neighbors2(self.z[l], I, self.bm[l])
                res = self.r[l][I] - (multiplier * self.z[l][I] -
                                    self.neighbor_sum2(self.z[l], I, self.bm[l]))
                self.r[l + 1][I // 2] += res * 1.0 / (self.dim-1.0)

    @ti.kernel
    def prolongate(self, l: ti.template()):
        for I in ti.grouped(self.z[l]):
            self.z[l][I] += self.z[l + 1][I // 2]

    @ti.kernel
    def smooth(self, l: ti.template(), phase: ti.template()):
        # phase = red/black Gauss-Seidel phase
        for I in ti.grouped(self.r[l]):
            if (self.bm[l][I] <= 0) and ((I.sum()) & 1 == phase):
                 if ((I.sum()) & 1 == phase): # only if a cell is not a solid, on level = l
                    multiplier = self.num_fluid_neighbors2(self.z[l], I, self.bm[l])
                    self.z[l][I] = (self.r[l][I] + self.neighbor_sum2(
                        self.z[l], I, self.bm[l])) / multiplier

    @ti.kernel
    def recenter(self, r: ti.template()): # so that the mean value of r is 0
        self.r_mean[None] = 0.0
        for I in ti.grouped(r):
            self.r_mean[None] += r[I] / self.num_entries    
        for I in ti.grouped(r):
            r[I] -= self.r_mean[None]

    def apply_preconditioner(self):
        self.z[0].fill(0)
        for l in range(self.n_mg_levels - 1):
            for i in range(self.pre_and_post_smoothing):
                self.smooth(l, 0)
                self.smooth(l, 1)
            self.z[l + 1].fill(0)
            self.r[l + 1].fill(0)
            self.restrict(l)

        for i in range(self.bottom_smoothing // 2):
            self.smooth(self.n_mg_levels - 1, 0)
            self.smooth(self.n_mg_levels - 1, 1)
        for i in range(self.bottom_smoothing // 2):
            self.smooth(self.n_mg_levels - 1, 1)
            self.smooth(self.n_mg_levels - 1, 0)

        for l in reversed(range(self.n_mg_levels - 1)):
            self.prolongate(l)
            for i in range(self.pre_and_post_smoothing):
                self.smooth(l, 1)
                self.smooth(l, 0)

    def solve(self,
              max_iters=-1,
              eps=1e-12,
              rel_tol=1e-16,
              tol=1e-14,
              verbose=False):
        '''
        Solve a Poisson problem.

        :parameter max_iters: Specify the maximal iterations. -1 for no limit.
        :parameter eps: Specify a non-zero value to prevent ZeroDivisionError.
        :parameter abs_tol: Specify the absolute tolerance of loss.
        :parameter rel_tol: Specify the tolerance of loss relative to initial loss.
        '''
        # downsample boundary mask before each solve
        for l in range(1, self.n_mg_levels):
            self.downsample_bm(self.bm[l - 1], self.bm[l])
        all_neumann = (self.boundary_types.sum() == 2 * 2 * self.dim)

        # self.r = b - Ax = b    since self.x = 0
        # self.p = self.r = self.r + 0 self.p
        
        
        if all_neumann:
            self.recenter(self.r[0])
        if self.use_multigrid:
            self.apply_preconditioner()
        else:
            self.z[0].copy_from(self.r[0])

        self.update_p()

        self.reduce(self.z[0], self.r[0])
        old_zTr = self.sum[None]
        
        print(f"init rtr = {old_zTr}")
        # tol = max(tol, old_zTr * rel_tol)

        # Conjugate gradients
        it = 0
        start_t = time.time()
        while max_iters == -1 or it < max_iters:
            # self.alpha = rTr / pTAp
            self.compute_Ap()
            self.reduce(self.p, self.Ap)
            pAp = self.sum[None]
            self.alpha[None] = old_zTr / (pAp + eps)

            # self.x = self.x + self.alpha self.p
            self.update_x()

            # self.r = self.r - self.alpha self.Ap
            self.update_r()

            # check for convergence
            self.reduce(self.r[0], self.r[0])
            # plt.imshow(self.r[0].to_numpy())
            # plt.colorbar()
            # plt.show()
            rTr = self.sum[None]

            if verbose:
                print(f'iter {it}, |residual|_2={math.sqrt(rTr)}')

            if rTr < tol:
                end_t = time.time()
                print("[MGPCG] Converged at iter: ", it, " with final error: ", math.sqrt(rTr), " using time: ", end_t-start_t)
                return

            if all_neumann:
                self.recenter(self.r[0])
            # self.z = M^-1 self.r
            if self.use_multigrid:
                self.apply_preconditioner()
            else:
                self.z[0].copy_from(self.r[0])

            # self.beta = new_rTr / old_rTr
            self.reduce(self.z[0], self.r[0])
            new_zTr = self.sum[None]
            self.beta[None] = new_zTr / (old_zTr + eps)

            # self.p = self.z + self.beta self.p
            self.update_p()
            old_zTr = new_zTr

            it += 1

        end_t = time.time()
        print("[MGPCG] Return without converging at iter: ", it, " with final error: ", math.sqrt(rTr), " using time: ", end_t-start_t)

# class MGPCG_2(MGPCG):

#     def __init__(self, boundary_types, N, base_level=3, real=float):
#         super().__init__(boundary_types, N, dim=2, base_level=base_level, real=real)

#         self.u_div = ti.field(float, shape=N)
#         self.p = ti.field(float, shape=N)
#         self.boundary_types = boundary_types

#     @ti.kernel
#     def apply_bc(self, u_horizontal: ti.template(), u_vertical: ti.template()):
#         u_dim, v_dim = u_horizontal.shape
#         for i, j in u_horizontal:
#             if i == 0 and self.boundary_types[0, 0] == 2:
#                 u_horizontal[i,j] = 0
#             if i == u_dim - 1 and self.boundary_types[0, 1] == 2:
#                 u_horizontal[i,j] = 0
#         u_dim, v_dim = u_vertical.shape
#         for i, j in u_vertical:
#             if j == 0 and self.boundary_types[1,0] == 2:
#                 u_vertical[i,j] = 0
#             if j == v_dim - 1 and self.boundary_types[1,1] == 2:
#                 u_vertical[i,j] = 0

#     @ti.kernel
#     def divergence(self, u_horizontal: ti.template(), u_vertical: ti.template()):
#         u_dim, v_dim = self.u_div.shape
#         for i, j in self.u_div:
#             vl = sample(u_horizontal, i, j)
#             vr = sample(u_horizontal, i + 1, j)
#             vb = sample(u_vertical, i, j)
#             vt = sample(u_vertical, i, j + 1)
#             self.u_div[i, j] = vr - vl + vt - vb

#     @ti.kernel
#     def subtract_grad_p(self, u_horizontal: ti.template(), u_vertical: ti.template()):
#         u_dim, v_dim = self.p.shape
#         for i, j in u_horizontal:
#             pr = sample(self.p, i, j)
#             pl = sample(self.p, i-1, j)
#             if i-1 < 0:
#                 pl = 0
#             if i >= u_dim:
#                 pr = 0
#             u_horizontal[i,j] -= (pr - pl)
#         for i, j in u_vertical:
#             pt = sample(self.p, i, j)
#             pb = sample(self.p, i, j-1)
#             if j-1 < 0:
#                 pb = 0
#             if j >= v_dim:
#                 pt = 0
#             u_vertical[i,j] -= pt - pb

#     def solve_pressure_MGPCG(self, verbose):
#         self.init(self.u_div, -1)
#         self.solve(max_iters=400, verbose=verbose, tol = 1.e-12)
#         self.get_result(self.p)

#     def Poisson(self, u_horizontal, u_vertical, verbose = False):
#         self.apply_bc(u_horizontal, u_vertical)
#         self.divergence(u_horizontal, u_vertical)
#         self.solve_pressure_MGPCG(verbose = verbose)
#         self.subtract_grad_p(u_horizontal, u_vertical)
#         self.apply_bc(u_horizontal, u_vertical)

class MGPCG_2(MGPCG):

    def __init__(self, boundary_types, boundary_mask, boundary_vel, N, levelset, p_rho, p_rho_w, base_level=3, real=float):
        super().__init__(boundary_types, boundary_mask, N, levelset, p_rho, p_rho_w, dim=2, base_level=base_level, real=real)

        self.u_div = ti.field(float, shape=N)
        self.boundary_vel = boundary_vel
        self.p = ti.field(float, shape=N)
        self.rhox = ti.field(float, shape=N)
        self.rhoy = ti.field(float, shape=N)
        self.boundary_types = boundary_types
        self.boundary_mask = boundary_mask
        self.left_vel = ti.field(float, shape=())
        
    @ti.kernel
    def apply_bc(self, u_horizontal: ti.template(), u_vertical: ti.template()):
        u_dim, v_dim = u_horizontal.shape
        for i, j in u_horizontal:
            if i == 0 and self.boundary_types[0, 0] == 2:
                u_horizontal[i,j] = 0
            elif i == 0 and self.boundary_types[0, 0] == 3:
                u_horizontal[i,j] = u_horizontal[u_dim-1,j]
            if i == u_dim - 1 and self.boundary_types[0, 1] == 2:
                u_horizontal[i,j] = 0
            elif i == u_dim - 1 and self.boundary_types[0, 1] == 3:
                u_horizontal[i,j] = u_horizontal[0,j]
        u_dim, v_dim = u_vertical.shape
        for i, j in u_vertical:
            if j == 0 and self.boundary_types[1,0] == 2:
                u_vertical[i,j] = 0
            if j == v_dim - 1 and self.boundary_types[1,1] == 2:
                u_vertical[i,j] = 0
                
        # for i, j in self.boundary_mask:
        #     if self.boundary_mask[i, j] > 0:
        #         u_vertical[i, j] = 0
        #         u_vertical[i, j + 1] = 0
        #         u_horizontal[i, j] = 0
        #         u_horizontal[i + 1, j] = 0

    @ti.kernel
    def divergence(self, u_horizontal: ti.template(), u_vertical: ti.template()):
        u_dim, v_dim = self.u_div.shape
        dx = 1.0 / u_dim
        for i, j in self.u_div:
            if self.boundary_mask[i,j] <= 0:
                vl = sample(u_horizontal, i, j)
                vr = sample(u_horizontal, i + 1, j)
                vb = sample(u_vertical, i, j)
                vt = sample(u_vertical, i, j + 1)
                self.u_div[i, j] = (vr - vl + vt - vb)
                # # multiply by rho
                # smoke_density = sample(smoke, i, j)[3] # between 0 and 1
                # rho = lerp(1.0, smoke_rho, smoke_density)
                # self.u_div[i, j] *= rho
            else:
                self.u_div[i, j] = 0.0
            # self.u_div[i, j] += (self.levelset.phi[i, j] - self.levelset.phi_temp[i, j]) / self.true_dt

#     @ti.kernel
#     def subtract_grad_p(self, u_horizontal: ti.template(), u_vertical: ti.template()):
#         u_dim, v_dim = self.p.shape
#         dx = 1.0 / u_dim
#         for i, j in u_horizontal:
#             pr = sample(self.p, i, j)
#             pl = sample(self.p, i-1, j)
#             if i-1 < 0:
#                 pl = 0
#             if i >= u_dim:
#                 pr = 0
#             # H = self.levelset.heaviside((ti.Vector([i, j]) + ti.Vector.unit(2, 1) * 0.5) * (1.0) / u_dim)
#             # rho = self.dt / (self.p_rho_w * H + self.p_rho * (1 - H))
#             # u_horizontal[i,j] -= rho * (pr - pl)
            
#             H1 = self.levelset.heaviside((ti.Vector([i, j]) + 0.5) * (1.0) / u_dim)
#             rho1 = self.dt / (self.p_rho_w * H1 + self.p_rho * (1 - H1))
#             H2 = self.levelset.heaviside((ti.Vector([i-1, j]) + 0.5) * (1.0) / u_dim)
#             rho2 = self.dt / (self.p_rho_w * H2 + self.p_rho * (1 - H2))
#             u_horizontal[i,j] -= rho1 * pr - rho2 * pl
#             self.rhox[i, j] = rho1 + rho2
#         for i, j in u_vertical:
#             pt = sample(self.p, i, j)
#             pb = sample(self.p, i, j-1)
#             # # interpolate rho on face
#             # smoke_density_t = sample(smoke, i, j)[3] # between 0 and 1
#             # smoke_density_b = sample(smoke, i, j-1)[3] # between 0 and 1
#             # smoke_density = 0.5 * (smoke_density_t + smoke_density_b)
#             # rho = lerp(1.0, smoke_rho, smoke_density)
#             if j-1 < 0:
#                 pb = 0
#             if j >= v_dim:
#                 pt = 0
#             # u_vertical[i,j] -= 1./rho * (pt - pb)
#             # H = self.levelset.heaviside((ti.Vector([i, j]) + ti.Vector.unit(2, 0) * 0.5) * (1.0) / u_dim)
#             # rho = self.dt / (self.p_rho_w * H + self.p_rho * (1 - H))
#             # u_vertical[i,j] -= rho * (pt - pb)
#             H1 = self.levelset.heaviside((ti.Vector([i, j]) + 0.5) * (1.0) / u_dim)
#             rho1 = self.dt / (self.p_rho_w * H1 + self.p_rho * (1 - H1))
#             H2 = self.levelset.heaviside((ti.Vector([i, j-1]) + 0.5) * (1.0) / u_dim)
#             rho2 = self.dt / (self.p_rho_w * H2 + self.p_rho * (1 - H2))
#             u_vertical[i,j] -= rho1 * pt - rho2 * pb
#             self.rhoy[i, j] = rho1 + rho2


    @ti.kernel
    def subtract_grad_p(self, u_horizontal: ti.template(), u_vertical: ti.template()):
        u_dim, v_dim = self.p.shape
        dx = 1.0 / v_dim
        for i, j in u_horizontal:
            ar, br = i, j
            al, bl = i-1, j
            pr = sample(self.p, i, j)
            pl = sample(self.p, i-1, j)
            if i-1 < 0:
                pl = pr
            if i >= u_dim:
                pr = pl
            
            # H2 = self.levelset.heaviside((ti.Vector([i-0.5, j]) + 0.5) * (1.0) / u_dim)
            # rho2 = self.dt / (self.p_rho_w * H2 + self.p_rho * (1 - H2))
            rho = self.levelset.heaviside((ti.Vector([i-0.5, j]) + 0.5) * dx)
            rho2 = self.dt / rho
            # rho2 = self.dt / self.p_rho_w
            # if H2 < 1:
            #     rho2 = self.dt / self.p_rho
            u_horizontal[i,j] -= (rho2 * pr - rho2 * pl)
            # self.rhox[i, j] = rho1 + rho2
        for i, j in u_vertical:
            pt = sample(self.p, i, j)
            pb = sample(self.p, i, j-1)
            if j-1 < 0:
                pb = pt
            if j >= v_dim:
                pt = pb
            
            # H2 = self.levelset.heaviside((ti.Vector([i, j-0.5]) + 0.5) * (1.0) / u_dim)
            # rho2 = self.dt / (self.p_rho_w * H2 + self.p_rho * (1 - H2))
            rho = self.levelset.heaviside((ti.Vector([i, j-0.5]) + 0.5) * dx)
            rho2 = self.dt / rho
            # rho2 = self.dt / self.p_rho_w
            # if H2 < 1:
            #     rho2 = self.dt / self.p_rho
            u_vertical[i,j] -= (rho2 * pt - rho2 * pb)
            # self.rhoy[i, j] = rho1 + rho2

    def solve_pressure_MGPCG(self, verbose):
        self.init(self.u_div, -1)
        self.solve(max_iters=100, verbose=verbose, tol = 1.e-12)
        self.get_result(self.p)
        # self.get_Ap(self.p, self.Ap)

    def Poisson(self, u_horizontal, u_vertical, dt, verbose = False):
        # self.dt = dt
        self.dt = 1.0
        self.true_dt = dt
        self.apply_bc(u_horizontal, u_vertical)
        self.divergence(u_horizontal, u_vertical)
        self.solve_pressure_MGPCG(verbose = verbose)
        self.subtract_grad_p(u_horizontal, u_vertical)
        self.apply_bc(u_horizontal, u_vertical)

