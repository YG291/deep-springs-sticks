
from sympy import *
from sympy.physics.mechanics import *
# from symengine import *

import numpy as np

import torch
import torch.nn as nn

from .utils import verbose_display
from itertools import product

from concurrent.futures import ThreadPoolExecutor, as_completed

import torchsde



class GroupGS3DE(nn.Module):
    """Grouped GS3DE models. It splits the input features into groups and instantiates a GS3DE model for each group."""
    def __init__(
        self, max_features, n_sticks, boundaries, n_labels, friction=0, temp=0, k=1, M=1,
        kb=1.38064852e-23, k2=0, verbose=False, group_strategy="sequential", parallel=False
    ):
        """
        Parameters:
          max_features (int or float): If int, the maximum number of features per group.
            If float, a fraction of the total features.
          n_sticks: int or tensor/array defining the discretization for each input dimension.
          boundaries: tuple (u_min, u_max) with each a tensor of shape (d,), where d is the number of input features.
          n_labels: int, number of labels (output dimensions).
          group_strategy: "sequential" or "random".
          parallel: whether to create underlying models in parallel.
        """
        super().__init__()
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        
        # Determine feature dimension from boundaries.
        self.num_features = boundaries[0].shape[0]
        self.num_labels = n_labels
        
        # Determine max_features based on type.
        self.max_features = max_features if isinstance(max_features, int) else int(max_features * self.num_features)
        self.group_strategy = group_strategy

        # Split the feature indices into groups.
        self.groups, self.groups_idx = self.split_features(self.num_features, strategy=group_strategy)

        models = []
        if parallel:
            def create_gs3de_model(idx, group_idx):
                group_boundaries = (boundaries[0][group_idx], boundaries[1][group_idx])
                return GS3DE(n_sticks[group_idx], group_boundaries, n_labels, friction, temp, k, M, kb, k2, verbose)
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(create_gs3de_model, idx, group_idx): idx for idx, group_idx in enumerate(self.groups_idx)}
                for future in as_completed(futures):
                    models.append(future.result())
        else:
            for idx, group_idx in enumerate(self.groups_idx):
                if verbose:
                    print(f"Creating GS3DE for group {idx+1}/{len(self.groups_idx)}...")
                group_boundaries = (boundaries[0][group_idx], boundaries[1][group_idx])
                model = GS3DE(n_sticks[group_idx], group_boundaries, n_labels, friction, temp, k, M, kb, k2, verbose)
                models.append(model)

        self.models = nn.ModuleList(models)
        self.state_size = sum([model.state_size for model in self.models])
        self.state_sizes = torch.tensor([model.state_size for model in self.models])
        self.cstate_sizes = torch.cat((torch.tensor([0]), self.state_sizes.cumsum(dim=0)))  # Cumulative state sizes

    def split_features(self, num_features, strategy="sequential"):
        """
        Splits feature indices [0, 1, ..., num_features-1] into groups.
        Returns a tuple (groups, groups_idx), where each element in groups_idx is a tensor of indices.
        """
        indices = torch.arange(num_features)
        if strategy == "sequential":
            split_indices = torch.split(indices, self.max_features)
        elif strategy == "random":
            permuted_indices = torch.randperm(num_features)
            split_indices = torch.split(permuted_indices, self.max_features)
        else:
            raise ValueError("strategy must be 'sequential' or 'random'")
        return split_indices, split_indices

    def f(self, t, theta):
        return torch.cat([model.f(t, self.get_theta_group(theta, i)) for i, model in enumerate(self.models)], dim=1)

    def g(self, t, theta):
        return torch.cat([model.g(t, self.get_theta_group(theta, i)) for i, model in enumerate(self.models)], dim=1)

    def cost(self, theta):
        return torch.sum(torch.stack([model.cost(self.get_theta_group(theta, i)) for i, model in enumerate(self.models)]), dim=0)

    def loss(self, theta, u_i, y_i):
        loss = torch.sum((self.num_y_prediction(u_i, theta) - y_i) ** 2)
        return loss / u_i.shape[0]

    def y_prediction(self, u, output_type="mean"):
        if output_type == "stack":
            return torch.stack([torch.tensor(model.y_prediction(u[:, indices])) for model, indices in zip(self.models, self.groups_idx)], dim=0)
        elif output_type == "mean":
            return torch.mean(torch.stack([torch.tensor(model.y_prediction(u[:, indices])) for model, indices in zip(self.models, self.groups_idx)]), dim=0)
        else:
            raise ValueError("output_type must be 'stack' or 'mean'")

    def num_y_prediction(self, u, theta, output_type="mean"):
        if output_type == "stack":
            return torch.stack([torch.tensor(model.num_y_prediction(u[:, indices], self.get_theta_group(theta, i, predict=True)))
                                for i, (model, indices) in enumerate(zip(self.models, self.groups_idx))], dim=0)
        elif output_type == "mean":
            return torch.mean(torch.stack([torch.tensor(model.num_y_prediction(u[:, indices], self.get_theta_group(theta, i, predict=True)))
                                             for i, (model, indices) in enumerate(zip(self.models, self.groups_idx))]), dim=0)
        else:
            raise ValueError("output_type must be 'stack' or 'mean'")

    def get_theta_group(self, theta, group, predict=False):
        if predict:
            return theta[self.cstate_sizes[group]:self.cstate_sizes[group + 1]]
        return theta[:, self.cstate_sizes[group]:self.cstate_sizes[group + 1]]

    def update_data(self, new_u_i, new_y_i):
        """
        Update the potential energy terms in each underlying GS3DE model based on new data.
        new_u_i is expected to be a tensor of shape (num_features, ...), and new_y_i of shape (n_labels, ...).
        If the number of features changes, the grouping is re-computed.
        """
        # if new_u_i.shape[0] != self.num_features:
        #     self.num_features = new_u_i.shape[0]
        #     self.groups, self.groups_idx = self.split_features(self.num_features, strategy=self.group_strategy)
        for idx, group_idx in enumerate(self.groups_idx):
            new_group = new_u_i[:, group_idx]
            self.models[idx].update_data(new_group, new_y_i)


class GS3DE(nn.Module):
    def __init__(self, n_sticks, boundaries, n_labels, friction=0, temp=0, k=1, M=1,
                 kb=1.38064852e-23, k2=0, verbose=False):
        """
        Generalized SS stochastic differential equation (GS3DE), where the non-potential Lagrangian is computed once.

        Parameters:
          n_sticks: int or array-like. If int, all input dimensions use the same number of pieces.
                    If array-like, it must have the same length as the input dimension.
          boundaries: tuple of two tensors/arrays, (u_min, u_max), each of shape (d,), where d is the input dimension.
          n_labels: int, number of labels (used to determine the size of the output space).
          friction, temp, k, M, kb, k2: physical parameters.
          verbose: optional verbosity flag.
          
        Note: u_i and y_i are not used in initialization. They must be supplied via update_data.
        """
        super().__init__()
        # Determine input dimension from boundaries.
        d = boundaries[0].shape[0]
        if isinstance(n_sticks, int):
            self.n_sticks = torch.ones(d, dtype=int) * n_sticks
        elif isinstance(n_sticks, torch.Tensor):
            self.n_sticks = n_sticks.clone().detach().to(torch.int)
        else:
            self.n_sticks = torch.tensor(n_sticks, dtype=torch.int)

        assert self.n_sticks.shape[0] == d, "n_sticks must have the same length as input dimension"

        self.noise_type = "diagonal"
        self.sde_type = "ito"
        # Boundaries must be provided.
        self.u_min = boundaries[0]
        self.u_max = boundaries[1]
        self.ell = ((self.u_max - self.u_min) / self.n_sticks)

        # Set state_size using n_labels.
        self.state_size = torch.prod(self.n_sticks + 1) * 2 * n_labels

        self.k = k
        self.k2 = k2
        self.M = M / torch.prod(self.n_sticks)
        self.kb = kb

        self.temp = temp
        self.friction = float(friction) # fric is defined as friction per unit mass
        self.eta_cte = float(np.sqrt(2 * self.friction * temp * kb / self.M))

        self.n_labels = n_labels  # used for symbolic variable shapes

        # Initialize symbolic variables, kinetic and elastic energies.
        self._init_symbols(n_labels)
        self._init_kinetic_energy()
        self._init_elastic_energy()
        self._init_lagrangian() 

    def _init_symbols(self, n_labels):
        # Create symbolic arrays with shape: (n_sticks+1) x ... x (n_labels)
        shape = tuple([int(n) for n in np.append(self.n_sticks + 1, n_labels)])
        self.symbols_shape = shape
        self.N = np.prod(shape)
        self.x_symbols = np.empty(shape, dtype=object)
        self.dx_symbols = np.empty_like(self.x_symbols, dtype=object)
        self.ddx_symbols = np.empty_like(self.x_symbols, dtype=object)

        for index in np.ndindex(shape):
            self.x_symbols[index] = dynamicsymbols(f"x_{''.join(map(str, index))}")
            self.dx_symbols[index] = self.x_symbols[index].diff()
            self.ddx_symbols[index] = self.dx_symbols[index].diff()

    def _init_kinetic_energy(self):
        # Translational kinetic energy.
        ktr = 0
        for i in range(len(self.symbols_shape) - 1):
            slice_front = [slice(None)] * len(self.symbols_shape)
            slice_back  = [slice(None)] * len(self.symbols_shape)
            slice_front[i] = slice(1, None)
            slice_back[i]  = slice(None, -1)
            difference = (self.dx_symbols[tuple(slice_front)] + self.dx_symbols[tuple(slice_back)]) ** 2
            difference = difference.ravel()
            ktr = Add(ktr, (self.M.item() / 8) * Add(*difference))
        self.ktr = simplify(ktr)

        # Rotational kinetic energy.
        krot = 0
        for i in range(len(self.symbols_shape) - 1):
            slice_front = [slice(None)] * len(self.symbols_shape)
            slice_back  = [slice(None)] * len(self.symbols_shape)
            slice_front[i] = slice(1, None)
            slice_back[i]  = slice(None, -1)
            difference = (self.dx_symbols[tuple(slice_front)] - self.dx_symbols[tuple(slice_back)]) ** 2
            difference = difference.ravel()
            krot = Add(krot, (self.M.item() / 24) * Add(*difference))
        self.krot = simplify(krot)

    def _init_elastic_energy(self):
        # Elastic energy to keep neighboring states near the rest length.
        uelastic = 0
        if self.k2 != 0:
            for i in range(len(self.x_symbols.shape) - 1):
                slice_front = [slice(None)] * len(self.x_symbols.shape)
                slice_back  = [slice(None)] * len(self.x_symbols.shape)
                slice_front[i] = slice(1, None)
                slice_back[i]  = slice(None, -1)
                difference = (self.x_symbols[tuple(slice_front)] - self.x_symbols[tuple(slice_back)] - float(self.ell[i])) ** 2
                difference = difference.ravel()
                uelastic = Add(uelastic, self.k2 / 2 * Add(*difference))
        self.uelastic_symbols = simplify(uelastic)

    def _init_lagrangian(self):
        # Compute non-potential Lagrangian (kinetic + elastic).
        self.L_nonpot = simplify(Add(self.ktr, self.krot, -self.uelastic_symbols))
        # Initially, set the potential energy U to zero.
        self.U = 0  
        LM_nonpot = LagrangesMethod(self.L_nonpot, self.x_symbols.flatten())
        LM_nonpot.form_lagranges_equations()
        self.mass_matrix = simplify(LM_nonpot.mass_matrix)
        try:
            self.inv_mass_matrix = simplify(LM_nonpot.mass_matrix.inv())
        except Exception as e:
            print(e)
            self.inv_mass_matrix = simplify(LM_nonpot.mass_matrix.pinv())
        self.forcing_nonpot = simplify(LM_nonpot.forcing)
        self._update_total_lagrangian()
        self.ypred = lambda u: lambdify([*self.x_symbols.flatten()], self.y_prediction(u))

    def _compute_potential_energy(self, u_i, y_i):
        """Compute the potential energy U that depends on new data u_i and y_i."""
        i_idx = self.find_box(u_i)
        point_pred = self.y_prediction(u_i, i_idx)
        difference = (point_pred - y_i.cpu().numpy()) ** 2
        difference = difference.ravel()
        U = Add(self.k / 2 * Add(*difference))
        self.U = simplify(U)
    
    def _update_total_lagrangian(self):
        """Update the full Lagrangian and derived dynamics using the current U."""
        self.lagrangian = Add(self.L_nonpot, -self.U)
        self.forcing_pot = Matrix([-diff(self.U, q) for q in self.x_symbols.flatten()])
        self.forcing_vector = Add(self.forcing_nonpot, self.forcing_pot)
        self.evol_dynamics = simplify(self.inv_mass_matrix * self.forcing_vector)
        self.lambdified_dyn = lambdify(
            [*self.x_symbols.flatten(), *self.dx_symbols.flatten()],
            self.evol_dynamics - self.friction * self.dx_symbols.reshape(-1, 1)
        )
        self.ue = lambdify([*self.x_symbols.flatten()], self.U)

    def update_data(self, new_u_i, new_y_i):
        """
        Update the model's potential energy based on new data.
        This recomputes U (and its derived dynamics) without redoing the non-potential computations.
        """
        self._compute_potential_energy(new_u_i, new_y_i)
        self._update_total_lagrangian()

    def lamd(self, i, u, flip_ind=None, div_by_ell=True):
        """Return the lambda function for the i-th dimension."""
        val = u - i * self.ell - self.u_min
        if div_by_ell:
            val /= self.ell
        if flip_ind is not None:
            val[flip_ind] *= -1
        val[np.abs(val) < 1e-8] = 0
        return val

    def find_box(self, u):
        """Return the box where the input u is located."""
        i = ((u - self.u_min) / self.ell).to(torch.int)
        return torch.clip(i, torch.tensor(0), self.n_sticks - 1)

    def y_prediction(self, u, i=None):
        return self._y_prediction(u, i)

    def _y_prediction(self, u, i=None):
        u = torch.clamp(u, self.u_min.unsqueeze(0), self.u_max.unsqueeze(0))
        if i is None:
            i = self.find_box(u)
        ld = self.lamd(i, u)
        dimension = i.shape[1]
        offsets = torch.tensor(list(product([0, 1], repeat=dimension)))
        grid_points = i.unsqueeze(1) + offsets.unsqueeze(0)
        weights = torch.prod((1 - ld.unsqueeze(1)) * (offsets.unsqueeze(0) == 0) + ld.unsqueeze(1) * (offsets.unsqueeze(0) != 0), axis=2)
        assert torch.all(weights.sum(axis=1) - 1 < 1e-5), f"Weights do not sum to 1. Sum: {weights.sum(axis=1)}"
        grid_points_np = grid_points.cpu().numpy().astype(int)
        weights_np = weights.cpu().numpy()
        index_tuple = tuple(grid_points_np[..., i] for i in range(grid_points_np.shape[-1]))
        symbols = self.x_symbols[index_tuple]
        lin_combinations = weights_np[..., None] * symbols
        pred = np.apply_along_axis(lambda col: Add(*col), axis=1, arr=lin_combinations)
        return pred

    def num_y_prediction(self, u, theta):
        q = theta[:self.N].t()
        ypred = self.ypred(u)
        return ypred(*q)

    def f(self, t, theta):
        """Compute the time derivative f(t, theta)."""
        q = theta[:, :self.N].t()
        dq = theta[:, self.N:].t()
        ddqdt = torch.tensor(self.lambdified_dyn(*q, *dq))
        ddqdt_shape = ddqdt.shape
        ddqdt = ddqdt.squeeze()
        if ddqdt_shape[-1] == 1:
            ddqdt = ddqdt.unsqueeze(-1)
        step = torch.vstack([dq, ddqdt])
        return step.t()

    def g(self, t, theta):
        # return self.eta_cte * torch.ones_like(theta)
        return torch.cat([torch.zeros_like(theta[:, :self.N]), self.eta_cte*torch.ones_like(theta[:, self.N:])], dim=1)

    def cost(self, theta):
        q = theta[:, :self.N].t()
        return self.ue(*q)
    
    def dcost(self, theta):
        dUdt = self.U.diff()
        q = theta[:, :self.N].t()
        dq = theta[:, self.N:].t()
        dUdt_num = lambdify([*self.x_symbols.flatten(), *self.dx_symbols.flatten()], dUdt)
        return dUdt_num(*q, *dq)
    
    def dw(self, theta):
        "derivative of the work with respect to time"
        q = theta[:, :self.N].t()
        dq = theta[:, self.N:].t()

        # force of the system
        ddqdt = torch.tensor(self.lambdified_dyn(*q, *dq)).squeeze()

        # F \dot v
        dwdt = torch.sum((ddqdt * dq), dim=0) * self.M
        return dwdt


    def loss(self, theta, u_i, y_i):
        """Compute the MSE loss using provided data."""        
        # loss = torch.sum((torch.tensor(self.num_y_prediction(u_i, theta)) - y_i) ** 2)
        # return loss / u_i.shape[0]
        return 2 * self.cost(theta) / (u_i.shape[0]* self.k)
    
class SympyCalc:
    def __init__(self, n_sticks, boundaries, n_labels, M=1):
        """
        To be initialized per DEQ instance
        """
        d = boundaries[0].shape[0]
        if isinstance(n_sticks, int):
            self.n_sticks = torch.ones(d, dtype=int) * n_sticks
        elif isinstance(n_sticks, torch.Tensor):
            self.n_sticks = n_sticks.clone().detach().to(torch.int)
        else:
            self.n_sticks = torch.tensor(n_sticks, dtype=torch.int)

        # Set state_size using n_labels.
        self.u_min, self.u_max = boundaries[0], boundaries[1]
        self.ell = (self.u_max - self.u_min) / self.n_sticks
        self.n_labels = n_labels
        self.M = M / torch.prod(self.n_sticks)
        self.state_size = torch.prod(self.n_sticks + 1) * 2 * n_labels
        self.k  = symbols('k')

        # Initialize symbolic variables, kinetic and elastic energies.
        self._init_symbols(n_labels)
        self._init_kinetic_energy()
        self._init_elastic_energy()
        self._init_lagrangian() 
        self._init_regr_params()
        self._init_potential()
        self._init_force()
        self._init_jacobians()
        self._lambdify_all()


    def _init_symbols(self, n_labels):
        # Create symbolic arrays with shape: (n_sticks+1) x ... x (n_labels)
        shape = tuple([int(n) for n in np.append(self.n_sticks + 1, n_labels)])
        self.symbols_shape = shape
        self.N = np.prod(shape)
        self.x_symbols = np.empty(shape, dtype=object)
        self.dx_symbols = np.empty_like(self.x_symbols, dtype=object)
        self.ddx_symbols = np.empty_like(self.x_symbols, dtype=object)

        for index in np.ndindex(shape):
            self.x_symbols[index] = dynamicsymbols(f"x_{''.join(map(str, index))}")
            self.dx_symbols[index] = self.x_symbols[index].diff()
            self.ddx_symbols[index] = self.dx_symbols[index].diff()

    def _init_kinetic_energy(self):
        # Translational kinetic energy.
        ktr = 0
        for i in range(len(self.symbols_shape) - 1):
            slice_front = [slice(None)] * len(self.symbols_shape)
            slice_back  = [slice(None)] * len(self.symbols_shape)
            slice_front[i] = slice(1, None)
            slice_back[i]  = slice(None, -1)
            difference = (self.dx_symbols[tuple(slice_front)] + self.dx_symbols[tuple(slice_back)]) ** 2
            difference = difference.ravel()
            ktr = Add(ktr, (self.M.item() / 8) * Add(*difference))
        self.ktr = simplify(ktr)

        # Rotational kinetic energy.
        krot = 0
        for i in range(len(self.symbols_shape) - 1):
            slice_front = [slice(None)] * len(self.symbols_shape)
            slice_back  = [slice(None)] * len(self.symbols_shape)
            slice_front[i] = slice(1, None)
            slice_back[i]  = slice(None, -1)
            difference = (self.dx_symbols[tuple(slice_front)] - self.dx_symbols[tuple(slice_back)]) ** 2
            difference = difference.ravel()
            krot = Add(krot, (self.M.item() / 24) * Add(*difference))
        self.krot = simplify(krot)

    def _init_elastic_energy(self):
        # Elastic energy to keep neighboring states near the rest length.
        springs = []
        for i in range(len(self.x_symbols.shape) - 1):
            slice_front = [slice(None)] * len(self.x_symbols.shape)
            slice_back  = [slice(None)] * len(self.x_symbols.shape)
            slice_front[i] = slice(1, None)
            slice_back[i]  = slice(None, -1)
            difference = (self.x_symbols[tuple(slice_front)] - self.x_symbols[tuple(slice_back)] - float(self.ell[i])) ** 2
            springs.extend(difference.ravel().tolist())

        # order of springs is axis 0, axis 1, axis 2, ...
        self.n_springs = len(springs)
        self.k2_symbols = np.array(symbols(f'k2_0:{self.n_springs}'))
        uelastic = Add(*[self.k2_symbols[e] / 2 * springs[e] for e in range(self.n_springs)])
        #each spring scaled symbolically by its own k2

        self.uelastic_symbols = simplify(uelastic)

    def _init_lagrangian(self):
        # Compute non-potential Lagrangian (kinetic + elastic).
        self.L_nonpot = simplify(Add(self.ktr, self.krot, -self.uelastic_symbols))
        # Initially, set the potential energy U to zero.
        self.U = 0  
        LM_nonpot = LagrangesMethod(self.L_nonpot, self.x_symbols.flatten())
        LM_nonpot.form_lagranges_equations()
        self.mass_matrix = simplify(LM_nonpot.mass_matrix)
        try:
            self.inv_mass_matrix = simplify(LM_nonpot.mass_matrix.inv())
        except Exception as e:
            print(e)
            self.inv_mass_matrix = simplify(LM_nonpot.mass_matrix.pinv())
        self.forcing_nonpot = simplify(LM_nonpot.forcing)
    
    def _init_regr_params(self):
        self.X_symbols = np.array(symbols(f'X_0:{self.n_labels}'))
        #X = W * y_hat + b with n_labels for each of such X symbols
        self.n_nodes = int(np.prod([int(n) + 1 for n in self.n_sticks]))
        #written as the number of "grid points" in the grid (i.e. intersections between grid lines)
        self.w_symbols = np.array(symbols(f'w_0:{self.n_nodes}'))
        #weights depending on which 'box' u is in

    def _init_potential(self):
        pos = self.x_symbols.reshape(self.n_nodes, self.n_labels)
        # position is (node from _init_regr_params, label)
        U = 0
        for i in range(self.n_labels):
            pred_i = Add(*[self.w_symbols[node] * pos[node, i] for node in range(self.n_nodes)])
            #prediction for label i is summ over nodes (w[node]*x[node, i]) 
            U = Add(U, (self.k / 2) * (pred_i - self.X_symbols[i])**2)
        self.U = U
        self.forcing_pot = Matrix([-diff(U, q) for q in self.x_symbols.flatten()])
        #potential force on each node is -∂U/∂x = -k(pred - X)*w
        #and, pred = wz* -> Jz_pot = -kw(wtranspose)
    
    def _init_force(self):
        self.F = self.forcing_nonpot + self.forcing_pot
        # net force F = ∂L/∂z*
        
        #direct consequence:
        #∂F/∂z* = ∂F_nonpot/∂z* + ∂F_pot/∂z* -> second piece is pointwise
    
    def _init_jacobians(self):
        x_flat = Matrix(self.x_symbols.flatten())
        X_vector = Matrix(self.X_symbols)

        #shared nonpotential derivatives
        self.Jz_nonpot = self.forcing_nonpot.jacobian(x_flat)
        #∂F_nonpot/∂z*
        self.Jk2_nonpot = self.forcing_nonpot.jacobian(Matrix(self.k2_symbols))
        #∂F/∂k2

        #pointwise (potential) derivatives
        self.Jz_pot = self.forcing_pot.jacobian(x_flat)
        #∂F_pot/∂z*
        self.Jk_pot = diff(self.forcing_pot, self.k)
        #∂F/∂k
        self.JX_pot = self.forcing_pot.jacobian(X_vector)
        #∂F/∂X

    def _lambdify_all(self):
        state_args = [*self.x_symbols.flatten(), *self.dx_symbols.flatten()]
        nonpot_args = [*state_args, *self.k2_symbols]
        pot_args = [*state_args, *self.w_symbols, *self.X_symbols, self.k]

        #shared nonpot
        self.f_nonpot = lambdify(nonpot_args, self.forcing_nonpot, cse=True)
        self.jz_nonpot = lambdify(nonpot_args, self.Jz_nonpot, cse = True)
        self.jk2_nonpot = lambdify(nonpot_args, self.Jk2_nonpot, cse = True)

        #pointwise (pot)
        self.f_pot = lambdify(pot_args, self.forcing_pot, cse=True)
        self.jz_pot = lambdify(pot_args, self.Jz_pot, cse=True)
        self.jk_pot = lambdify(pot_args, self.Jk_pot, cse=True)
        self.jx_pot = lambdify(pot_args, self.JX_pot, cse=True)

        self.inv_mass = np.array(self.inv_mass_matrix.tolist(), dtype=np.float64)
        #const mass matr

class _Equilibrium(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, k, k2, model, u_model):
        z = model.forward(u_model, X)
        q = z[0, :model.N]
        yhat = model.y_prediction_approx(u_model, q)
        ctx.model = model
        ctx.u_model = u_model
        ctx.save_for_backward(q, model.w, X)
        return yhat
    
    @staticmethod
    def backward(ctx, grad_yhat):
        model = ctx.model
        u_model = ctx.u_model
        q,w,X = ctx.saved_tensors
        with torch.enable_grad():
            q_leaf = q.detach().requires_grad_(True)
            yhat = model.y_prediction_approx(u_model, q_leaf)
            dL_dz = torch.autograd.grad(yhat, q_leaf, grad_outputs=grad_yhat)[0]
        dL_dX, dL_dk, dL_dk2 = model.backward_vjp(q, w, X, dL_dz)
        dL_dk  = torch.as_tensor(dL_dk,  dtype=X.dtype, device=X.device)
        dL_dk2 = torch.as_tensor(dL_dk2, dtype=X.dtype, device=X.device)
        return dL_dX, dL_dk, dL_dk2, None, None
    
    #Critical: if we want to end up parameterizing k, k2 then include the other two

class DEQ(nn.Module):
    def __init__(self, n_sticks, boundaries, n_labels, sympy, 
                 friction=0, temp=0, k=1, M=1, kb=1.38064852e-23, k2=0, verbose=False, tau=0.05):
        super().__init__()
        self.noise_type = "diagonal"
        self.sde_type = "ito"

        d = boundaries[0].shape[0]
        if isinstance(n_sticks, int):
            self.n_sticks = torch.ones(d, dtype=int) * n_sticks
        elif isinstance(n_sticks, torch.Tensor):
            self.n_sticks = n_sticks.clone().detach().to(torch.int)
        else:
            self.n_sticks = torch.tensor(n_sticks, dtype=torch.int)
        assert self.n_sticks.shape[0] == d, "n_sticks must have the same length as input dimension"

        self.u_min = boundaries[0]
        self.u_max = boundaries[1]
        self.ell = ((self.u_max - self.u_min) / self.n_sticks)
        self.n_labels = n_labels
        self.N = int(torch.prod(self.n_sticks + 1)) * n_labels
        self.state_size = torch.prod(self.n_sticks + 1) * 2 * n_labels

        self.k = nn.Parameter(torch.tensor(float(k)))
        self.k2 = nn.Parameter(torch.full((sympy.n_springs,), float(k2)))
        self.M = M / torch.prod(self.n_sticks)
        self.kb = kb
        self.temp = temp
        self.friction = float(friction)
        self.eta_cte = float(np.sqrt(2 * self.friction * temp * kb / self.M))
        #eta_cte is the noise term

        self.sympy = sympy
        #sympy is literally an instance of SympyCalc

        self.register_buffer('tau', torch.as_tensor(float(tau)))
        self.n_boxes = int(torch.prod(self.n_sticks))
        self._grid_constants()

    def _grid_constants(self):
        d = self.u_min.shape[0]
        offsets = torch.tensor(list(product([0, 1], repeat=d)), dtype=torch.long)
        node_radix = (self.n_sticks + 1).tolist()
        box_radix = self.n_sticks.tolist()
        node_strides = [1] * d
        box_strides = [1] * d
        for j in range(d - 2, -1, -1):
            node_strides[j] = node_strides[j + 1] * node_radix[j + 1]
            box_strides[j] = box_strides[j + 1] * box_radix[j + 1]
        self.register_buffer('offsets', offsets)
        self.register_buffer('node_strides', torch.tensor(node_strides, dtype=torch.long))
        self.register_buffer('box_strides', torch.tensor(box_strides, dtype=torch.long))

    def find_box_approx(self, u):
        device = u.device
        um = self.u_min.to(device)
        el = self.ell.to(device)
        u = torch.clamp(u, um.unsqueeze(0), self.u_max.to(device).unsqueeze(0))
        memberships = []
        for j in range(u.shape[1]):
            n_j = int(self.n_sticks[j])
            lines = um[j] + el[j] * torch.arange(n_j + 1, device=device, dtype=u.dtype)
            tau_j = self.tau * el[j]
            ind = torch.sigmoid((u[:, j:j + 1] - lines.unsqueeze(0)) / tau_j)
            m_j = ind[:, :-1] - ind[:, 1:]
            m_j = m_j / (m_j.sum(dim=1, keepdim=True) + 1e-12)
            memberships.append(m_j)
        return memberships

    def y_prediction_approx(self, u, theta):
        memberships = self.find_box_approx(u)
        batch_size = u.shape[0]
        device = u.device
        d = u.shape[1]
        um = self.u_min.to(device)
        el = self.ell.to(device)
        offsets = self.offsets.to(device)
        node_strides = self.node_strides.to(device)
        box_strides = self.box_strides.to(device)
        n_sticks = self.n_sticks.to(device)

        ypred = torch.zeros(batch_size, self.n_labels, device=device)
        theta_points = theta[:self.N].view(-1, self.n_labels)
        for i in range(self.n_boxes):
            grid_idx = (i // box_strides) % n_sticks
            box_start = grid_idx.to(u.dtype) * el + um
            ld = torch.clamp((u - box_start.unsqueeze(0)) / el.unsqueeze(0), 0.0, 1.0)
            ld_exp = ld.unsqueeze(1)
            off_exp = offsets.unsqueeze(0)
            corner_w = torch.prod((1 - ld_exp) * (off_exp == 0) + ld_exp * (off_exp == 1), dim=2)
            corner_indices = torch.sum((grid_idx.unsqueeze(0) + offsets) * node_strides.unsqueeze(0), dim=1)
            pred_i = corner_w @ theta_points[corner_indices]
            gate = torch.ones(batch_size, device=device)
            for j in range(d):
                gate = gate * memberships[j][:, int(grid_idx[j])]
            ypred = ypred + gate.unsqueeze(-1) * pred_i
        return ypred # output [n_points, n_labels]
    
    def soft_weights(self, u):
        """same as y_prediction_approx, but we get [n_points, n_nodes]
        which is used for f_pot(*q, *dq, *w_row, ...)"""
        memberships = self.find_box_approx(u)
        batch_size = u.shape[0]
        device = u.device
        d = u.shape[1]
        um = self.u_min.to(device); el = self.ell.to(device)
        offsets = self.offsets.to(device)
        node_strides = self.node_strides.to(device)
        box_strides = self.box_strides.to(device)
        n_sticks = self.n_sticks.to(device)
        n_nodes = int(torch.prod(self.n_sticks + 1))

        w = torch.zeros(batch_size, n_nodes, device=device)
        for i in range(self.n_boxes):
            grid_idx = (i // box_strides) % n_sticks
            box_start = grid_idx.to(u.dtype) * el + um
            ld = torch.clamp((u - box_start.unsqueeze(0)) / el.unsqueeze(0), 0.0, 1.0)
            ld_exp = ld.unsqueeze(1); off_exp = offsets.unsqueeze(0)
            corner_w = torch.prod((1 - ld_exp) * (off_exp == 0) + ld_exp * (off_exp == 1), dim=2)  # [batch, 2^d]
            corner_indices = torch.sum((grid_idx.unsqueeze(0) + offsets) * node_strides.unsqueeze(0), dim=1)  # [2^d]
            gate = torch.ones(batch_size, device=device)
            for j in range(d):
                gate = gate * memberships[j][:, int(grid_idx[j])]
            w[:, corner_indices] += gate.unsqueeze(-1) * corner_w
        return w   # [n_points, n_nodes]
    
    def net_force(self, q, dq, w, X):
        """
        q, dq - torch[N]
        w- torch[n_points, n_nodes]
        X- torch[n_points, n_labels]

        They are converted into numpy because lambdified functions are NumPy
        -> converting the pot portions to closed-form aggregate is wayy faster
        """
        qn  = q.detach().cpu().numpy()
        dqn = dq.detach().cpu().numpy()
        wn  = w.detach().cpu().numpy()
        Xn  = X.detach().cpu().numpy()
        k= float(self.k)
        k2 = self.k2.detach().cpu().numpy()

        #nonpot
        F = np.asarray(self.sympy.f_nonpot(*qn, *dqn, *k2)).reshape(self.N)

        #pot
        w_args = [wn[:,j] for j in range(wn.shape[1])]
        #n_nodes arrays, each [n_points]
        X_args = [Xn[:,l] for l in range(Xn.shape[1])]
        #n_labels arrays, each [n_points]
        fpot = np.asarray(self.sympy.f_pot(*qn, *dqn, *w_args, *X_args, k))
        #(N, 1, n_points)
        F=F+ fpot.reshape(self.N, -1).sum(axis=1)

        return torch.as_tensor(F, dtype = q.dtype, device = q.device)
    
    def f(self, t, theta):
        """Compute the time derivative f(t, theta).
        
        This is the drift"""
        q = theta[:, :self.N]
        dq = theta[:, self.N:]
        F = self.net_force(q[0], dq[0], self.w, self.X)
        inv_mass = torch.as_tensor(self.sympy.inv_mass, dtype =theta.dtype, device = theta.device)
        accel = inv_mass@F - self.friction * dq[0]
        return torch.cat([dq[0], accel]).unsqueeze(0)

    def g(self, t, theta):
        """
        diffusion (for diagonal noise)
        """
        return torch.cat(
            [torch.zeros_like(theta[:, :self.N]), 
             self._eta_cur*torch.ones_like(theta[:, self.N:])], dim=1)
        #_eta_cte defined in forward
    
    def forward(self, u, X, y0=None, 
                dt = 0.01, window=20, max_iters=100, tol=1e-3, floor = 1e-3, cooling = 0.85):
        """
        eta_cte : base noise amplitude
        cooling : rate at which we decrease _eta_cur geometrically
        _eta_cur : noise injected in the window
        """
        self.w = self.soft_weights(u)
        self.X = X
        if y0 is None:
            y0 = torch.zeros(1,2*self.N, device=u.device)
        ts = torch.tensor([0.0, window*dt], device=u.device)
        y=y0
        self._eta_cur = self.eta_cte

        #all this _eta_cur stuff is defined to construct a geometric decrease in alpha
        #and corresponding noise level -> alpha = 0 forces F to converge to approx 0
        #-> we force it to converge to exactly 0.0

        #affects "multi-basin stochasticity of the gradient estimate" which I dont understand so its ok...
        for k in range(max_iters):
            if self._eta_cur < floor:
                self._eta_cur = 0.0
            self._eta_cur *= cooling
            ys = torchsde.sdeint(self, y, ts, dt=dt, method="euler")
            y = ys[-1]
            q, dq = y[:,:self.N], y[:,self.N:]
            F=self.net_force(q[0], dq[0], self.w, self.X)
            if F.norm() < tol and self._eta_cur == 0.0:
                break
            #F is ~0
        self.z_star = y
        return y
    
    def assemble_Jz(self, q, w):
        """
        builds ∂F/∂z*
        """
        # q: torch [N] (equilibrium positions)
        qn = q.detach().cpu().numpy()
        dqn = np.zeros(self.N)
        wn = w.detach().cpu().numpy()
        k = float(self.k)
        k2 = self.k2.detach().cpu().numpy()

        #nonpot
        Jz = np.asarray(self.sympy.jz_nonpot(*qn, *dqn, *k2)).reshape(self.N, self.N)

        #pot
        w_args = [wn[:,j] for j in range(wn.shape[1])]
        X_args = [np.zeros(wn.shape[0]) for _ in range(self.n_labels)]
        Jz_pot = np.asarray(self.sympy.jz_pot(*qn, *dqn, *w_args, *X_args, k))
        Jz = Jz + Jz_pot.reshape(self.N, self.N, -1).sum(axis=2)
        
        return torch.as_tensor(Jz, dtype=q.dtype, device=q.device)
    
    def backward_vjp(self, q, w, X, dL_dz):
        """
        vector jacobian product, that is

        q: [N] equilibrium positions
        X: [n_points, n_labels]
        Computes JX, Jk, Jk2, v
        """
        Jz = self.assemble_Jz(q, w)
        v = torch.linalg.solve(Jz.t(), -dL_dz)
        #this solves the v_row = -(∂L/∂z*)*(∂F/∂z*)^-1

        qn  = q.detach().cpu().numpy()
        dqn = np.zeros(self.N)
        wn  = w.detach().cpu().numpy()
        Xn  = X.detach().cpu().numpy()
        vn  = v.detach().cpu().numpy()
        k = float(self.k)
        k2 = self.k2.detach().cpu().numpy()

        w_args = [wn[:, j] for j in range(wn.shape[1])]
        X_args = [Xn[:, l] for l in range(Xn.shape[1])]

        #now we compute dL/dX per data point X [n_points, n_labels]
        JX = np.asarray(self.sympy.jx_pot(*qn, *dqn, *w_args, *X_args, k))
        dL_dX = np.einsum('m, mln->nl', vn, JX)

        #compute dL/dk sum over points
        Jk = np.asarray(self.sympy.jk_pot(*qn, *dqn, *w_args, *X_args, k))   # [N, 1, n_points]
        dL_dk = float(vn @ Jk.reshape(self.N, -1).sum(axis=1))

        #compute dL/dk2 sum over points
        Jk2 = np.asarray(self.sympy.jk2_nonpot(*qn, *dqn, *k2)).reshape(self.N, self.sympy.n_springs)
        dL_dk2 = vn @ Jk2
        #each of these vn @ Jk2 -> distributing vn to each Jacobian via matrix mult
        #not float(vn@Jk2) cuz Jk2 needs to be a vector

        dL_dX = torch.as_tensor(dL_dX, dtype=q.dtype, device=q.device)
        return dL_dX, dL_dk, dL_dk2
    
    def predict_equilibrium(self, u, X):
        return _Equilibrium.apply(X, self.k, self.k2, self, u)

class MeshNet(nn.Module):
    def __init__(self, layers_spec, n_sticks, boundaries, n_labels,
                 friction=0, temp=0, k=1, M=1, kb=1.38064852e-23, k2=0, tau=0.05):
        """
        note that we must have k2> 0 otherwise inv Jz doesnt exist
        """
        super().__init__()
        self.layers_spec = layers_spec

        #example:
        # layers_spec = [
        # [ {"u_dims": [0,1]}, {"u_dims": [2,3]}, ... ],              
        # [ {"u_dims": [0,1], "sources": [(0,0),(1,0)]}, ... ],     
        # ]


        self.n_labels = n_labels
        u_min, u_max = boundaries

        self._sympy_cache = {}
        self.layers = nn.ModuleList()
        self.W = nn.ParameterDict()
        self.b = nn.ParameterDict()

        for layer_idx, layer in enumerate(layers_spec):
            layer_models = nn.ModuleList()
            for model_idx, spec in enumerate(layer):
                u_dims = spec["u_dims"]
                d = len(u_dims)
                model_bounds = (u_min[u_dims], u_max[u_dims])
                sig = (int(n_sticks), d, n_labels)
                if sig not in self._sympy_cache:
                    self._sympy_cache[sig] = SympyCalc([n_sticks] * d, model_bounds, n_labels, M=M)
                sympy = self._sympy_cache[sig]
                deq = DEQ([n_sticks] * d, model_bounds, n_labels, sympy,
                          friction=friction, temp=temp, k=k, M=M, kb=kb, k2=k2, tau=tau)
                layer_models.append(deq)

                if "sources" in spec:
                    n_src = len(spec["sources"])
                    key = f"{layer_idx}_{model_idx}"
                    self.W[key] = nn.Parameter(0.1 * torch.randn(self.n_labels, n_src))
                    self.b[key] = nn.Parameter(torch.zeros(self.n_labels))
            self.layers.append(layer_models)

    def forward(self, u, y):
        """
        u: [n_points, num_features]
        y: [n_points, n_labels]
        """
        prev_outputs = None
        for layer_idx, layer_models in enumerate(self.layers):
            spec_layer = self.layers_spec[layer_idx]
            #each model's entry in layers_spec tracks that model's u_dims
            layer_outputs = []
            for model_idx, model in enumerate(layer_models):
                spec = spec_layer[model_idx]
                u_model = u[:,spec["u_dims"]]
                # dim [n_points, d]
                if "sources" in spec:
                    key = f"{layer_idx}_{model_idx}"
                    S = torch.stack([prev_outputs[m][:,dim] for m,dim in spec["sources"]], dim = 1)
                    X_model=S@self.W[key].t() + self.b[key]

                    #this is the Wyhat + b part of the MeshNet

                else:
                    X_model = y
                yhat = model.predict_equilibrium(u_model, X_model)
                layer_outputs.append(yhat)
            prev_outputs = layer_outputs
        return prev_outputs
    
    def predict(self, u):
        """
        out of sample inference
        works by freezing layer 0 -> generates yhat_0 -> get X for layer 1
        -> generates yhat_1 after equilibrium -> get X for layer 2
        -> generates yhat_2 after equilibrium -> get X for layer 3 -> ...
        -> final layer result after equilibrium is prediction

        """
        prev_outputs = None
        for layer_idx, layer_models in enumerate(self.layers):
            spec_layer = self.layers_spec[layer_idx]
            layer_outputs = []
            for model_idx, model in enumerate(layer_models):
                spec = spec_layer[model_idx]
                u_model = u[:,spec["u_dims"]]
                if "sources" in spec:
                    key = f"{layer_idx}_{model_idx}"
                    S = torch.stack([prev_outputs[m][:,dim] for m,dim in spec["sources"]], dim = 1)
                    X_model=S@self.W[key].t() + self.b[key]
                    yhat = model.predict_equilibrium(u_model, X_model)
                else:
                    qstar = model.z_star[0,:model.N]
                    #remember z_star is [1, 2N] where first N are positions, rest N are velocities

                    #note that z_star is one step stale because it's the z_star of the last forward pass, 
                    #, occuring BEFORE the actual final parameters were optimized by our backprop 
                    #with torch.no_grad():
                    #   net.forward(u_train, y)
                    yhat = model.y_prediction_approx(u_model, qstar)
                layer_outputs.append(yhat)
            prev_outputs = layer_outputs
        return prev_outputs