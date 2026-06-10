from sympy import *
from sympy.physics.mechanics import *
# from symengine import *

import numpy as np

import torch
import torch.nn as nn

from .utils import verbose_display
from itertools import product

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from itertools import combinations
from math import comb

import torch
import torchsde

class DynamicsConnector(torch.autograd.Function):
    """Differentiable bridge for the sympy-lambdified SDE drift.

    forward computes step = [dq; ddqdt(q,dq)]^T.  backward uses the symbolic
    Jacobian of ddqdt w.r.t. [q, dq] so gradients can flow through torchsde.
    """

    @staticmethod
    def forward(ctx, theta, lambdified_dyn, lambdified_jac, N):
        q = theta[:, :N].t().detach()
        dq = theta[:, N:].t().detach()
        ddqdt = lambdified_dyn(*q, *dq)
        ddqdt = torch.as_tensor(np.asarray(ddqdt), dtype=theta.dtype, device=theta.device)
        ddqdt_shape = ddqdt.shape
        ddqdt = ddqdt.squeeze()
        if ddqdt_shape[-1] == 1:
            ddqdt = ddqdt.unsqueeze(-1)
        step = torch.vstack([dq, ddqdt]).t()
        ctx.save_for_backward(theta)
        ctx.lambdified_jac = lambdified_jac
        ctx.N = N
        return step

    @staticmethod
    def backward(ctx, grad_output):
        theta, = ctx.saved_tensors
        N = ctx.N
        batch_size = theta.shape[0]
        grad_theta = torch.zeros_like(theta)
        # output[:, :N] = dq = theta[:, N:]  →  identity passthrough
        grad_theta[:, N:] = grad_output[:, :N]
        # output[:, N:] = ddqdt(q, dq)  →  use symbolic Jacobian per batch
        q_np = theta[:, :N].detach().cpu().numpy()
        dq_np = theta[:, N:].detach().cpu().numpy()
        for b in range(batch_size):
            jac = np.asarray(ctx.lambdified_jac(*q_np[b], *dq_np[b]), dtype=np.float64)
            jac = jac.reshape(N, 2 * N)
            jac_t = torch.as_tensor(jac, dtype=theta.dtype, device=theta.device)
            grad_theta[b, :] += grad_output[b, N:] @ jac_t
        return grad_theta, None, None, None

class Orchestrator(nn.Module):
    def __init__(self, layers_list: list):
        super().__init__()
        self.network_layers = nn.ModuleList(layers_list)
        self.global_state_size = 0
        for layer in layers_list:
            self.global_state_size += layer.state_size
    
    def forward(self, u: torch.tensor, global_theta: torch.tensor, time):
        new_input = u
        start = 0
        for layer in self.network_layers:
            if layer.state_size > 0:
                end = start + layer.state_size
                integrated = torchsde.sdeint(layer, global_theta[:, start:end], time)
                new_input = layer(new_input, integrated[-1])
                start = end
            else:
                new_input = layer(new_input)
            start = end
        return new_input

class MeshLayer(nn.Module):
    def __init__(
        self, num_features: int, max_features: int | float, n_sticks: int, n_labels: int, SS_dimension: int, friction=0, temp=0, k=1, M=1,
        kb=1.38064852e-23, k2=0, verbose=False, parallel=False, boundaries: tuple[float, float] = (0, 1)):
        """
        Parameters:
          max_features (int or float): If int, the maximum number of features per group.
            If float, a fraction of the total features.
          n_sticks: int defining the discretization for each input dimension.
          boundaries: tuple (u_min, u_max), both of which are type float
          n_labels: int, number of labels (output dimensions).
          parallel: whether to create underlying models in parallel.
        """
        super().__init__()
        self.SS_dimension = SS_dimension
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        
        # Determine feature dimension from boundaries.
        self.num_features = num_features
        self.n_sticks = n_sticks
        self.boundaries = boundaries
        self.indices = self._split_features()

        self.num_labels = n_labels
        
        # Determine max_features based on type.
        self.max_features = max_features if isinstance(max_features, int) else int(max_features * self.num_features)

        models = []
        self.models = nn.ModuleList(models)
        self._instantiate_GS3DE(n_labels, friction, temp, k, M, kb, k2, verbose, parallel)
        self.state_size = sum([model.state_size for model in self.models])
        #integer sum fo the state size of this mesh layer --summed from constituent models
        self.state_sizes = torch.tensor([model.state_size for model in self.models]) 
        #pytorch tensor of the state sizes for this mesh layer
        self.cstate_sizes = torch.cat((torch.tensor([0]), self.state_sizes.cumsum(dim=0)))  
        #cumulative state sizes
    
    def _split_features(self):
        feature_combinations = list(torch.tensor(item, dtype=torch.int64) for item in combinations(range(self.num_features), self.SS_dimension))
        return torch.tensor(feature_combinations, dtype=torch.int64)

    def _instantiate_GS3DE(self, n_labels, friction, temp, k, M, kb, k2, verbose, parallel):
        u_min = torch.tensor([self.boundaries[0]] * self.SS_dimension)
        u_max = torch.tensor([self.boundaries[1]] * self.SS_dimension)
        def create_gs3de():
            return GS3DE(self.n_sticks, [u_min, u_max], n_labels, friction, temp, k, M, kb, k2, verbose)
        if parallel:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(create_gs3de) for _ in self.indices]
                for future in as_completed(futures):
                    self.models.append(future.result())
        else:
            for _ in self.indices:
                self.models.append(create_gs3de())
    
    def forward(self, u: torch.tensor, layer_theta: torch.tensor):
        """u: batch size * dimension of input matrix
            layer_theta: 
        """
        outputs = []
        for i, model in enumerate(self.models):
            uSlice = u[:,self.indices[i]]
            start = self.cstate_sizes[i]
            end = self.cstate_sizes[i+1]
            thetaSlice = layer_theta[:,start: end]
            prediction = model.num_y_prediction(uSlice, thetaSlice)
            outputs.append(prediction)
        return torch.stack(outputs, dim=1)
    
    def f(self, time, layer_theta):
        drift_components = []
        for i, model in enumerate(self.models):
            start = self.cstate_sizes[i]
            end = self.cstate_sizes[i+1]
            thetaSlice = layer_theta[:, start:end]
            block_drift = model.f(time, thetaSlice)
            drift_components.append(block_drift)
        return torch.cat(drift_components, dim=1)
    
    def g(self, time, layer_theta):
        noise_components = []
        for i, model in enumerate(self.models):
            start = self.cstate_sizes[i]
            end = self.cstate_sizes[i+1]
            thetaSlice = layer_theta[:, start:end]
            block_noise = model.g(time, thetaSlice)
            noise_components.append(block_noise)
        return torch.cat(noise_components, dim=1)


class PoolingLayer(nn.Module):
    _REDUCTIONS = ("mean", "softmax")
    _SQUASH = (None, "tanh", "sigmoid")

    def __init__(self, mesh_layer: MeshLayer, reduction: str = "mean", squash: str | None = None):
        super().__init__()
        if reduction not in self._REDUCTIONS:
            raise ValueError(f"reduction must be one of {self._REDUCTIONS}; got {reduction!r}")
        if squash not in self._SQUASH:
            raise ValueError(f"squash must be one of {self._SQUASH}; got {squash!r}")
        num_cols = comb(mesh_layer.num_features - 1, mesh_layer.SS_dimension - 1)
        combination_map = torch.empty((mesh_layer.num_features, num_cols), dtype=torch.int64)
        for pixel in range(mesh_layer.num_features):
            # Find every row index (Model ID) where either column equals this pixel
            model_ids = (mesh_layer.indices == pixel).any(dim=1).nonzero().squeeze()
            combination_map[pixel] = model_ids
        self.register_buffer('combination_map', combination_map)
        self.reduction = reduction
        self.squash = squash
        self.state_size = 0

    def _reduce(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        if self.reduction == "mean":
            return x.mean(dim=dim)
        # softmax-weighted sum: bounded by the values being aggregated (unlike logsumexp)
        weights = torch.softmax(x, dim=dim)
        return (weights * x).sum(dim=dim)

    def _maybe_squash(self, x: torch.Tensor) -> torch.Tensor:
        if self.squash == "tanh":
            return torch.tanh(x)
        if self.squash == "sigmoid":
            return torch.sigmoid(x)
        return x

    def forward(self, inputs: torch.tensor):
        grouped_mesh_inputs = inputs[:, self.combination_map, :]
        return self._reduce(grouped_mesh_inputs, dim=2)

class HiddenPoolingLayer(PoolingLayer):
    def forward(self, inputs: torch.tensor):
        return self._maybe_squash(super().forward(inputs).squeeze(-1))

class ConvergenceLayer(PoolingLayer):
    def forward(self, inputs: torch.tensor):
        pool_group_reduced = super().forward(inputs)
        convergence_reduced = self._reduce(pool_group_reduced, dim=1)
        return self._maybe_squash(convergence_reduced)


class GroupGS3DE(nn.Module):
    """Grouped GS3DE models."""
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
        preds = torch.stack([
            model.num_y_prediction(u[:, indices], self.get_theta_group(theta, i))
            for i, (model, indices) in enumerate(zip(self.models, self.groups_idx))
        ], dim=0)
        if output_type == "stack":
            return preds
        if output_type == "mean":
            return preds.mean(dim=0)
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
            self.ddx_symbols[index] = self.dx_symbols[index].diff() # TODO 1: should be a parameter

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
                difference = (self.x_symbols[tuple(slice_front)] - self.x_symbols[tuple(slice_back)] - self.ell[i]) ** 2
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
        symbols_list = [*self.x_symbols.flatten()]
        self.ypred = lambda u: lambdify(symbols_list, self.y_prediction(u))

    def _compute_potential_energy(self, u_i, y_i: torch.Tensor | None):
        """Compute the potential energy U that depends on new data u_i and y_i."""
        if y_i is None:
            self.U = 0
            return
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
        all_vars = [*self.x_symbols.flatten(), *self.dx_symbols.flatten()]
        dyn_expr = self.evol_dynamics - self.friction * self.dx_symbols.reshape(-1, 1)
        self.lambdified_dyn = lambdify(all_vars, dyn_expr)
        self.lambdified_dyn_jac = lambdify(all_vars, Matrix(dyn_expr).jacobian(all_vars))
        self.ue = lambdify([*self.x_symbols.flatten()], self.U)

    def update_data(self, new_u_i=None, new_y_i=None):
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
            val = val / self.ell
        if flip_ind is not None:
            val = val.clone()
            val[flip_ind] = val[flip_ind] * -1
        return torch.where(val.abs() < 1e-8, torch.zeros_like(val), val)

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
<<<<<<< HEAD
        u = torch.clamp(u, self.u_min.unsqueeze(0), self.u_max.unsqueeze(0))
        i = self.find_box(u)
        ld = self.lamd(i, u)
        dimension = i.shape[1]
        offsets = torch.tensor(
            list(product([0, 1], repeat=dimension)),
            device=u.device, dtype=torch.int64,
        )
        weights = torch.prod(
            (1 - ld.unsqueeze(1)) * (offsets == 0) + ld.unsqueeze(1) * (offsets != 0),
            dim=2,
        )
        grid_points = i.unsqueeze(1) + offsets.unsqueeze(0)
        theta_grid = theta[:, :self.N].reshape(-1, *self.symbols_shape)
        batch_size = u.shape[0]
        n_corners = offsets.shape[0]
        batch_idx = torch.arange(batch_size, device=u.device).view(-1, 1).expand(-1, n_corners)
        corner_idx = (batch_idx,) + tuple(grid_points[..., d] for d in range(dimension))
        corner_values = theta_grid[corner_idx]
        return (weights.unsqueeze(-1) * corner_values).sum(dim=1)
=======
        q = theta[:self.N].t()
        ypred = self.ypred(u)
        return ypred(*q)
>>>>>>> parent of b152a6f (forward pass bug fixes)

    def f(self, t, theta):
        """Compute the time derivative f(t, theta) with autograd-aware bridge."""
        return DynamicsConnector.apply(theta, self.lambdified_dyn, self.lambdified_dyn_jac, self.N)

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
        #return 2 * self.cost(theta) / (u_i.shape[0]* self.k)
        y_pred = self.num_y_prediction(u_i, theta)
        loss = torch.mean((y_pred - y_i) ** 2)
        return loss