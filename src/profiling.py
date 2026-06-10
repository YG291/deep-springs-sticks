import cProfile
import sys
import time

import torch
import torch.optim as optim
import torchsde
from torch.utils.data import TensorDataset, DataLoader

from model import (
    Orchestrator, MeshLayer, HiddenPoolingLayer, ConvergenceLayer,
    GroupGS3DE, GS3DE,
)

sys.setrecursionlimit(10000)


def make_data(n_points, num_features, num_labels, eps=0.01):
    u = torch.rand(n_points, num_features)
    y = torch.stack([
        torch.sin(u[:, 0] * u[:, 1]),
        torch.cos(u[:, 0] * u[:, 1]),
    ]).t() + eps * torch.randn(n_points, num_labels)
    return u, y


def build_3mesh(num_features, num_labels, n_sticks, SS_dimension, friction, boundaries):
    kwargs = dict(
        num_features=num_features, max_features=num_features,
        n_sticks=n_sticks, SS_dimension=SS_dimension,
        friction=friction, temp=0.001, k=1, M=1, boundaries=boundaries,
    )
    mesh1 = MeshLayer(n_labels=1, **kwargs)
    mesh2 = MeshLayer(n_labels=1, **kwargs)
    mesh3 = MeshLayer(n_labels=num_labels, **kwargs)
    # Middle layers: mean reduction keeps activations bounded; tanh squash adds nonlinearity.
    # Final layer: softmax-weighted reduction for MNIST-style class scores.
    return Orchestrator([
        mesh1, HiddenPoolingLayer(mesh1, reduction='mean', squash='tanh'),
        mesh2, HiddenPoolingLayer(mesh2, reduction='mean', squash='tanh'),
        mesh3, ConvergenceLayer(mesh3, reduction='softmax'),
    ])


def train(name, predict_fn, theta, loader, num_epochs, lr=0.1):
    optimizer = optim.Adam([theta], lr=lr)
    print(f"\n=== {name} (state_size={theta.shape[1]}) ===")
    start = time.perf_counter()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for u_batch, y_batch in loader:
            optimizer.zero_grad()
            predictions = predict_fn(u_batch, theta[:u_batch.shape[0]])
            loss = torch.mean((predictions - y_batch) ** 2)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"  Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss / len(loader):.4f}")
    print(f"  Train time: {time.perf_counter() - start:.2f}s")


def run_main():
    n_points = 10
    num_features = 3
    num_labels = 2
    SS_dimension = 2
    n_sticks = 2
    friction = 100
    batch_size = 5
    num_epochs = 5

    u_train, y_train = make_data(n_points, num_features, num_labels)
    u_min_vec = torch.min(u_train, dim=0).values
    u_max_vec = torch.max(u_train, dim=0).values
    boundaries_vec = (u_min_vec, u_max_vec)
    boundaries_scalar = (torch.min(u_train).item(), torch.max(u_train).item())
    n_sticks_vec = torch.tensor([n_sticks] * num_features)
    # [(n_sticks+1)^num_features] * 2 * num_labels -> state_size of the ss model
    #this is because we need to model position + velocity, and also we need a phase-space for every dimension so *2 *2 
    n_sticks_vec_group = torch.tensor([n_sticks*3] * num_features)
    n_sticks_vec_raw = torch.tensor([n_sticks + 1] * num_features) # +1  hard coded
    loader = DataLoader(TensorDataset(u_train, y_train), batch_size=batch_size, shuffle=True)
    ts = torch.linspace(0, 1, 2)

    start = time.perf_counter()
    orchestrator = build_3mesh(num_features, num_labels, n_sticks, SS_dimension, friction, boundaries_scalar)
    print(f"\n3-mesh Orchestrator build: {time.perf_counter() - start:.2f}s")
    theta_mesh = torch.rand(batch_size, orchestrator.global_state_size, requires_grad=True)
    train(
        "3-mesh Orchestrator",
        lambda u, th: orchestrator(u, th, ts),
        theta_mesh, loader, num_epochs,
    )

    start = time.perf_counter()
    group_sde = GroupGS3DE(
        max_features=2, n_sticks=n_sticks_vec_group, boundaries=boundaries_vec,
        n_labels=num_labels, friction=friction, temp=0.001, k=1, M=1,
        group_strategy="sequential",
    )
    print(f"\nGroupGS3DE build: {time.perf_counter() - start:.2f}s")
    theta_group = torch.rand(batch_size, group_sde.state_size, requires_grad=True)
    train(
        "GroupGS3DE (max_features=2)",
        lambda u, th: group_sde.num_y_prediction(u, torchsde.sdeint(group_sde, th, ts)[-1]),
        theta_group, loader, num_epochs,
    )

    start = time.perf_counter()
    raw_sde = GS3DE(
        n_sticks=n_sticks_vec_raw, boundaries=boundaries_vec, n_labels=num_labels,
        friction=friction, temp=0.001, k=1, M=1,
    )
    print(f"\nRaw GS3DE build: {time.perf_counter() - start:.2f}s")
    theta_raw = torch.rand(batch_size, raw_sde.state_size, requires_grad=True)
    train(
        "Raw GS3DE (no grouping)",
        lambda u, th: raw_sde.num_y_prediction(u, torchsde.sdeint(raw_sde, th, ts)[-1]),
        theta_raw, loader, num_epochs,
    )


if __name__ == '__main__':
    cProfile.run('run_main()', 'profile_stats.prof')
