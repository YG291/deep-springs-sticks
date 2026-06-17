import numpy as np
import torch

from .model import MeshNet


def build_random_spec(num_features, depth, width, n_labels, dim=2, n_mix=2, seed=0):
    """
    Generate a MeshNet layers_spec with random pair-dimensional wiring.

    Each (layer, model) reads `dim` randomly chosen raw input dims (1- or 2-D meshes).
    Every model past layer 0 mixes `n_mix` randomly chosen previous-layer outputs
    into its target (random "pair" swaps when n_mix == 2).
    """
    rng = np.random.default_rng(seed)
    spec = []
    for layer_idx in range(depth):
        layer = []
        for _ in range(width):
            d = min(dim, num_features)
            u_dims = sorted(int(x) for x in rng.choice(num_features, size=d, replace=False))
            entry = {"u_dims": u_dims}
            if layer_idx > 0:
                entry["sources"] = [(int(rng.integers(width)), int(rng.integers(n_labels)))
                                    for _ in range(n_mix)]
            layer.append(entry)
        spec.append(layer)
    return spec


def train_meshnet(u, y, depth, width, n_sticks=2, dim=2, n_mix=2,
                  epochs=100, lr=1e-2, friction=5.0, temp=0.1, kb=1.0, k=1.0, k2=0.5,
                  seed=0, verbose=True):
    """
    Build and train a MeshNet of the given depth/width on (u, y) by MSE regression.

    u : [n_points, num_features]   inputs (auto-normalized to [0,1] per dim)
    y : [n_points, n_labels]       regression targets

    Returns (net, losses, normalize), where `normalize` maps raw inputs into the
    [0,1] domain the net was trained on (apply it before net.forward at inference).
    Requires depth >= 2 (layer 0 has no learnable W/b).
    """
    num_features = u.shape[1]
    n_labels = y.shape[1]

    # normalize inputs to [0,1] per dim so every mesh shares one SympyCalc
    u_lo = u.min(dim=0).values
    u_hi = u.max(dim=0).values
    span = (u_hi - u_lo).clamp_min(1e-8)
    normalize = lambda x: (x - u_lo) / span
    u_norm = normalize(u)
    boundaries = (torch.zeros(num_features), torch.ones(num_features))

    spec = build_random_spec(num_features, depth, width, n_labels,
                             dim=dim, n_mix=n_mix, seed=seed)
    net = MeshNet(spec, n_sticks=n_sticks, boundaries=boundaries, n_labels=n_labels,
                  friction=friction, temp=temp, kb=kb, k=k, k2=k2)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        opt.zero_grad()
        out = net.forward(u_norm, y)                 # list of [n_points, n_labels]
        pred = torch.stack(out, dim=0).mean(dim=0)   # average the final-layer meshes
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for layer in net.layers:
                for m in layer:
                    m.k.clamp_(min=1e-1)
                    m.k2.clamp_(min=1e-1)
                    #YOU MUST have this because otherwise the k,k2 will become 0
                    #k and k2 cannot be too close to 0 because then our adam optimizes against noise
        losses.append(loss.item())
    with torch.no_grad():
        net.forward(u_norm, y)
    return net, losses, normalize
