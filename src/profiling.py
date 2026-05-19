import torch
import torchsde
import numpy as np

from torch.utils.data import TensorDataset, DataLoader
import cProfile

import matplotlib.pyplot as plt
import concurrent.futures

import sys
from model import Orchestrator, MeshLayer, PoolingLayer, ConvergenceLayer

sys.setrecursionlimit(10000)  # or a higher value as needed

def run_main():
    eps = 0.01
    n_points = 10
    
    # Let's mock a 3-feature input space (matching your existing synthetic setup)
    # If testing an MNIST-style pipeline later, change 3 to 784
    num_features = 3 
    SS_dimension = 2  # Combination pairing size (e.g., pairs of features)
    num_labels = 2    # Class dimension (e.g., sine vs cosine outputs)

    # Synthetic Input generation
    u_i_train = torch.rand(n_points, num_features)
    y_i_train = (torch.stack([
        torch.sin(u_i_train[:, 0] * u_i_train[:, 1]), 
        torch.cos(u_i_train[:, 0] * u_i_train[:, 1])
    ]).t() + eps * torch.randn(n_points, num_labels))

    # Calculate spatial boundaries
    u_min = torch.min(u_i_train).item()
    u_max = torch.max(u_i_train).item()
    boundaries = (u_min, u_max)

    n_sticks = 2
    fric = 100

    print("--- STEP 1: Initializing Custom Structural Layers ---")
    
    # 1. Instantiate the SDE physics Mesh Layer
    mesh_layer = MeshLayer(
        num_features=num_features,
        max_features=num_features,
        n_sticks=n_sticks,
        n_labels=num_labels,
        SS_dimension=SS_dimension,
        friction=fric,
        temp=0.001,
        k=1,
        M=1,
        boundaries=boundaries
    )
    print(f"Mesh Layer tracking {len(mesh_layer.models)} unique sub-models.")
    print(f"Total state parameter requirements: {mesh_layer.state_size} variables.")

    # 2. Instantiate the Local Pooling Layer (Sub-models -> Features)
    pooling_layer = PoolingLayer(mesh_layer=mesh_layer)
    print(f"Local Pooling map initialized with shape: {pooling_layer.combination_map.shape}")

    # 3. Instantiate the Global Convergence Layer (Features -> Global Class Scores)
    convergence_layer = ConvergenceLayer(mesh_layer=mesh_layer)

    # 4. Bind them sequentially inside the master Orchestrator
    orchestrator = Orchestrator([mesh_layer, pooling_layer, convergence_layer])
    print(f"Orchestrator verified total network state size: {orchestrator.global_state_size}")

    # Prepare standard batch loading data structures
    batch_size = 5
    dataset_train = TensorDataset(u_i_train, y_i_train)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    print("\n--- STEP 2: Running Sample SDE Integration & Forward Pass Validation ---")
    
    t_size = 2
    ts = torch.linspace(0, 1, t_size)
    
    # Simulate an incoming initial global parameter state from your SDE integration engine
    # In execution, the SDE solver outputs a batch of evolved states over time.
    # We mock a batch dimension matching our DataLoader batch sizes
    mock_global_theta = torch.rand(batch_size, orchestrator.global_state_size)

    for i, (u_batch, y_batch) in enumerate(loader_train):
        print(f"\nProcessing Batch {i+1}:")
        print(f" -> Input Batch Shape: {u_batch.shape}")
        print(f" -> Target Batch Shape: {y_batch.shape}")
        
        # Adjust mock parameter batch size if the final batch drops in size
        current_batch_size = u_batch.shape[0]
        batch_theta = mock_global_theta[:current_batch_size, :]
        
        # Execute the full pipeline via the clean Orchestrator wrapper function call syntax
        try:
            final_predictions = orchestrator(u_batch, batch_theta)
            
            print(" -> [SUCCESS] Forward pass executed without shape mismatch or runtime errors!")
            print(f" -> Output Tensor Shape: {final_predictions.shape} (Expected: ({current_batch_size}, {num_labels}))")
            print(" -> Sample output raw scores:\n", final_predictions)
            
        except Exception as e:
            print(f" -> [CRASH] Execution failed during the forward pass.")
            print(f" -> Error Details: {str(e)}")
            raise e
            
        # Break immediately after 1 step—we only care about testing a clean forward pass cycle
        break 

if __name__ == '__main__':
    cProfile.run('run_main()', 'profile_stats.prof')