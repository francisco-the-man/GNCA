import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from pygsp.graphs import Bunny
import numpy as np

from gnca import GNCAModel

def get_graph_edges():
    # same as in growing_shapes.py
    graph = Bunny()
    rows, cols = graph.W.nonzero()
    edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)
    pos = torch.tensor(graph.coords, dtype=torch.float)
    return pos, edge_index

def visualize():
    # setup (this is for when using colab)
    device = torch.device('cpu')
    pos, edge_index = get_graph_edges()
    
    # normalize target exactly as in the training
    target_mean = pos.mean(dim=0)
    target_scale = pos.abs().max()
    
    # Initial seed (collapsed state)
    # I.e. very close to zero (the mean) but slightly biased in the direction of pos
    current_state = ((pos - target_mean) / target_scale) * 0.01
    
    model = GNCAModel(input_channels=3, output_channels=3, output_activation='tanh')
    model.load_state_dict(torch.load('checkpoints/bunny_gnca.pt', map_location=device))
    model.eval()
    
    # Setup Plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # update function for the animation
    def update(frame):
        nonlocal current_state
        ax.clear()
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_title(f"GNCA Growth - Step {frame}")
        
        np_state = current_state.numpy()
        ax.scatter(np_state[:, 0], np_state[:, 1], np_state[:, 2], 
                   c=np_state[:, 2], cmap='viridis', s=10) # Color by Z-height
        
        # Evolve
        delta = model(current_state, edge_index) # unrolling
        current_state = current_state + delta
        current_state = torch.clamp(current_state, -1.0, 1.0)
        
        return ax,
    
    ani = animation.FuncAnimation(fig, update, frames=60, interval=100)
    ani.save('bunny_growth.gif', writer='pillow', fps=10)
    print("Saved gif")

if __name__ == '__main__':
    visualize()