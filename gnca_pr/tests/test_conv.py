import pytest
import torch
from torch_geometric.nn import MessagePassing

from gnca.conv import GNCAConv
from gnca.model import GNCAModel

def test_gnca_conv_basic():
    """
    Checksif the layer works on one small graph with NO edge attributes.
    """
    in_channels, out_channels = 16, 32
    # test graph...
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2]
    ], dtype=torch.long)
    num_nodes = 4
    x = torch.randn((num_nodes, in_channels))

    conv = GNCAConv(in_channels, out_channels)
    
    # Check string matches expected
    assert str(conv) == 'GNCAConv(16, 32, edge_dim=0)'

    # FORWARD
    out = conv(x, edge_index)
    
    # Check output shape: [N, in_channels + out_channels]
    # ...because GNCA concatenates input with message
    assert out.size() == (num_nodes, in_channels + out_channels)


def test_gnca_conv_w_edge_features():
    """
    Checks if the layer handles edge attributes correctly!
    """
    in_channels, out_channels = 16, 32
    edge_dim = 4 # test edge features
    
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    num_nodes = 2
    num_edges = edge_index.size(1)
    
    x = torch.randn((num_nodes, in_channels))
    edge_attr = torch.randn((num_edges, edge_dim))

    conv = GNCAConv(in_channels, out_channels, edge_dim=edge_dim)
    
    assert str(conv) == f'GNCAConv(16, 32, edge_dim={edge_dim})'

    out = conv(x, edge_index, edge_attr=edge_attr)
    assert out.size() == (num_nodes, in_channels + out_channels)


def test_gnca_conv_jit_script():
    """
    Checks against TorchScript (JIT).
    """
    in_channels, out_channels = 8, 16
    conv = GNCAConv(in_channels, out_channels)
    
    # try to script the model
    jit_conv = torch.jit.script(conv)
    
    x = torch.randn((4, in_channels))
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    
    out = jit_conv(x, edge_index)
    assert out.size() == (4, in_channels + out_channels)

def test_gnca_model_loop_stability():
    """
    Runs the actual model in a loop (simulating time steps) 
    becaue there can be issues with gradients and values exploding.
    """
    N, F = 5, 4
    x = torch.randn(N, F)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    
    model = GNCAModel(input_channels=F, output_channels=F)
    
    # sim 10 steps of evolution
    for _ in range(10):
        update = model(x, edge_index)
        x = x + update
        x = torch.clamp(x, -10.0, 10.0)

    assert not torch.isnan(x).any(), "Model produced NaN values after looping"


def test_gnca_model_w_edge_attrs():
    """
    Checking if the model correctly passes edge_attr through 
    to the Conv layer.
    """
    N, F = 5, 4
    E = 10
    edge_dim = 3
    
    x = torch.randn(N, F)
    edge_index = torch.randint(0, N, (2, E))
    edge_attr = torch.randn(E, edge_dim)
    
    # Initialize model WITH edge_dim
    model = GNCAModel(input_channels=F, output_channels=F, edge_dim=edge_dim)
    
    # Run forward pass with attributes
    out = model(x, edge_index, edge_attr=edge_attr)
    
    assert out.shape == (N, F)