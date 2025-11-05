"""Simple test script to verify GNCAConv works correctly."""

import torch
import sys

sys.path.insert(0, '/Users/averylouis/Documents/Stanford/CS224W/GNCA_implementation')

from gnca.nn.conv import GNCAConv, GNCAConvSimple


def test_gnca_conv_basic():
    """Test basic GNCAConv functionality."""
    print("Testing GNCAConv basic functionality...")

    # Create a simple graph: 10 nodes, fully connected
    num_nodes = 10
    in_channels = 16
    out_channels = 16

    # Create features and edges
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, 50))

    # Create layer
    conv = GNCAConv(in_channels=in_channels, out_channels=out_channels, persistence=True)

    # Forward pass
    out = conv(x, edge_index)

    # Check output shape (with persistence: in_channels + out_channels)
    expected_shape = (num_nodes, in_channels + out_channels)
    assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"

    print(f"✓ Output shape correct: {out.shape}")
    print("✓ GNCAConv basic test passed")


def test_gnca_conv_without_persistence():
    """Test GNCAConv without persistence."""
    print("\nTesting GNCAConv without persistence...")

    num_nodes = 10
    in_channels = 16
    out_channels = 32

    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, 50))

    # Create layer without persistence
    conv = GNCAConv(in_channels=in_channels, out_channels=out_channels, persistence=False)

    out = conv(x, edge_index)

    # Without persistence, output should be just out_channels
    expected_shape = (num_nodes, out_channels)
    assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"

    print(f"✓ Output shape correct: {out.shape}")
    print("✓ GNCAConv without persistence test passed")


def test_gnca_conv_aggregations():
    """Test different aggregation schemes."""
    print("\nTesting different aggregation schemes...")

    num_nodes = 10
    in_channels = 8
    out_channels = 8

    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, 50))

    for aggr in ['add', 'mean', 'max']:
        conv = GNCAConv(
            in_channels=in_channels,
            out_channels=out_channels,
            aggr=aggr,
            persistence=False
        )
        out = conv(x, edge_index)
        assert out.shape == (num_nodes, out_channels)
        print(f"✓ Aggregation '{aggr}' works correctly")

    print("✓ All aggregation schemes test passed")


def test_gnca_conv_mlp_layers():
    """Test different MLP configurations."""
    print("\nTesting different MLP layer counts...")

    num_nodes = 10
    in_channels = 8
    out_channels = 8

    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, 50))

    for num_layers in [1, 2, 3]:
        conv = GNCAConv(
            in_channels=in_channels,
            out_channels=out_channels,
            mlp_layers=num_layers,
            persistence=False
        )
        out = conv(x, edge_index)
        assert out.shape == (num_nodes, out_channels)
        print(f"✓ MLP with {num_layers} layer(s) works correctly")

    print("✓ MLP layers test passed")


def test_gnca_conv_batch_norm_dropout():
    """Test batch norm and dropout."""
    print("\nTesting batch norm and dropout...")

    num_nodes = 100  # Need more nodes for batch norm
    in_channels = 16
    out_channels = 16

    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, 500))

    conv = GNCAConv(
        in_channels=in_channels,
        out_channels=out_channels,
        batch_norm=True,
        dropout=0.1,
        persistence=False
    )

    # Training mode
    conv.train()
    out_train = conv(x, edge_index)

    # Eval mode
    conv.eval()
    out_eval = conv(x, edge_index)

    assert out_train.shape == (num_nodes, out_channels)
    assert out_eval.shape == (num_nodes, out_channels)

    print("✓ Batch norm and dropout work correctly")


def test_gnca_conv_simple():
    """Test GNCAConvSimple."""
    print("\nTesting GNCAConvSimple...")

    num_nodes = 10
    channels = 16

    x = torch.randn(num_nodes, channels)
    edge_index = torch.randint(0, num_nodes, (2, 50))

    # With persistence
    conv = GNCAConvSimple(channels=channels, persistence=True)
    out = conv(x, edge_index)
    assert out.shape == (num_nodes, channels * 2)  # doubled due to persistence
    print(f"✓ GNCAConvSimple with persistence: {out.shape}")

    # Without persistence
    conv = GNCAConvSimple(channels=channels, persistence=False)
    out = conv(x, edge_index)
    assert out.shape == (num_nodes, channels)
    print(f"✓ GNCAConvSimple without persistence: {out.shape}")

    print("✓ GNCAConvSimple test passed")


def test_message_passing():
    """Test that message passing actually updates node states."""
    print("\nTesting message passing mechanism...")

    # Create a simple path graph: 0 -> 1 -> 2 -> 3
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

    # Initialize with different values
    x = torch.zeros(4, 8)
    x[0] = 1.0  # First node has value 1

    conv = GNCAConv(in_channels=8, out_channels=8, persistence=False)

    # Forward pass
    out = conv(x, edge_index)

    # Node 1 should have received a message from node 0
    # So it should be non-zero
    assert not torch.allclose(out[1], torch.zeros(8)), "Node 1 should receive message from node 0"

    # Node 3 should have received a message from node 2
    assert not torch.allclose(out[3], torch.zeros(8)), "Node 3 should receive message from node 2"

    print("✓ Message passing updates nodes correctly")


def test_gradient_flow():
    """Test that gradients flow through the layer."""
    print("\nTesting gradient flow...")

    num_nodes = 10
    in_channels = 8
    out_channels = 8

    x = torch.randn(num_nodes, in_channels, requires_grad=True)
    edge_index = torch.randint(0, num_nodes, (2, 50))

    conv = GNCAConv(in_channels=in_channels, out_channels=out_channels)

    # Forward pass
    out = conv(x, edge_index)

    # Backward pass
    loss = out.sum()
    loss.backward()

    # Check that gradients exist
    assert x.grad is not None, "Gradients should flow to input"
    assert not torch.allclose(x.grad, torch.zeros_like(x.grad)), "Gradients should be non-zero"

    # Check layer parameters have gradients
    for param in conv.parameters():
        assert param.grad is not None, "Layer parameters should have gradients"

    print("✓ Gradients flow correctly")


def test_edge_weights():
    """Test that edge weights are applied correctly."""
    print("\nTesting edge weights...")

    num_nodes = 10
    in_channels = 8
    out_channels = 8

    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, 50))
    edge_weight = torch.rand(edge_index.size(1))  # Random edge weights

    conv = GNCAConv(in_channels=in_channels, out_channels=out_channels, persistence=False)

    # Forward with edge weights
    out_weighted = conv(x, edge_index, edge_weight=edge_weight)

    # Forward without edge weights
    out_unweighted = conv(x, edge_index, edge_weight=None)

    # Outputs should be different
    assert not torch.allclose(out_weighted, out_unweighted), \
        "Edge weights should affect the output"

    print("✓ Edge weights applied correctly")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Running GNCAConv Tests")
    print("=" * 60)

    try:
        test_gnca_conv_basic()
        test_gnca_conv_without_persistence()
        test_gnca_conv_aggregations()
        test_gnca_conv_mlp_layers()
        test_gnca_conv_batch_norm_dropout()
        test_gnca_conv_simple()
        test_message_passing()
        test_gradient_flow()
        test_edge_weights()

        print()
        print("=" * 60)
        print("✓ All GNCAConv tests passed!")
        print("=" * 60)

    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ Test failed with error: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
