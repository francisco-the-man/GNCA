"""Simple test script to verify initializers work correctly without pytest."""

import torch
from torch_geometric.data import Data
import sys

# Add the parent directory to path to import gnca
sys.path.insert(0, '/Users/averylouis/Documents/Stanford/CS224W/GNCA_implementation')

from gnca.data import (
    ZeroState,
    OneState,
    RandomState,
    MiddleOneState,
    MultiPointState,
    SphericalizeState,
    FeatureBasedState,
    ConstantState,
    get_initializer,
)


def test_zero_state():
    """Test ZeroState initializer."""
    print("Testing ZeroState...")
    init = ZeroState()
    state = init(num_nodes=10, channels=16)
    assert state.shape == (10, 16)
    assert torch.all(state == 0.0)
    print("✓ ZeroState works correctly")


def test_one_state():
    """Test OneState initializer."""
    print("Testing OneState...")
    init = OneState()
    state = init(num_nodes=10, channels=16)
    assert state.shape == (10, 16)
    assert torch.all(state == 1.0)
    assert state.sum() == 160.0
    print("✓ OneState works correctly")


def test_random_state():
    """Test RandomState initializer."""
    print("Testing RandomState...")
    init = RandomState(distribution='normal', mean=0.0, std=1.0)
    state = init(num_nodes=1000, channels=16)
    assert state.shape == (1000, 16)
    assert abs(state.mean().item()) < 0.1  # Mean should be close to 0
    print("✓ RandomState (normal) works correctly")

    init = RandomState(distribution='uniform', low=-1.0, high=1.0)
    state = init(num_nodes=1000, channels=16)
    assert torch.all(state >= -1.0)
    assert torch.all(state <= 1.0)
    print("✓ RandomState (uniform) works correctly")


def test_middle_one_state():
    """Test MiddleOneState initializer."""
    print("Testing MiddleOneState...")
    init = MiddleOneState()
    state = init(num_nodes=11, channels=16)
    assert torch.all(state[5] == 1.0)  # Middle node
    assert torch.all(state[:5] == 0.0)  # Before middle
    assert torch.all(state[6:] == 0.0)  # After middle
    print("✓ MiddleOneState works correctly")


def test_multi_point_state():
    """Test MultiPointState initializer."""
    print("Testing MultiPointState...")
    init = MultiPointState(indices=[0, 5, 9])
    state = init(num_nodes=10, channels=16)
    assert torch.all(state[0] == 1.0)
    assert torch.all(state[5] == 1.0)
    assert torch.all(state[9] == 1.0)
    assert state.sum() == 3 * 16  # 3 nodes * 16 channels
    print("✓ MultiPointState works correctly")


def test_sphericalize_state():
    """Test SphericalizeState initializer."""
    print("Testing SphericalizeState...")
    base_init = RandomState(distribution='normal', std=2.0)
    init = SphericalizeState(base_init)
    state = init(num_nodes=100, channels=16)

    # Compute norms
    norms = torch.norm(state, p=2, dim=-1)

    # All norms should be approximately 1.0
    assert torch.allclose(norms, torch.ones(100), atol=1e-5)
    print("✓ SphericalizeState works correctly")


def test_feature_based_state():
    """Test FeatureBasedState initializer."""
    print("Testing FeatureBasedState...")
    data = Data(x=torch.randn(10, 8))
    init = FeatureBasedState(padding_channels=8, padding_mode='zeros')
    state = init.from_data(data)

    assert state.shape == (10, 16)
    assert torch.equal(state[:, :8], data.x)
    assert torch.all(state[:, 8:] == 0.0)
    print("✓ FeatureBasedState works correctly")


def test_constant_state():
    """Test ConstantState initializer."""
    print("Testing ConstantState...")
    init = ConstantState(value=0.75)
    state = init(num_nodes=10, channels=16)
    assert torch.all(state == 0.75)
    print("✓ ConstantState works correctly")


def test_get_initializer():
    """Test get_initializer factory function."""
    print("Testing get_initializer...")
    init = get_initializer('zeros')
    assert isinstance(init, ZeroState)

    init = get_initializer('random', distribution='uniform', low=-1, high=1)
    assert isinstance(init, RandomState)

    init = get_initializer('sphericalize', base='random',
                          base_kwargs={'distribution': 'normal', 'std': 0.5})
    assert isinstance(init, SphericalizeState)
    print("✓ get_initializer works correctly")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Running GNCA Initializers Tests")
    print("=" * 60)

    try:
        test_zero_state()
        test_one_state()
        test_random_state()
        test_middle_one_state()
        test_multi_point_state()
        test_sphericalize_state()
        test_feature_based_state()
        test_constant_state()
        test_get_initializer()

        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
