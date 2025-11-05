"""Basic test script to verify initializers work without PyG dependency."""

import torch
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
    ConstantState,
    get_initializer,
)


def test_zero_state():
    """Test ZeroState initializer."""
    print("Testing ZeroState...")
    init = ZeroState()
    state = init(num_nodes=10, channels=16)
    assert state.shape == (10, 16), f"Expected shape (10, 16), got {state.shape}"
    assert torch.all(state == 0.0), "Expected all zeros"
    print("✓ ZeroState works correctly")


def test_one_state():
    """Test OneState initializer."""
    print("Testing OneState...")
    init = OneState()
    state = init(num_nodes=10, channels=16)
    assert state.shape == (10, 16), f"Expected shape (10, 16), got {state.shape}"
    assert torch.all(state == 1.0), "Expected all ones"
    assert state.sum() == 160.0, f"Expected sum 160, got {state.sum()}"
    print("✓ OneState works correctly")


def test_random_state():
    """Test RandomState initializer."""
    print("Testing RandomState...")

    # Test normal distribution
    init = RandomState(distribution='normal', mean=0.0, std=1.0)
    state = init(num_nodes=1000, channels=16)
    assert state.shape == (1000, 16), f"Expected shape (1000, 16), got {state.shape}"
    mean_val = abs(state.mean().item())
    assert mean_val < 0.1, f"Mean should be close to 0, got {mean_val}"
    print("✓ RandomState (normal) works correctly")

    # Test uniform distribution
    init = RandomState(distribution='uniform', low=-1.0, high=1.0)
    state = init(num_nodes=1000, channels=16)
    assert torch.all(state >= -1.0), "All values should be >= -1.0"
    assert torch.all(state <= 1.0), "All values should be <= 1.0"
    print("✓ RandomState (uniform) works correctly")


def test_middle_one_state():
    """Test MiddleOneState initializer."""
    print("Testing MiddleOneState...")
    init = MiddleOneState()
    state = init(num_nodes=11, channels=16)

    assert torch.all(state[5] == 1.0), "Middle node should be all ones"
    assert torch.all(state[:5] == 0.0), "Nodes before middle should be zero"
    assert torch.all(state[6:] == 0.0), "Nodes after middle should be zero"
    print("✓ MiddleOneState works correctly")


def test_multi_point_state():
    """Test MultiPointState initializer."""
    print("Testing MultiPointState...")
    init = MultiPointState(indices=[0, 5, 9])
    state = init(num_nodes=10, channels=16)

    assert torch.all(state[0] == 1.0), "Node 0 should be all ones"
    assert torch.all(state[5] == 1.0), "Node 5 should be all ones"
    assert torch.all(state[9] == 1.0), "Node 9 should be all ones"
    expected_sum = 3 * 16  # 3 nodes * 16 channels
    assert state.sum() == expected_sum, f"Expected sum {expected_sum}, got {state.sum()}"
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
    assert torch.allclose(norms, torch.ones(100), atol=1e-5), \
        f"All norms should be ~1.0, got range [{norms.min():.6f}, {norms.max():.6f}]"
    print("✓ SphericalizeState works correctly")


def test_constant_state():
    """Test ConstantState initializer."""
    print("Testing ConstantState...")
    init = ConstantState(value=0.75)
    state = init(num_nodes=10, channels=16)
    assert torch.all(state == 0.75), "All values should be 0.75"
    print("✓ ConstantState works correctly")


def test_get_initializer():
    """Test get_initializer factory function."""
    print("Testing get_initializer...")

    init = get_initializer('zeros')
    assert isinstance(init, ZeroState), "Should return ZeroState instance"

    init = get_initializer('random', distribution='uniform', low=-1, high=1)
    assert isinstance(init, RandomState), "Should return RandomState instance"

    init = get_initializer('sphericalize', base='random',
                          base_kwargs={'distribution': 'normal', 'std': 0.5})
    assert isinstance(init, SphericalizeState), "Should return SphericalizeState instance"
    assert isinstance(init.base_initializer, RandomState), \
        "Base should be RandomState instance"

    print("✓ get_initializer works correctly")


def test_device_placement():
    """Test that states are created on correct device."""
    print("Testing device placement...")
    init = ZeroState()
    state = init(num_nodes=5, channels=8, device='cpu')
    assert state.device.type == 'cpu', f"Expected CPU device, got {state.device}"
    print("✓ Device placement works correctly")


def test_dtype():
    """Test that states have correct dtype."""
    print("Testing dtype...")
    init = ZeroState()
    state = init(num_nodes=5, channels=8, dtype=torch.float64)
    assert state.dtype == torch.float64, f"Expected float64, got {state.dtype}"
    print("✓ Dtype handling works correctly")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Running GNCA Initializers Tests")
    print("=" * 60)
    print()

    try:
        test_zero_state()
        test_one_state()
        test_random_state()
        test_middle_one_state()
        test_multi_point_state()
        test_sphericalize_state()
        test_constant_state()
        test_get_initializer()
        test_device_placement()
        test_dtype()

        print()
        print("=" * 60)
        print("✓ All tests passed!")
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
