"""Unit tests for GNCA state initializers."""

import pytest
import torch
from torch_geometric.data import Data

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


class TestZeroState:
    """Tests for ZeroState initializer."""

    def test_basic_initialization(self):
        """Test that ZeroState creates all-zero tensors."""
        init = ZeroState()
        state = init(num_nodes=10, channels=16)

        assert state.shape == (10, 16)
        assert torch.all(state == 0.0)

    def test_device_placement(self):
        """Test that state is created on the correct device."""
        init = ZeroState()
        state = init(num_nodes=5, channels=8, device='cpu')

        assert state.device.type == 'cpu'

    def test_dtype(self):
        """Test that state has the correct data type."""
        init = ZeroState()
        state = init(num_nodes=5, channels=8, dtype=torch.float64)

        assert state.dtype == torch.float64


class TestOneState:
    """Tests for OneState initializer."""

    def test_basic_initialization(self):
        """Test that OneState creates all-one tensors."""
        init = OneState()
        state = init(num_nodes=10, channels=16)

        assert state.shape == (10, 16)
        assert torch.all(state == 1.0)

    def test_sum(self):
        """Test that the sum equals num_nodes * channels."""
        init = OneState()
        state = init(num_nodes=10, channels=16)

        assert state.sum() == 160.0


class TestRandomState:
    """Tests for RandomState initializer."""

    def test_normal_distribution(self):
        """Test normal distribution initialization."""
        init = RandomState(distribution='normal', mean=0.0, std=1.0)
        state = init(num_nodes=1000, channels=16)

        # Check shape
        assert state.shape == (1000, 16)

        # Check statistical properties (with tolerance)
        assert abs(state.mean().item()) < 0.1  # Mean should be close to 0
        assert abs(state.std().item() - 1.0) < 0.1  # Std should be close to 1

    def test_uniform_distribution(self):
        """Test uniform distribution initialization."""
        init = RandomState(distribution='uniform', low=-1.0, high=1.0)
        state = init(num_nodes=1000, channels=16)

        # All values should be in range
        assert torch.all(state >= -1.0)
        assert torch.all(state <= 1.0)

    def test_custom_mean_std(self):
        """Test normal distribution with custom mean and std."""
        init = RandomState(distribution='normal', mean=5.0, std=0.1)
        state = init(num_nodes=1000, channels=16)

        # Check that mean is close to 5.0
        assert abs(state.mean().item() - 5.0) < 0.05

    def test_invalid_distribution(self):
        """Test that invalid distribution raises error."""
        with pytest.raises(ValueError, match="Distribution must be"):
            RandomState(distribution='invalid')


class TestMiddleOneState:
    """Tests for MiddleOneState initializer."""

    def test_middle_node_activation(self):
        """Test that only the middle node is activated."""
        init = MiddleOneState()
        state = init(num_nodes=11, channels=16)

        # Middle node (index 5) should be all ones
        assert torch.all(state[5] == 1.0)

        # All other nodes should be zero
        assert torch.all(state[:5] == 0.0)
        assert torch.all(state[6:] == 0.0)

    def test_even_number_of_nodes(self):
        """Test behavior with even number of nodes."""
        init = MiddleOneState()
        state = init(num_nodes=10, channels=16)

        # Middle should be index 5 (10 // 2)
        assert torch.all(state[5] == 1.0)

        # Check sum equals number of channels
        assert state.sum() == 16.0


class TestMultiPointState:
    """Tests for MultiPointState initializer."""

    def test_list_indices(self):
        """Test initialization with list of indices."""
        init = MultiPointState(indices=[0, 5, 9])
        state = init(num_nodes=10, channels=16)

        # Specified nodes should be ones
        assert torch.all(state[0] == 1.0)
        assert torch.all(state[5] == 1.0)
        assert torch.all(state[9] == 1.0)

        # Other nodes should be zero
        assert torch.all(state[1:5] == 0.0)
        assert torch.all(state[6:9] == 0.0)

    def test_tensor_indices(self):
        """Test initialization with tensor indices."""
        indices = torch.tensor([2, 4, 6])
        init = MultiPointState(indices=indices)
        state = init(num_nodes=10, channels=16)

        assert torch.all(state[2] == 1.0)
        assert torch.all(state[4] == 1.0)
        assert torch.all(state[6] == 1.0)

    def test_sum(self):
        """Test that sum equals num_indices * channels."""
        init = MultiPointState(indices=[0, 5, 9])
        state = init(num_nodes=10, channels=16)

        assert state.sum() == 3 * 16  # 3 nodes * 16 channels


class TestSphericalizeState:
    """Tests for SphericalizeState initializer."""

    def test_unit_norm(self):
        """Test that all vectors have unit norm."""
        base_init = RandomState(distribution='normal', std=2.0)
        init = SphericalizeState(base_init)
        state = init(num_nodes=100, channels=16)

        # Compute norms
        norms = torch.norm(state, p=2, dim=-1)

        # All norms should be approximately 1.0
        assert torch.allclose(norms, torch.ones(100), atol=1e-5)

    def test_preserves_shape(self):
        """Test that sphericalization preserves tensor shape."""
        base_init = RandomState()
        init = SphericalizeState(base_init)
        state = init(num_nodes=50, channels=32)

        assert state.shape == (50, 32)

    def test_with_zero_base(self):
        """Test behavior when base initializer produces zeros."""
        base_init = ZeroState()
        init = SphericalizeState(base_init, eps=1e-8)
        state = init(num_nodes=10, channels=16)

        # With eps, should handle zero vectors gracefully
        assert not torch.any(torch.isnan(state))


class TestFeatureBasedState:
    """Tests for FeatureBasedState initializer."""

    def test_no_padding(self):
        """Test initialization without padding."""
        data = Data(x=torch.randn(10, 8))
        init = FeatureBasedState(padding_channels=0)
        state = init.from_data(data)

        assert state.shape == (10, 8)
        assert torch.equal(state, data.x)

    def test_zero_padding(self):
        """Test initialization with zero padding."""
        data = Data(x=torch.randn(10, 8))
        init = FeatureBasedState(padding_channels=8, padding_mode='zeros')
        state = init.from_data(data)

        assert state.shape == (10, 16)
        # First 8 channels should be original features
        assert torch.equal(state[:, :8], data.x)
        # Last 8 channels should be zeros
        assert torch.all(state[:, 8:] == 0.0)

    def test_random_padding(self):
        """Test initialization with random padding."""
        data = Data(x=torch.randn(10, 8))
        init = FeatureBasedState(padding_channels=8, padding_mode='random')
        state = init.from_data(data)

        assert state.shape == (10, 16)
        # First 8 channels should be original features
        assert torch.equal(state[:, :8], data.x)
        # Last 8 channels should not all be zero
        assert not torch.all(state[:, 8:] == 0.0)

    def test_override_padding_channels(self):
        """Test overriding padding_channels in from_data()."""
        data = Data(x=torch.randn(10, 8))
        init = FeatureBasedState(padding_channels=4, padding_mode='zeros')
        state = init.from_data(data, padding_channels=12)

        assert state.shape == (10, 20)  # 8 features + 12 padding

    def test_invalid_padding_mode(self):
        """Test that invalid padding mode raises error."""
        with pytest.raises(ValueError, match="padding_mode must be"):
            FeatureBasedState(padding_mode='invalid')


class TestConstantState:
    """Tests for ConstantState initializer."""

    def test_default_value(self):
        """Test initialization with default constant value."""
        init = ConstantState()
        state = init(num_nodes=10, channels=16)

        assert torch.all(state == 0.5)

    def test_custom_value(self):
        """Test initialization with custom constant value."""
        init = ConstantState(value=0.75)
        state = init(num_nodes=10, channels=16)

        assert torch.all(state == 0.75)

    def test_negative_value(self):
        """Test initialization with negative constant value."""
        init = ConstantState(value=-1.5)
        state = init(num_nodes=5, channels=8)

        assert torch.all(state == -1.5)


class TestGetInitializer:
    """Tests for the get_initializer factory function."""

    def test_zeros(self):
        """Test getting ZeroState initializer."""
        init = get_initializer('zeros')
        assert isinstance(init, ZeroState)

    def test_ones(self):
        """Test getting OneState initializer."""
        init = get_initializer('ones')
        assert isinstance(init, OneState)

    def test_random(self):
        """Test getting RandomState initializer with kwargs."""
        init = get_initializer('random', distribution='uniform', low=-1, high=1)
        assert isinstance(init, RandomState)
        assert init.distribution == 'uniform'

    def test_middle_one(self):
        """Test getting MiddleOneState initializer."""
        init = get_initializer('middle_one')
        assert isinstance(init, MiddleOneState)

    def test_constant(self):
        """Test getting ConstantState initializer with custom value."""
        init = get_initializer('constant', value=0.25)
        assert isinstance(init, ConstantState)
        assert init.value == 0.25

    def test_sphericalize(self):
        """Test getting SphericalizeState initializer."""
        init = get_initializer(
            'sphericalize',
            base='random',
            base_kwargs={'distribution': 'normal', 'std': 0.5}
        )
        assert isinstance(init, SphericalizeState)
        assert isinstance(init.base_initializer, RandomState)

    def test_invalid_name(self):
        """Test that invalid initializer name raises error."""
        with pytest.raises(ValueError, match="Unknown initializer"):
            get_initializer('invalid_name')


class TestIntegration:
    """Integration tests for initializers."""

    def test_chain_multiple_initializers(self):
        """Test using multiple initializers in sequence."""
        # Create a random initializer and sphericalize it
        base = RandomState(distribution='normal', mean=0, std=2.0)
        init = SphericalizeState(base)

        state = init(num_nodes=50, channels=16)

        # Should have correct shape
        assert state.shape == (50, 16)

        # Should have unit norm
        norms = torch.norm(state, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones(50), atol=1e-5)

    def test_reproducibility_with_seed(self):
        """Test that setting seed gives reproducible results."""
        init = RandomState(distribution='normal')

        torch.manual_seed(42)
        state1 = init(num_nodes=10, channels=16)

        torch.manual_seed(42)
        state2 = init(num_nodes=10, channels=16)

        assert torch.equal(state1, state2)

    def test_batch_processing(self):
        """Test that initializers work with different sizes."""
        init = OneState()

        small_state = init(num_nodes=5, channels=8)
        large_state = init(num_nodes=1000, channels=128)

        assert small_state.shape == (5, 8)
        assert large_state.shape == (1000, 128)
        assert torch.all(small_state == 1.0)
        assert torch.all(large_state == 1.0)
