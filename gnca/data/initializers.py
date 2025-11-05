"""
Different ways to initialize the state of the nodes in the GNCA.\
  IE this is like the 'seed state' for a normal CA
Includes these options:
        'zeros': ZeroState,
        'ones': OneState,
        'random': RandomState,
        'middle_one': MiddleOneState,
        'constant': ConstantState,
        'multi_point': MultiPointState,
        'feature_based': FeatureBasedState,

    and then sphericalize norms any of those to unit length
"""

from abc import ABC, abstractmethod
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor


class GNCAInitializer(ABC):
    """
    Generate initial node states for cellular automata evolution.
    Returns: Tensor w/ initial node states of shape [num_nodes, channels].
    """

    @abstractmethod
    def __call__(
        self,
        num_nodes: int, # number of nodes in the graph
        channels: int, # number of channels (i.e. features) per node
        device: Optional[Union[str, torch.device]] = None, 
        dtype: torch.dtype = torch.float32
    ) -> Tensor:
        
        pass


class ZeroState(GNCAInitializer):
    """
    Init all node states to zero.
    """

    def __call__(
        self,
        num_nodes: int,
        channels: int,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32
    ) -> Tensor:
        return torch.zeros(num_nodes, channels, device=device, dtype=dtype)


class OneState(GNCAInitializer):
    """
    Init all node states to one.
    """

    def __call__(
        self,
        num_nodes: int,
        channels: int,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32
    ) -> Tensor:
        return torch.ones(num_nodes, channels, device=device, dtype=dtype)


class RandomState(GNCAInitializer):
    """
    Init node states with random vals from a distribution (normal-DEFAULT or uniform)
    Returns tensor of shape [num_nodes, channels] with random vals from the distribution!
    """

    def __init__(
        self,
        distribution: str = 'normal', # either 'normal' or 'uniform'.
        mean: float = 0.0,  # mean for normal distribution, default is 0
        std: float = 1.0,  # std dev for normal distribution, default is 1  
        low: float = 0.0,  # lower bound for uniform distribution, default is 0
        high: float = 1.0  # upper bound for uniform distribution, default is 1
    ):
        if distribution not in ['normal', 'uniform']:
            raise ValueError("distribution has to be normal or uniform")

        self.distribution = distribution
        self.mean = mean
        self.std = std
        self.low = low
        self.high = high

    def __call__(
        self,
        num_nodes: int,
        channels: int,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32
    ) -> Tensor:
        if self.distribution == 'normal':
            state = torch.randn(num_nodes, channels, device=device, dtype=dtype)
            state = state * self.std + self.mean
        elif self.distribution == 'uniform':
            state = torch.rand(num_nodes, channels, device=device, dtype=dtype)
            state = state * (self.high - self.low) + self.low

        return state


class MiddleOneState(GNCAInitializer):
    """
    Init only the middle node to one, with all other nodes at zero.
    ^ can use this to look at pattern formation and wave propagation bc 
    it's a single point activation pattern initially
    Note: Grattola et al use this... it's in repo under modules/init_state.py
    """

    def __call__(
        self,
        num_nodes: int,
        channels: int,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32
    ) -> Tensor:
        state = torch.zeros(num_nodes, channels, device=device, dtype=dtype)
        state[num_nodes // 2] = 1.0 # middle node set to 1
        return state


class MultiPointState(GNCAInitializer):
    """
    Init specific nodes to one, with all other nodes at zero.
    ^ Grattola et al DON'T use this one but it might be useful
    for multiple seed pts and looking at pattern completion?
    """

    def __init__(self, indices: Union[list, Tensor]):
        # use specified node indices to set to 1(list or tensor)
        if isinstance(indices, list): # if list, convert to tensor
            indices = torch.tensor(indices)
        self.indices = indices

    def __call__(
        self,
        num_nodes: int,
        channels: int,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32
    ) -> Tensor:
        state = torch.zeros(num_nodes, channels, device=device, dtype=dtype)
        indices = self.indices.to(device)
        state[indices] = 1.0 # set specified nodes to 1
        return state


class SphericalizeState(GNCAInitializer):
    """
    Normalize state vectors to unit length (L2 norm).
    So we can normalize length of any other initializers to unit length
    In Grattola et al's implementation (repo): modules/init_state.py


    Args:
        base_initializer (GNCAInitializer): The base initializer to wrap.
        eps (float): Small constant for numerical stability. Default: 1e-8.

    Example:
        >>> base_init = RandomState(distribution='normal', std=2.0)
        >>> init = SphericalizeState(base_init)
        >>> state = init(num_nodes=10, channels=16)
        >>> norms = torch.norm(state, p=2, dim=-1)
        >>> torch.allclose(norms, torch.ones(10), atol=1e-5)
        True
    """

    def __init__(self, base_initializer: GNCAInitializer, eps: float = 1e-8):
        self.base_initializer = base_initializer # the actual setting we want to norm
        self.eps = eps # pytorch has this in functional.normalize() idk why we need it

    def __call__(
        self,
        num_nodes: int,
        channels: int,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32
    ) -> Tensor:
        """Generate and normalize state to unit vectors."""
        state = self.base_initializer(num_nodes, channels, device, dtype)
        # L2 norm along the last dim
        return F.normalize(state, p=2, dim=-1, eps=self.eps)


class FeatureBasedState(GNCAInitializer):
    """
    Init state from existing node features in a PyG Data object (like a graph)
    NOT in Grattola et al BUT could be very cool to use for starting from an
    existing graph w/ features and having GNCA evolve based on that.
    We could use molecular graphs or maybe ecology ones!

    Args:
        feature_key (str): Key to access features in the Data object. Default: 'x'.
        padding_channels (int): Number of additional channels to pad. Default: 0.
        padding_mode (str): How to initialize padding. Either 'zeros' or 'random'. Default: 'zeros'.

    Initialize state from existing node features in a PyG Data object.

    This is useful when you want to start the cellular automaton with
    meaningful node features (e.g., node attributes, embeddings) rather
    than random or zero initialization. Optionally pads with zeros or
    random values to match the desired channel count.

    Args:
        feature_key (str): Key to access features in the Data object. Default: 'x'.
        padding_channels (int): Number of additional channels to pad. Default: 0.
        padding_mode (str): How to initialize padding. Either 'zeros' or 'random'. Default: 'zeros'.

    Example:
        >>> from torch_geometric.data import Data
        >>> data = Data(x=torch.randn(10, 8))
        >>> init = FeatureBasedState(padding_channels=8, padding_mode='zeros')
        >>> state = init.from_data(data)
        >>> state.shape
        torch.Size([10, 16])
    """

    def __init__(
        self,
        feature_key: str = 'x',
        padding_channels: int = 0,
        padding_mode: str = 'zeros'
    ):
        if padding_mode not in ['zeros', 'random']:
            raise ValueError("padding_mode must be 'zeros' or 'random'")

        self.feature_key = feature_key # key for features in the data object (default is 'x')
        self.padding_channels = padding_channels # number of additional channels to pad
        self.padding_mode = padding_mode # how to initialize that padding (zeros or random)

    def __call__(
        self,
        num_nodes: int,
        channels: int,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32
    ) -> Tensor:
        # make tensor with correct shape
        if self.padding_mode == 'zeros':
            return torch.zeros(num_nodes, channels, device=device, dtype=dtype)
        else:  # random
            return torch.randn(num_nodes, channels, device=device, dtype=dtype) * 0.1

    def from_data(self, data, padding_channels: Optional[int] = None) -> Tensor:
        # use data object to get features and number of nodes
        features = data[self.feature_key]
        num_nodes, feature_dim = features.shape

        pad_channels = padding_channels if padding_channels is not None else self.padding_channels

        if pad_channels == 0:
            return features

        # Create padding
        if self.padding_mode == 'zeros':
            padding = torch.zeros(num_nodes, pad_channels,
                                 device=features.device, dtype=features.dtype)
        else:  # random
            padding = torch.randn(num_nodes, pad_channels,
                                 device=features.device, dtype=features.dtype) * 0.1

        # Concatenate features and padding then output final tensor
        state = torch.cat([features, padding], dim=-1)
        return state


class ConstantState(GNCAInitializer):
    """
    Init all node states to a constant value.
    Grattola et al don't use this one but it might be useful
    for testing symmetry breaking in the automaton?
    """

    def __init__(self, value: float = 0.5):
        self.value = value # constant

    def __call__(
        self,
        num_nodes: int,
        channels: int,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32
    ) -> Tensor:
        return torch.full((num_nodes, channels), self.value, device=device, dtype=dtype)


def get_initializer(name: str, **kwargs) -> GNCAInitializer:
    """
    Get an initializer from its name (must add here as we add initializers)
    """
    initializers = {
        'zeros': ZeroState,
        'ones': OneState,
        'random': RandomState,
        'middle_one': MiddleOneState,
        'constant': ConstantState,
        'multi_point': MultiPointState,
        'feature_based': FeatureBasedState,
    }

    if name == 'sphericalize':
        base_name = kwargs.pop('base', 'random')
        # pass extra args to appropriate initializer
        base_init = get_initializer(base_name, **kwargs.pop('base_kwargs', {}))
        return SphericalizeState(base_init, **kwargs)

    if name not in initializers:
        raise ValueError(
            f"Unknown initializer '{name}'. Available: {list(initializers.keys())}"
        )

    return initializers[name](**kwargs)
