"""
GNCA message-passing layer from Grattola et al, with extensibility
for different message MLP architectures, aggregation, activation etc.

Based on the paper's message-passing formulation:
$\mathbf{h}_i'\in R^{d'_h}$ using:
$$\mathbf{h_i'}=\mathbf{h}_i\|\sum_{j\in N(i)}\text{ReLU}(\mathbf{Wh}_j+\mathbf{b})$$
where $\|$ is concatenation.

Single update:
1. Compute messages from neighbours (MLP)
2. Aggregate messages
3. (Optional) concatenate original state with aggregated messages

"""

from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor


class GNCAConv(MessagePassing):
    """
    Graph Neural Cellular Automata convolution layer.

    Does asingle GNCA update step using existing PyG MessagePassing framework.
    1. Compute messages from neighbours (MLP)
    2. Aggregate messages
    3. (Optional) concatenate original state with aggregated messages


    Args:
        in_channels (int): Input feature dimensionality (state size per node).
        out_channels (int): Output feature dimensionality.
        hidden_channels (int): Hidden layer size for message MLP. Default: 128.
        aggr (str): Aggregation scheme ('add', 'mean', 'max'). Default: 'add'.
        persistence (bool): Whether to concatenate input state with aggregated
            messages. If True, output will be [h_i || aggregated_messages].
            Default: True.
        mlp_layers (int): Number of layers in message MLP. Default: 2.
        activation (str): Activation function ('relu', 'tanh', 'elu'). Default: 'relu'.
        batch_norm (bool): Apply batch normalization in MLP. Default: False.
        dropout (float): Dropout probability in MLP. Default: 0.0.
        bias (bool): Use bias in linear layers. Default: True.

    Example:
        >>> conv = GNCAConv(in_channels=16, out_channels=16)
        >>> x = torch.randn(100, 16)  # 100 nodes, 16 features
        >>> edge_index = torch.randint(0, 100, (2, 500))
        >>> out = conv(x, edge_index)
        >>> out.shape
        torch.Size([100, 32])  # 16 (original) + 16 (aggregated) if persistence=True
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 128,
        aggr: str = 'add',
        persistence: bool = True,
        mlp_layers: int = 2,
        activation: str = 'relu',
        batch_norm: bool = False,
        dropout: float = 0.0,
        bias: bool = True,
        **kwargs
    ):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.persistence = persistence
        self.batch_norm = batch_norm
        self.dropout = dropout

        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        """
        Build message MLP to transform neighbor features h_j
        paper uses ReLU(W h_j + b)
        """
        mlp_layers_list = []

        if mlp_layers == 1:
            # Single layer: in_channels -> out_channels
            mlp_layers_list.append(nn.Linear(in_channels, out_channels, bias=bias))
            if batch_norm:
                mlp_layers_list.append(nn.BatchNorm1d(out_channels))
            mlp_layers_list.append(self.activation)
        else:
            # Multi-layer MLP
            # First layer: in_channels -> hidden_channels
            mlp_layers_list.append(nn.Linear(in_channels, hidden_channels, bias=bias))
            if batch_norm:
                mlp_layers_list.append(nn.BatchNorm1d(hidden_channels))
            mlp_layers_list.append(self.activation)
            if dropout > 0:
                mlp_layers_list.append(nn.Dropout(dropout))

            # Hidden layers: hidden_channels -> hidden_channels
            for _ in range(mlp_layers - 2):
                mlp_layers_list.append(nn.Linear(hidden_channels, hidden_channels, bias=bias))
                if batch_norm:
                    mlp_layers_list.append(nn.BatchNorm1d(hidden_channels))
                mlp_layers_list.append(self.activation)
                if dropout > 0:
                    mlp_layers_list.append(nn.Dropout(dropout))

            # Final layer: hidden_channels -> out_channels
            mlp_layers_list.append(nn.Linear(hidden_channels, out_channels, bias=bias))
            if batch_norm:
                mlp_layers_list.append(nn.BatchNorm1d(out_channels))
            mlp_layers_list.append(self.activation)

        self.message_mlp = nn.Sequential(*mlp_layers_list)

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.message_mlp:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        """
        Forward pass of the GNCA layer, returns updates node features
        1. Computes messages from neighbours (message() method)
        2. Aggregates messages (aggregate() method)
        3. (Optional) concatenates original state  (self) with aggregated messages

        Returns updated node features [num_nodes, out_channels] or
        [num_nodes, in_channels + out_channels] if persistence=True.
        """
        # Store original state if persistence (concat) is enabled
        x_orig = x if self.persistence else None

        # Propagate messages: calls message(), aggregate(), and update()
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        # If persistence, concatenate original state with aggregated messages
        if self.persistence:
            out = torch.cat([x_orig, out], dim=-1)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor = None) -> Tensor:
        """
        Construct messages from neighbor nodes.
        i.e. ReLU(W h_j + b)

        Note: x_j is tensor with neighbor node feats

        Returns tensor of messages to be aggregated [num_edges, out_channels]
        """
        # Apply message MLP: ReLU(W h_j + b)
        msg = self.message_mlp(x_j)

        # CAN ALSO weight messages by edge weights (not in paper)
        if edge_weight is not None:
            msg = edge_weight.view(-1, 1) * msg

        return msg

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None
    ) -> Tensor:
        """
        Aggregate messages from neighbors (uses PyG's built-in aggregation)
        Default is add which is the same as the paper's formulation
        Note: inputs is tensor with messages from neighbors (from prev step)
        and index is based on PyG -> says which node each message is for

        Returns tensor of aggregated messages [num_nodes, out_channels]
        """
        # Use PyG's built-in aggregation (sum, mean, max, etc.) -> default is sum
        return super().aggregate(inputs, index, ptr=ptr, dim_size=dim_size)

    def update(self, aggr_out: Tensor) -> Tensor:
        """
        Update node embeddings after aggregation 
        Doesn't do anything rn bc we're not using update() in the paper:
        In the basic GNCA formulation, we just return the aggregated messages
        Then the concatenation with original state happens in forward()!
        """
        return aggr_out

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'{self.in_channels}, '
            f'{self.out_channels}, '
            f'hidden_channels={self.hidden_channels}, '
            f'aggr={self.aggr}, '
            f'persistence={self.persistence})' # concatenation (optional)
        )


class GNCAConvSimple(MessagePassing):
    """
    Simplified GNCA convolution layer with minimal configuration.

    This is a streamlined version of GNCAConv with fewer hyperparameters,
    closer to the original paper's formulation. Good for quick experiments.

    Args:
        channels (int): Feature dimensionality (same for input and output).
        hidden_channels (int): Hidden layer size. Default: 256.
        aggr (str): Aggregation scheme. Default: 'add'.
        persistence (bool): Enable state persistence. Default: True.

    Example:
        >>> conv = GNCAConvSimple(channels=16)
        >>> x = torch.randn(100, 16)
        >>> edge_index = torch.randint(0, 100, (2, 500))
        >>> out = conv(x, edge_index)
        >>> out.shape
        torch.Size([100, 32])  # 16 + 16 with persistence
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int = 256,
        aggr: str = 'add',
        persistence: bool = True,
        **kwargs
    ):
        super().__init__(aggr=aggr, **kwargs)

        self.channels = channels
        self.hidden_channels = hidden_channels
        self.persistence = persistence

        # Simple MLP: channels -> hidden -> channels
        self.message_mlp = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, channels),
            nn.ReLU()
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Reset learnable parameters."""
        for module in self.message_mlp:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        """Forward pass."""
        x_orig = x if self.persistence else None
        out = self.propagate(edge_index, x=x)

        if self.persistence:
            out = torch.cat([x_orig, out], dim=-1)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        """Construct messages from neighbors."""
        return self.message_mlp(x_j)

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'{self.channels}, '
            f'hidden_channels={self.hidden_channels}, '
            f'persistence={self.persistence})'
        )
