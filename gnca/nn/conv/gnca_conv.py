from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor
from typing import Optional, Union


from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor
from typing import Optional, Union


class GNCAConv(MessagePassing):
    """
    Graph Neural Cellular Automata convolution layer.

    Does a single GNCA update step using existing PyG MessagePassing framework.
    1. Compute messages from neighbours (MLP)
    2. Aggregate messages
    3. (Optional) concatenate original state with aggregated messages


    Args:
        in_channels (int): Input feature dimensionality (state size per node).
        out_channels (int): Output feature dimensionality.
        hidden_channels (int): Hidden layer size for message MLP. Default: 128.
        aggr (str): Aggregation scheme within message-passing('add', 'mean', 'max', 'cat'). Default: 'add'.
        persistence (bool): Whether to concatenate input state with aggregated
            messages. If True, output will be [h_i || aggregated_messages].
            Default: True.
        mlp_layers (int): Number of layers in pre-message passing MLP. Default: 1.
        mp_layers (int): Number of message passing layers. Default: 1.
        activation (str): Activation function ('relu', 'tanh', 'elu', 'sigmoid'). Default: 'relu'.
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
        pre_mlp_layers: int = 1,
        mp_layers: int=1,
        post_mlp_layers : int = 1,
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
        self.mp_layers = mp_layers
        self.pre_mlp_layers = pre_mlp_layers # Store for __repr__
        self.post_mlp_layers = post_mlp_layers

        # Activation function for internal layers (applied manually where needed)
        if activation == 'relu':
            self.internal_activation = nn.ReLU()
        elif activation == 'tanh':
            self.internal_activation = nn.Tanh()
        elif activation == 'elu':
            self.internal_activation = nn.ELU()
        elif activation == "sigmoid": # Sigmoid can be used internally too, though less common for hidden layers
            self.internal_activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # 1. Pre-processing MLP (applied to input 'x' before message passing starts)
        # This transforms the input feature `in_channels` to `hidden_channels`.
        preproc_mlp_modules = []
        preproc_mlp_modules.append(nn.Linear(in_channels, hidden_channels, bias=bias))
        if batch_norm:
            preproc_mlp_modules.append(nn.BatchNorm1d(hidden_channels))
        preproc_mlp_modules.append(self.internal_activation) # Always apply activation here
        if dropout > 0:
            preproc_mlp_modules.append(nn.Dropout(dropout))
        self.preproc_mlp = nn.Sequential(*preproc_mlp_modules)

        # 2. Message MLP (used in the `message` method, transforms `x_j`)
        # It takes features of `hidden_channels` (output of preproc_mlp) and outputs `hidden_channels`.
        # This MLP should *not* have a final activation within its Sequential block, as the activation
        # is applied *after* this MLP in the `message` method.
        message_mlp_modules = []
        current_in_dim = hidden_channels
        for i in range(pre_mlp_layers):
            message_mlp_modules.append(nn.Linear(current_in_dim, hidden_channels, bias=bias))
            if batch_norm:
                message_mlp_modules.append(nn.BatchNorm1d(hidden_channels))
            if i < pre_mlp_layers - 1: # Apply internal activation for all but the last linear layer
                message_mlp_modules.append(self.internal_activation)
                if dropout > 0: # Dropout typically applied after activation
                    message_mlp_modules.append(nn.Dropout(dropout))
            elif dropout > 0 and pre_mlp_layers == 1: # Apply dropout if it's the only layer and dropout is needed
                message_mlp_modules.append(nn.Dropout(dropout))

            current_in_dim = hidden_channels

        #Commented out because sometimes we WANT no preprocessing embedding
        # Ensure message_mlp is not empty if pre_mlp_layers is 0 or logic creates an empty list
        #if not message_mlp_modules:
        #    message_mlp_modules.append(nn.Linear(hidden_channels, hidden_channels, bias=bias))
        #    if batch_norm:
        #        message_mlp_modules.append(nn.BatchNorm1d(hidden_channels))

        self.message_mlp = nn.Sequential(*message_mlp_modules)

        # 3. Decoder MLP (final post-processing)
        # Input dimension depends on `persistence`: hidden_channels (if no concat) or 2*hidden_channels (if concat).
        decoder_input_dim = hidden_channels * 2 if persistence else hidden_channels
        decoder_modules = []
        for j in range(self.post_mlp_layers-1):
            decoder_modules.append(nn.Linear(decoder_input_dim, decoder_input_dim, bias=bias))
            if batch_norm:
                decoder_modules.append(nn.BatchNorm1d(out_channels))
            message_mlp_modules.append(self.internal_activation)
        if post_mlp_layers > 0:
            decoder_modules.append(nn.Linear(decoder_input_dim,out_channels,bias=bias))
        
        # Final activation for the decoder
        if out_channels == 1: # Typically Sigmoid for binary output
            decoder_modules.append(nn.Sigmoid())
        else: # For other types of output, use the internal activation or no activation
            decoder_modules.append(self.internal_activation)

        self.decoder = nn.Sequential(*decoder_modules)

        self.reset_parameters()

    def reset_parameters(self):
        # Iterate over all sequential modules to reset parameters
        for module in self.preproc_mlp:
            if hasattr(module, 'reset_parameters'): module.reset_parameters()
        for module in self.message_mlp:
            if hasattr(module, 'reset_parameters'): module.reset_parameters()
        for module in self.decoder:
            if hasattr(module, 'reset_parameters'): module.reset_parameters()

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
        3. (Optional) concatenates original state (self) with aggregated messages

        Returns updated node features [num_nodes, out_channels] or
        [num_nodes, in_channels + out_channels] if persistence=True.
        """

        # Pre-process input features. x: [N, in_channels] -> [N, hidden_channels]
        x = self.preproc_mlp(x)

        # Store original pre-processed state if persistence is enabled
        x_orig = x.clone() if self.persistence else None

        # Perform message passing for mp_layers iterations
        # The `propagate` method calls message, aggregate, and update.
        # `aggr_out` will be [N, hidden_channels] after aggregation.
        aggr_out = x # Initialize with the pre-processed features for the first iteration
        for _ in range(self.mp_layers):
            aggr_out = self.propagate(edge_index, x=aggr_out, edge_weight=edge_weight)

         # Concatenate original state with aggregated messages if persistence is True
        if self.persistence:
            # x_orig is [N, hidden_channels], aggr_out is [N, hidden_channels]
            combined_features = torch.cat([x_orig, aggr_out], dim=-1) # [N, 2 * hidden_channels]
        else:
            combined_features = aggr_out # [N, hidden_channels]

        # Final post-processing using the decoder MLP
        out = self.decoder(combined_features) # [N, out_channels]

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor = None) -> Tensor:
        """
        Construct messages from neighbor nodes.
        i.e. activation(message_mlp(x_j))
        Note: x_j is tensor with neighbor node feats, which are already
              transformed by preproc_mlp, so they are `hidden_channels` dim.

        Returns tensor of messages to be aggregated [num_edges, hidden_channels]
        """
        # Apply message MLP (linear layers + batchnorm, no final activation within MLP itself)
        msg_transformed = self.message_mlp(x_j)
        # Apply the internal activation as per the paper's formulation for GNCA
        msg = self.internal_activation(msg_transformed)

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

        Returns tensor of aggregated messages [num_nodes, hidden_channels]
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
            f'persistence={self.persistence}, '
            f'preprocessing_mlp_layers={self.pre_mlp_layers}, '
            f'mp_layers={self.mp_layers}), '
            f'post-processing_mlp_layers={self.post_mlp_layers}'
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