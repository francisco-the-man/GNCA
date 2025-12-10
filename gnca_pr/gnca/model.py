import torch
import torch.nn as nn
from typing import Optional
from .conv import GNCAConv

class GNCAModel(nn.Module):
    r"""The full Graph Neural Cellular Automata model architecture.

    This model implements the three-part update rule described in 
    Appendix A.2 of the paper:
    
    1. Pre-processing: :math:`h_i \leftarrow \text{MLP}_{\text{pre}}(s_i)`
    2. Message Passing: :math:`h_i \leftarrow h_i \, \| \, \sum_{j \in \mathcal{N}(i)} \text{ReLU}(\mathbf{W} h_j + \mathbf{b})`
    3. Post-processing: :math:`s_i' \leftarrow \text{MLP}_{\text{post}}(h_i)`

    Args:
        input_channels (int): Dimensionality of the input state space (e.g., 2 for 2D coords).
        output_channels (int): Dimensionality of the output state space.
        hidden_channels (int, optional): Number of hidden units in the MLPs and 
            convolution layer. (default: :obj:`256`)
        output_activation (str, optional): The activation to apply to the final output.
            Options: :obj:`"tanh"` (for continuous bounded states), :obj:`"sigmoid"` 
            (for binary states), or :obj:`None`. (default: :obj:`"tanh"`)
    """
    def __init__(self, 
                 input_channels: int, 
                 output_channels: int,
                 edge_dim: int = 0,
                 hidden_channels: int = 256,
                 output_activation: Optional[str] = 'tanh'):
        super().__init__()

        # 1. Pre-processing MLP
        # Linear -> ReLU -> Linear structure
        self.mlp_pre = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            # Note: We do not apply ReLU at the very end of pre-processing to allow 
            # the embedding to occupy the full latent space before the conv layer.
        )

        # 2. Message Passing Layer (The GNCAConv)
        # Input size is hidden_channels (from mlp_pre).
        # Output size of the convolution message is hidden_channels.
        # The layer itself handles the concatenation, resulting in 2 * hidden_channels.
        self.conv = GNCAConv(hidden_channels, hidden_channels, edge_dim=edge_dim)
        
        # 3. Post-processing MLP
        # Input is 2 * hidden_channels because GNCAConv concatenates [self || neighbor_agg]
        self.mlp_post = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, output_channels)
        )

        self.output_activation = output_activation

    def forward(self, x, edge_index, edge_attr: Optional[torch.Tensor] = None):
        r"""
        Calculates the update delta (i.e. next state for the cellular automaton).

        Args:
            x (Tensor): Node feature matrix :math:`(|\mathcal{V}|, F_{in})`.
            edge_index (LongTensor): Graph edge connectivity :math:`(2, |\mathcal{E}|)`.

        Returns:
            Tensor: The output node features :math:`(|\mathcal{V}|, F_{out})`.
        """
        # 1. Pre-process current state into hidden representation
        h = self.mlp_pre(x)

        # 2. Evolve hidden representation using GNCAConv
        # Returns shape [N, 2 * hidden_channels]
        h = self.conv(h, edge_index, edge_attr=edge_attr)

        # 3. Post-process to get the update/next state
        out = self.mlp_post(h)

        # 4. Apply specific output activation based on the task (Appendix A.2)
        if self.output_activation == 'tanh':
            return torch.tanh(out)
        elif self.output_activation == 'sigmoid':
            return torch.sigmoid(out)
        else:
            return out