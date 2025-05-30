"""
bert.py
=======

This module defines the `SBERT` class, which implements the SITS-BERT model
based on the Transformer architecture. The model is designed for processing
time series data with spatial and temporal features, leveraging self-attention
mechanisms for feature extraction.

The `SBERT` class includes:
- An embedding layer (`BERTEmbedding`) to encode input features and day-of-year
  information.
- Multiple Transformer blocks (`TransformerBlock`) for self-attention and
  feature extraction.
- A forward method to process input sequences and generate contextualized
  feature representations.

Classes
-------
SBERT : torch.nn.Module
    The SITS-BERT model for time series data.

Usage
-----
Example usage of the `SBERT` class:

>>> from bert import SBERT
>>> model = SBERT(num_features=10, hidden=128, n_layers=6, attn_heads=8, dropout=0.1)
>>> x = torch.randn(32, 64, 10)  # Batch of 32, sequence length 64, 10 features
>>> doy = torch.randint(1, 366, (32, 64))  # Day of year for each time step
>>> mask = torch.ones(32, 64)  # Mask indicating valid time steps
>>> output = model(x, doy, mask)
>>> print(output.shape)  # (32, 64, 128)
"""

# Initial imports.
import torch
import torch.nn as nn

from .embedding import BERTEmbedding
from .transformer import TransformerBlock


class SBERT(nn.Module):
    """
    The SITS-BERT model for time series data.

    This class implements a Transformer-based model for processing time series
    data with spectral and temporal features. It includes an embedding layer
    and multiple Transformer blocks for feature extraction.

    Parameters
    ----------
    num_features : int
        Number of input features per time step.
    hidden : int
        Hidden size of the SITS-BERT model.
    n_layers : int
        Number of Transformer blocks (layers).
    attn_heads : int
        Number of attention heads in each Transformer block.
    dropout : float, optional
        Dropout rate (default is 0.1).
    """

    def __init__(
        self,
        num_features: int,
        hidden: int,
        n_layers: int,
        attn_heads: int,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize the SBERT model.

        Parameters
        ----------
        num_features : int
            Number of input features per time step.
        hidden : int
            Hidden size of the SITS-BERT model.
        n_layers : int
            Number of Transformer blocks (layers).
        attn_heads : int
            Number of attention heads in each Transformer block.
        dropout : float, optional
            Dropout rate (default is 0.1).
        """

        # Initialize the parent class.
        super().__init__()

        # Main attributes.
        self.hidden: int = hidden
        self.n_layers: int = n_layers
        self.attn_heads: int = attn_heads

        self.feed_forward_hidden: int = hidden * 4

        self.embedding: BERTEmbedding = BERTEmbedding(num_features, int(hidden / 2))

        self.transformer_blocks: nn.ModuleList = nn.ModuleList(
            [
                TransformerBlock(hidden, attn_heads, hidden * 4, dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(
        self, x: torch.Tensor, doy: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the SBERT model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, num_features).
        doy : torch.Tensor
            Day-of-year tensor of shape (batch_size, seq_len).
        mask : torch.Tensor
            Mask tensor of shape (batch_size, seq_len), where 1 indicates valid
            time steps and 0 indicates padding.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, hidden), containing
            contextualized feature representations.
        """
        mask: torch.Tensor = (
            (mask > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        )

        x: torch.Tensor = self.embedding(input_sequence=x, doy_sequence=doy)

        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x
