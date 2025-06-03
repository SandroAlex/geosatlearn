"""
Transformer Block implementation for the SITS-BERT model.

This module provides the core TransformerBlock class used in transformer-based
architectures for processing satellite image time series data. The transformer
block combines multi-headed self-attention with feed-forward neural networks,
including residual connections and dropout for regularization.

Examples
--------
>>> import torch
>>> from sitsbert.model.transformer import TransformerBlock
>>>
>>> # Create a transformer block.
>>> hidden_size = 256
>>> attn_heads = 8
>>> feed_forward_hidden = 1024
>>> dropout_rate = 0.1
>>>
>>> transformer = TransformerBlock(
>>>    hidden_size, attn_heads, feed_forward_hidden, dropout_rate
>>> )
>>>
>>> # Process a batch of sequences.
>>> batch_size = 32
>>> seq_length = 24
>>> x = torch.randn(batch_size, seq_length, hidden_size)
>>> mask = torch.ones(batch_size, seq_length)  # No masking.
>>>
>>> output = transformer(x, mask)
>>> print(output.shape)  # Should be [batch_size, seq_length, hidden_size].
"""

import torch
import torch.nn as nn

from .attention import MultiHeadedAttention
from .utils import PositionwiseFeedForward, SublayerConnection


class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention mechanism.

    This block implements a standard transformer architecture component with
    multi-headed self-attention followed by a feed-forward network, with residual
    connections and layer normalization around each sub-layer.
    """

    def __init__(
        self, hidden: int, attn_heads: int, feed_forward_hidden: int, dropout: float
    ) -> None:
        """
        Initialize a transformer block.

        Parameters
        ----------
        hidden : int
            Hidden size of transformer (embedding dimension).
        attn_heads : int
            Number of attention heads.
        feed_forward_hidden : int
            Size of the feed-forward network, usually 4 * hidden_size.
        dropout : float
            Dropout rate.
        """

        # Initialize the parent class.
        super().__init__()

        # See Figure 2 of the original paper.
        self.attention: MultiHeadedAttention = MultiHeadedAttention(
            h=attn_heads, d_model=hidden
        )
        self.feed_forward: PositionwiseFeedForward = PositionwiseFeedForward(
            d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout
        )
        self.input_sublayer: SublayerConnection = SublayerConnection(
            size=hidden, dropout=dropout
        )
        self.output_sublayer: SublayerConnection = SublayerConnection(
            size=hidden, dropout=dropout
        )
        self.dropout: nn.Dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Process input sequence through the transformer block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, seq_length, hidden_size].
        mask : torch.Tensor
            Attention mask of shape [batch_size, seq_length].

        Returns
        -------
        torch.Tensor
            Output tensor of shape [batch_size, seq_length, hidden_size].
        """

        x = self.input_sublayer(
            x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask)
        )
        x = self.output_sublayer(x=x, sublayer=self.feed_forward)

        return self.dropout(x)
