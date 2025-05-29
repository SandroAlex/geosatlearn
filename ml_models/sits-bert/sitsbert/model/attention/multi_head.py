"""
multi_head.py
=============

This module defines the `MultiHeadedAttention` class, which implements the
multi-headed attention mechanism used in Transformer models. Multi-headed
attention allows the model to jointly attend to information from different
representation subspaces at different positions.

The `MultiHeadedAttention` class performs:
- Linear projections of the input query, key, and value tensors.
- Scaled dot product attention for each head using the `Attention` class.
- Concatenation of attention outputs from all heads.
- Final linear projection to combine the outputs.

Classes
-------
MultiHeadedAttention : torch.nn.Module
    Implements multi-headed attention mechanism.

Usage
-----
Example usage of the `MultiHeadedAttention` class:

>>> import torch
>>> from multi_head import MultiHeadedAttention
>>> mha = MultiHeadedAttention(h=8, d_model=128, dropout=0.1)
>>> query = torch.randn(32, 64, 128)  # Batch of 32, sequence length 64, hidden size 128
>>> key = torch.randn(32, 64, 128)
>>> value = torch.randn(32, 64, 128)
>>> mask = torch.ones(32, 1, 64)  # Mask indicating valid positions (TODO: is not working!)
>>> output = mha(query, key, value, mask)
>>> print(output.shape)  # (32, 64, 128)
"""

# Initial imports.
import torch
import torch.nn as nn

from .single import Attention


class MultiHeadedAttention(nn.Module):
    """
    Implements multi-headed attention mechanism.

    This class performs linear projections of the input query, key, and value
    tensors, applies scaled dot product attention for each head, concatenates
    the outputs, and applies a final linear projection.
    """

    def __init__(self, h: int, d_model: int, dropout: float = 0.1) -> None:
        """
        Initialize the MultiHeadedAttention module.

        Parameters
        ----------
        h : int
            Number of attention heads.
        d_model : int
            Dimensionality of the model (hidden size).
        dropout : float, optional
            Dropout rate applied to attention probabilities (default is 0.1).
        """
        
        # Call the parent constructor.
        super().__init__()
        
        # Multihead attention takes it a step further by first mapping 
        # Q, K, and V into different lower-dimensional feature subspaces 
        # via different linear dense layers, and then using the results to 
        # calculate attention.
        assert d_model % h == 0, "d_model must be divisible by the number of heads!"

        # Set subspace dimensionality.
        self.d_k: int = d_model // h
        
        # Set the number of attention heads.
        self.h: int = h

        # Create linear layers for query, key, and value projections (3).
        self.linear_layers: nn.ModuleList = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        
        # Output linear layer to combine the outputs from all heads.
        self.output_linear: nn.Linear = nn.Linear(d_model, d_model)
        
        # Scaled dot product attention mechanism.
        self.attention: Attention = Attention()

        # Dropout layer to apply dropout to attention probabilities.
        self.dropout: nn.Dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Perform multi-headed attention.

        Parameters
        ----------
        query : torch.Tensor
            Query tensor of shape (batch_size, seq_len, d_model).
        key : torch.Tensor
            Key tensor of shape (batch_size, seq_len, d_model).
        value : torch.Tensor
            Value tensor of shape (batch_size, seq_len, d_model).
        mask : torch.Tensor, optional
            Mask tensor of shape (batch_size, 1, seq_len), where 1 indicates
            valid positions and 0 indicates masked positions (default is None).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, d_model), containing
            the combined attention outputs.
        """
        
        # Grab the batch size from the query tensor.
        batch_size: int = query.size(0)

        print(f">>> Query shape: {query.shape}, Key shape: {key.shape}, Value shape: {value.shape}")

        # Perform linear projections for query, key, and value.
        query, key, value = [
            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))
        ]
        
        # Apply attention for each head. Mask indicating valid positions (TODO: is not working!)
        x, _ = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # Concatenate attention outputs and apply final linear projection.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)