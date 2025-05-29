"""
single.py
=========

This module defines the `Attention` class, which implements the scaled dot
product attention mechanism. Attention is a key component of Transformer
models, enabling them to focus on relevant parts of the input sequence.

The `Attention` class computes attention scores based on the query, key, and
value tensors, applies masking if provided, and optionally applies dropout to
the attention probabilities.

Classes
-------
Attention : torch.nn.Module
    Implements scaled dot product attention.

Usage
-----
Example usage of the `Attention` class:

>>> import torch
>>> attention = Attention()
>>> query = torch.randn(32, 64, 128)  # Batch of 32, sequence length 64, hidden size 128
>>> key = torch.randn(32, 64, 128)
>>> value = torch.randn(32, 64, 128)
>>> mask = torch.ones(32, 1, 64)  # Mask indicating valid positions
>>> output, attn_probs = attention(query, key, value, mask)
>>> print(output.shape)  # (32, 64, 128)
>>> print(attn_probs.shape)  # (32, 64, 64)
"""

# Initial imports.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Implements scaled dot product attention.

    This class computes attention scores based on the query, key, and value
    tensors, applies masking if provided, and optionally applies dropout to
    the attention probabilities.
    """

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
        dropout: nn.Module = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform scaled dot product attention.

        Parameters
        ----------
        query : torch.Tensor
            Query tensor of shape (batch_size, seq_len, hidden_size).
        key : torch.Tensor
            Key tensor of shape (batch_size, seq_len, hidden_size).
        value : torch.Tensor
            Value tensor of shape (batch_size, seq_len, hidden_size).
        mask : torch.Tensor, optional
            Mask tensor of shape (batch_size, 1, seq_len), where 1 indicates
            valid positions and 0 indicates masked positions (default is None).
        dropout : nn.Module, optional
            Dropout module to apply to attention probabilities (default is None).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - Output tensor of shape (batch_size, seq_len, hidden_size), which
              contains the weighted sum of values.
            - Attention probabilities tensor of shape (batch_size, seq_len, seq_len).
        """

        # Compute scaled dot product attention scores.
        scores: torch.Tensor = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            query.size(-1)
        )

        # If a mask is provided, apply it to the scores. Put very large negative values
        # to the positions that are masked, so that they will not contribute to the softmax.
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to the scores to get attention probabilities.
        # The softmax is applied along the last dimension (seq_len) to normalize the scores.
        # This ensures that the attention weights sum to 1 for each query.
        p_attn: torch.Tensor = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
