"""
Positional Encoding module for Transformer-based models.

This module implements positional encoding as described in "Attention Is All
You Need" (Vaswani et al., 2017). Positional encodings are added to the input
embeddings to provide the model with information about the sequence order,
as transformer models have no inherent way of processing sequential information.

The positions are encoded using sine and cosine functions of different
frequencies.

Examples
--------
>>> import torch
>>> from sitsbert.model.embedding.position import PositionalEncoding
>>> 
>>> # Create positional encoding for a model with dimension 512
>>> pos_encoder = PositionalEncoding(d_model=512)
>>> 
>>> # Get positional encodings for days 1, 10, and 100
>>> days = torch.tensor([1, 10, 100])
>>> encodings = pos_encoder(days)
>>> print(encodings.shape)
>>> torch.Size([3, 512])
"""

# Initial imports.
import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Positional encoding using sine and cosine functions.

    This class implements positional encodings for transformer models, where
    each position is encoded using sine and cosine functions at different
    frequencies.

    Attributes
    ----------
    pe : torch.Tensor
        Pre-computed positional encoding vectors of shape
        (max_len + 1, d_model). The 0th position is left as zeros for padding
        or special tokens.
    """

    def __init__(self, d_model: int, max_len: int = 366) -> None:
        """
        Initialize the PositionalEncoding module.

        Parameters
        ----------
        d_model : int
            The dimension of the model / embeddings.
        max_len : int, optional
            The maximum length of the expected sequence. Defaults to 366, which
            is suitable for day-of-year encoding.
        """

        # Call the parent constructor.
        super().__init__()

        # Compute the positional encodings once in log space.
        pe: torch.Tensor = torch.zeros(max_len + 1, d_model).float()

        # Freeze the positional encoding parameters to prevent them from being
        # updated during training.
        pe.require_grad = False

        # Vector of positions (0 to max_len - 1).
        position: torch.Tensor = torch.arange(0, max_len).float().unsqueeze(1)

        # Compute the division term for sine and cosine functions.
        div_term: torch.Tensor = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        # Apply sine and cosine functions to even and odd indices.
        pe[1:, 0::2] = torch.sin(position * div_term)  # [max_len, d_model/2]
        pe[1:, 1::2] = torch.cos(position * div_term)  # [max_len, d_model/2]

        # Register the positional encodings as a buffer to ensure they are
        # not treated as model parameters, but still part of the module's state.
        self.register_buffer("pe", pe)

    def forward(self, doy: torch.Tensor) -> torch.Tensor:
        """
        Retrieve the positional encodings for the given day indices.

        Parameters
        ----------
        doy : torch.Tensor
            Tensor containing day-of-year indices for which to retrieve
            positional encodings.

        Returns
        -------
        torch.Tensor
            Tensor containing the positional encodings for the given day
            indices. Shape will be (doy.shape[0], d_model).
        """

        return self.pe[doy, :]
