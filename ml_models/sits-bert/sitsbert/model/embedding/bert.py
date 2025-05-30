"""
BERT Embedding module for time series data.

This module provides implementations of embedding components used in BERT-like
models for time series data. The BERTEmbedding class combines input embedding
and positional encoding to create embeddings that preserve both feature
information and temporal position information.

Examples
--------
>>> import torch
>>> from sitsbert.model.embedding.bert import BERTEmbedding
>>>
>>> # Create a BERT embedding layer
>>> bert_embed = BERTEmbedding(num_features=10, embedding_dim=512, dropout=0.1)
>>>
>>> # Sample input sequence (batch_size=2, seq_length=5, features=10)
>>> input_seq = torch.randn(2, 5, 10)
>>>
>>> # Sample day-of-year sequence (batch_size=2, seq_length=5)
>>> doy_seq = torch.tensor([[32, 64, 96, 128, 160], [50, 100, 150, 200, 250]])
>>>
>>> # Get embeddings
>>> embeddings = bert_embed(input_seq, doy_seq)
>>> print(embeddings.shape)
>>> torch.Size([2, 5, 1024])
"""

# Initial imports.
import torch
import torch.nn as nn

from .position import PositionalEncoding


class BERTEmbedding(nn.Module):
    """
    BERT Embedding combining input features and positional encoding.

    This embedding layer combines:

    1. InputEmbedding: Projects the input features to embedding dimension
       through a fully connected layer.
    2. PositionalEncoding: Adds positional information using sine and cosine
       functions.

    The final embedding is a concatenation of both embeddings.

    Attributes
    ----------
    input : nn.Linear
        Linear projection layer for input features.
    position : PositionalEncoding
        Positional encoding module.
    dropout : nn.Dropout
        Dropout layer applied to final embeddings.
    embed_size : int
        Size of the feature embedding dimension.
    """

    def __init__(
        self, num_features: int, embedding_dim: int, dropout: float = 0.1
    ) -> None:
        """
        Initialize the BERTEmbedding module.

        Parameters
        ----------
        num_features : int
            Number of input features (satellite bands).
        embedding_dim : int
            Embedding size for the feature embedding.
        dropout : float, optional
            Dropout rate. Default is 0.1.
        """

        # Call the parent constructor.
        super().__init__()

        # Initialize the input projection layer.
        self.input = nn.Linear(in_features=num_features, out_features=embedding_dim)

        # Initialize the positional encoding layer.
        self.position = PositionalEncoding(d_model=embedding_dim, max_len=366)

        # Initialize the dropout layer.
        self.dropout = nn.Dropout(p=dropout)

        # Store the embedding size for later use.
        self.embed_size: int = embedding_dim

    def forward(
        self, input_sequence: torch.Tensor, doy_sequence: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply embedding to input features and day-of-year sequences.

        Parameters
        ----------
        input_sequence : torch.Tensor
            Input feature sequence with shape (batch_size, seq_length,
            num_features).
        doy_sequence : torch.Tensor
            Day-of-year sequence with shape (batch_size, seq_length).

        Returns
        -------
        torch.Tensor
            Combined embeddings with shape (batch_size, seq_length,
            embedding_dim*2), where the first half contains feature embeddings
            and the second half contains positional encodings.
        """

        # input_sequence is a 3D tensor (batch_size, seq_length, num_features).
        batch_size: int = input_sequence.size(0)
        seq_length: int = input_sequence.size(1)

        # Project input features to embedding dimension.
        obs_embed: torch.Tensor = self.input(input_sequence)

        # Create space for both embeddings (feature + positional).
        x: torch.Tensor = obs_embed.repeat(1, 1, 2)

        # Add positional embeddings for each sequence in batch.
        for i in range(batch_size):
            x[i, :, self.embed_size :] = self.position(doy_sequence[i, :])

        return self.dropout(x)
