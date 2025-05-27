"""
pretrain_dataset.py
====================

This module defines the `PreTrainDataset` class, which is a PyTorch Dataset
designed for pretraining the SITS-BERT model. The dataset processes time
series data into sequences of features, masks, and labels suitable for
self-supervised learning tasks.

The dataset expects input data in a CSV format, where each row corresponds
to a single time series sample. Each sample contains:
- Band reflectance values for multiple time steps.
- Day of year for each time step.

The `PreTrainDataset` class handles:
- Loading and parsing the input data file.
- Scaling band reflectance values using a configurable scale factor.
- Padding or truncating time series to a fixed sequence length.
- Randomly masking time steps and adding noise for self-supervised learning.
- Generating masks to indicate valid time steps and masked positions.
- Returning data in a format compatible with the SITS-BERT model.

Classes
-------
PreTrainDataset : torch.utils.data.Dataset
    A dataset class for pretraining the SITS-BERT model.

Usage
-----
Example usage of the `PreTrainDataset` class:

>>> from pretrain_dataset import PreTrainDataset
>>> dataset = PreTrainDataset(
...     file_path="data/pretrain.csv",
...     num_features=10,
...     seq_len=64,
...     bands_scale_factor=1/10000,
...     probability_for_masking=0.15,
...     positive_noise_amplitude=0.5
... )
>>> sample = dataset[0]
>>> print(sample["bert_input"].shape)  # (64, 10)
>>> print(sample["bert_target"].shape)  # (64, 10)
>>> print(sample["bert_mask"].shape)  # (64,)
>>> print(sample["loss_mask"].shape)  # (64,)
>>> print(sample["time"].shape)  # (64,)

"""

# Initial imports.
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class PreTrainDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        num_features: int,
        seq_len: int,
        bands_scale_factor: float = 1.0 / 10000.0,
        probability_for_masking: float = 0.15,
        positive_noise_amplitude: float = 0.5,
    ) -> None:
        """
        Initialize the PreTrainDataset.

        Parameters
        ----------
        file_path : str
            Path to the pretrain data file.
        num_features : int
            Number of input features per time step.
        seq_len : int
            Padded sequence length for each sample.
        bands_scale_factor : float, optional
            Scale factor for reflectance values (default is 1 / 10000).
        probability_for_masking : float, optional
            Probability of masking a time step in the sequence (default is 0.15).
        positive_noise_amplitude : float, optional
            Amplitude of noise added to the time series (default is 0.5).
        """

        # Initialize parameters.
        self.seq_len: int = seq_len
        self.dimension: int = num_features
        self.bands_scale_factor: float = bands_scale_factor
        self.probability_for_masking: float = probability_for_masking
        self.positive_noise_amplitude: float = positive_noise_amplitude

        # Read into memory.
        with open(file_path, "r") as ifile:
            self.Data: list[str] = ifile.readlines()
            self.TS_num: list[str] = len(self.Data)
            print(">>> Loading data successful ...")

    def __len__(self) -> int:
        return self.TS_num

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.

        Parameters
        ----------
        item : int
            Index of the item to retrieve.
        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary containing:
            - 'bert_input': torch.Tensor, shape (seq_len, num_features)
                BOA reflectances, normalized.
            - 'bert_target': torch.Tensor, shape (seq_len, num_features)
                Original BOA reflectances, used for training.
            - 'bert_mask': torch.Tensor, shape (seq_len,)
                Mask indicating valid time steps in the sequence.
            - 'loss_mask': torch.Tensor, shape (seq_len,)
                Mask indicating which time steps were modified (for loss calculation).
            - 'time': torch.Tensor, shape (seq_len,)
                Day of year for each time step in the sequence.
        """

        # Read data instance.
        line: str = self.Data[item]

        # Line[-1] == "\n" should be discarded.
        line_data: list[str] = line[:-1].split(",")
        line_data: list[float] = list(map(float, line_data))
        line_data: np.array = np.array(line_data, dtype=float)

        # Extract time series data and reshape it.
        # shape: (number of times steps, number of features + 1).
        # + 1 for the day of year.
        ts: np.array = np.reshape(line_data, (self.dimension + 1, -1)).T

        # Number of time steps.
        ts_length: int = ts.shape[0]

        # Pad the time series data to the specified sequence length.
        bert_mask: np.array = np.zeros((self.seq_len,), dtype=int)
        bert_mask[:ts_length] = 1

        # BOA reflectances scaled.
        ts_origin: np.array = np.zeros((self.seq_len, self.dimension))
        ts_origin[:ts_length, :] = self.bands_scale_factor * ts[:, :-1]

        # Day of year.
        doy: np.array = np.zeros((self.seq_len,), dtype=int)
        doy[:ts_length] = np.squeeze(ts[:, -1])

        # Randomly add noise to some observations.
        ts_masking, mask = self.random_masking(ts=ts_origin, ts_length=ts_length)

        # Fill the output dictionary.
        output: dict = {
            "bert_input": ts_masking,
            "bert_target": ts_origin,
            "bert_mask": bert_mask,
            "loss_mask": mask,
            "time": doy,
        }

        return {key: torch.from_numpy(value) for key, value in output.items()}

    def random_masking(
        self, ts: np.ndarray, ts_length: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Randomly mask time steps in the time series and add noise.

        Parameters
        ----------
        ts : np.ndarray
            Time series data with shape (seq_len, dimension).
        ts_length : int
            Length of the time series (number of valid time steps).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - ts_masking: np.ndarray, time series with noise added.
            - mask: np.ndarray, mask indicating which time steps were modified.
        """

        # Create a copy of the time series for masking.
        ts_masking = ts.copy()

        # Initialize the mask.
        mask: np.array = np.zeros((self.seq_len,), dtype=int)

        # Loop through each time step in the time series.
        for i in range(ts_length):

            # From doctrings: random() -> x in the interval [0, 1).
            prob: float = random.random()

            # If prob is less than selected cut, apply masking.
            if prob < self.probability_for_masking:

                # Rescale probaility to the range [0, 1].
                prob_reescaled: float = prob / self.probability_for_masking

                # Mark the position in the mask.
                mask[i] = 1

                # Negative noise.
                if prob_reescaled < 0.5:
                    ts_masking[i, :] += np.random.uniform(
                        low=-self.positive_noise_amplitude,
                        high=0,
                        size=(self.dimension,),
                    )

                # Positive noise.
                else:
                    ts_masking[i, :] += np.random.uniform(
                        low=0,
                        high=self.positive_noise_amplitude,
                        size=(self.dimension,),
                    )

        return ts_masking, mask
