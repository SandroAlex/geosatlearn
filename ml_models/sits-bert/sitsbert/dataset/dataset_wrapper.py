"""
dataset_wrapper.py
===================

This module defines the `DataSetWrapper` class, which provides an interface
for creating PyTorch DataLoaders for training and validation datasets. It
wraps the `PreTrainDataset` class and handles splitting the dataset into
training and validation subsets.

The `DataSetWrapper` class is designed to:
- Load the dataset using the `PreTrainDataset` class.
- Split the dataset into training and validation subsets based on a
  configurable validation size.
- Create PyTorch DataLoaders for both subsets with random sampling.

Classes
-------
DataSetWrapper : object
    A wrapper class for creating training and validation DataLoaders.

Usage
-----
Example usage of the `DataSetWrapper` class:

>>> from dataset_wrapper import DataSetWrapper
>>> wrapper = DataSetWrapper(
...     batch_size=64,
...     valid_size=0.2,
...     data_path="data/pretrain.csv",
...     num_features=10,
...     max_length=128
... )
>>> train_loader, valid_loader = wrapper.get_data_loaders()
>>> for batch in train_loader:
...     print(batch["bert_input"].shape)  # (64, 128, 10)
"""

# Initial imports.
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from .pretrain_dataset import PreTrainDataset

np.random.seed(0)

class DataSetWrapper:
    """
    A wrapper class for creating training and validation DataLoaders.

    This class loads a dataset using the `PreTrainDataset` class, splits it
    into training and validation subsets, and creates PyTorch DataLoaders
    for both subsets.

    Parameters
    ----------
    batch_size : int
        The batch size for the DataLoaders.
    valid_size : float
        The proportion of the dataset to use for validation (0.0 to 1.0).
    data_path : str
        The path to the dataset file.
    num_features : int
        The number of features (bands) in the dataset.
    max_length : int
        The maximum sequence length for each sample.
    """

    def __init__(
        self, 
        batch_size: int, 
        valid_size: float, 
        data_path: str, 
        num_features: int, 
        max_length: int
    ) -> None:
        """
        Initialize the DataSetWrapper.

        Parameters
        ----------
        batch_size : int
            The batch size for the DataLoaders.
        valid_size : float
            The proportion of the dataset to use for validation (0.0 to 1.0).
        data_path : str
            The path to the dataset file.
        num_features : int
            The number of features (bands) in the dataset.
        max_length : int
            The maximum sequence length for each sample.
        """

        # Main attributes.
        self.batch_size: int = batch_size
        self.valid_size: float = valid_size
        self.data_path: str = data_path
        self.num_features: int = num_features
        self.max_length: int = max_length

    def get_data_loaders(self) -> tuple[DataLoader, DataLoader]:
        """
        Create and return training and validation DataLoaders.

        Returns
        -------
        tuple[DataLoader, DataLoader]
            A tuple containing the training DataLoader and validation
            DataLoader.
        """

        # Instance for PreTrainDataset.
        dataset: PreTrainDataset = PreTrainDataset(
            file_path=self.data_path, 
            num_features=self.num_features, 
            seq_len=self.max_length
        )
        
        # Get the training and validation DataLoaders.
        train_loader, valid_loader = self.get_train_validation_data_loaders(dataset=dataset)
        
        return train_loader, valid_loader

    def get_train_validation_data_loaders(
        self, dataset: PreTrainDataset
    ) -> tuple[DataLoader, DataLoader]:
        """
        Split the dataset into training and validation subsets and create
        DataLoaders for both.

        Parameters
        ----------
        dataset : PreTrainDataset
            The dataset to split and create DataLoaders for.

        Returns
        -------
        tuple[DataLoader, DataLoader]
            A tuple containing the training DataLoader and validation
            DataLoader.
        """
        
        # Get the number of training samples and create indices.
        num_train: int = len(dataset)
        indices: list[int] = list(range(num_train))
        
        # Shuffle the indices for random sampling.
        np.random.shuffle(indices)

        # Calculate the split index for validation.
        split: int = int(np.floor(self.valid_size * num_train))
        
        # Print the number of training and validation samples.
        print(f">>> Samples; Total: {num_train}; Training: {num_train - split}; Validation: {split}")
        
        # Split the indices into training and validation sets.
        valid_idx: list[int] = indices[:split]
        train_idx: list[int] = indices[split:]
        
        # Create samplers for training and validation. These samplers will
        # randomly sample from the respective indices.
        train_sampler: SubsetRandomSampler = SubsetRandomSampler(indices=train_idx)
        valid_sampler: SubsetRandomSampler = SubsetRandomSampler(indices=valid_idx)

        # Create DataLoaders for training and validation datasets.
        train_loader: DataLoader = DataLoader(
            dataset=dataset, 
            batch_size=self.batch_size, 
            sampler=train_sampler, 
            drop_last=True
        )
        valid_loader: DataLoader = DataLoader(
            dataset=dataset, 
            batch_size=self.batch_size, 
            sampler=valid_sampler, 
            drop_last=True
        )

        return train_loader, valid_loader