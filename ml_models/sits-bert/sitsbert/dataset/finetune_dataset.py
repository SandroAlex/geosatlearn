# Initial setup.
import numpy as np
import torch
from torch.utils.data import Dataset

class FineTuneDataset(Dataset):
    """
    Fine-tuning dataset for SITS-BERT.

    This dataset loads time series data from a file, processes it into
    sequences of features, masks, and labels suitable for fine-tuning
    the SITS-BERT model.
    """

    def __init__(
            self, 
            file_path: str, 
            num_features: int, 
            seq_len: int, 
            bands_scale_factor: float = 1 / 10000
        ):
        """
        Initialize the FinetuneDataset.

        Parameters
        ----------
        file_path : str
            Path to the fine-tuning data file.
        num_features : int
            Number of input features per time step.
        seq_len : int
            Padded sequence length for each sample.
        bands_scale_factor : float, optional
            Scale factor for reflectance values (default is 1/10000).
        """
        
        # Main parameters.
        self.seq_len: int = seq_len
        self.dimension: int = num_features
        self.bands_scale_factor: float = bands_scale_factor

        with open(file_path, "r") as ifile:
            self.Data: list[str] = ifile.readlines()
            self.TS_num: int = len(self.Data)

    def __len__(self) -> int:
        """
        Return the number of time series samples in the dataset.

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return self.TS_num

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Parameters
        ----------
        item : int
            Index of the sample to retrieve.

        Returns
        -------
        dict of str to torch.Tensor
        Dictionary containing:
        - 'bert_input': torch.Tensor, shape (seq_len, num_features)
            BOA reflectances, normalized.
        - 'bert_mask': torch.Tensor, shape (seq_len,)
            Mask indicating valid time steps.
        - 'class_label': torch.Tensor, shape (1,)
            Class label for the sample.
        - 'time': torch.Tensor, shape (seq_len,)
            Day of year for each time step.
        """

        # Read the line corresponding to the sample index.
        line: str = self.Data[item]

        # line[-1] == '\n' should be discarded.
        line_data: list[str] = line[:-1].split(",")
        line_data: list[float] = list(map(float, line_data))
        line_data: np.array = np.array(line_data, dtype=float)

        # Extract class label and convert to numpy array.
        class_label: np.array = np.array([line_data[-1]], dtype=int)

        # Extract time series data and reshape it.
        # shape: (number of times steps, number of features + 1).
        # + 1 for the day of year.
        ts: np.array = np.reshape(line_data[:-1], (self.dimension + 1, -1)).T
        
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

        # Fill results. 
        # The mask is 1 for valid time steps and 0 for padding.
        output: dict = {
            "bert_input": ts_origin,
            "bert_mask": bert_mask,
            "class_label": class_label,
            "time": doy,
        }

        # Convert numpy arrays to PyTorch tensors.
        return {key: torch.from_numpy(value) for key, value in output.items()}
