# Initial setup.
import numpy as np
import torch
from torch.utils.data import Dataset

class FinetuneDataset(Dataset):
    """
    Fine-tuning dataset for SITS-BERT.

    This dataset loads time series data from a file, processes it into
    sequences of features, masks, and labels suitable for fine-tuning
    the SITS-BERT model.
    """

    def __init__(self, file_path: str, feature_num: int, seq_len: int):
        """
        Initialize the FinetuneDataset.

        Parameters
        ----------
        file_path : str
            Path to the fine-tuning data file.
        feature_num : int
            Number of input features per time step.
        seq_len : int
            Padded sequence length for each sample.
        """
        
        # Main parameters.
        self.seq_len: int = seq_len
        self.dimension: int = feature_num

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
                - 'bert_input': torch.Tensor, shape (seq_len, feature_num)
                  BOA reflectances, normalized.
                - 'bert_mask': torch.Tensor, shape (seq_len,)
                  Mask indicating valid time steps.
                - 'class_label': torch.Tensor, shape (1,)
                  Class label for the sample.
                - 'time': torch.Tensor, shape (seq_len,)
                  Day of year for each time step.
        """
        line = self.Data[item]

        # line[-1] == '\n' should be discarded
        line_data = line[:-1].split(",")
        line_data = list(map(float, line_data))
        line_data = np.array(line_data, dtype=float)

        class_label = np.array([line_data[-1]], dtype=int)

        ts = np.reshape(line_data[:-1], (self.dimension + 1, -1)).T
        ts_length = ts.shape[0]

        bert_mask = np.zeros((self.seq_len,), dtype=int)
        bert_mask[:ts_length] = 1

        # BOA reflectances
        ts_origin = np.zeros((self.seq_len, self.dimension))
        ts_origin[:ts_length, :] = ts[:, :-1] / 10000.0

        # day of year
        doy = np.zeros((self.seq_len,), dtype=int)
        doy[:ts_length] = np.squeeze(ts[:, -1])

        output = {
            "bert_input": ts_origin,
            "bert_mask": bert_mask,
            "class_label": class_label,
            "time": doy,
        }

        return {key: torch.from_numpy(value) for key, value in output.items()}
