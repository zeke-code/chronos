import numpy as np
import torch
from torch.utils.data import Dataset

def create_sequences(features, target, seq_length):
    """
    Transforms a time series dataset into sequences.

    Args:
        features (np.array): Array of input features.
        target (np.array): Array of target values.
        seq_length (int): The length of the sequences.

    Returns:
        Tuple[np.array, np.array]: A tuple containing the sequences (X) and their corresponding targets (y).
    """
    xs, ys = [], []
    for i in range(len(features) - seq_length):
        # A sequence is 'seq_length' consecutive feature sets
        x = features[i:(i + seq_length)]
        # The target is the value immediately after the sequence
        y = target[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


class TimeSeriesDataset(Dataset):
    """
    Custom PyTorch Dataset for time series data.
    """
    def __init__(self, X, y):
        # Convert to PyTorch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        # The number of samples is the number of sequences we have
        return len(self.X)

    def __getitem__(self, idx):
        # Returns one sample: a sequence and its corresponding target
        return self.X[idx], self.y[idx]