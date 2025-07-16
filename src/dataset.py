import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class EnergyLoadDataset(Dataset):
    """
    Custom PyTorch Dataset for creating sequences from time-series data.
    """
    def __init__(self, data: pd.DataFrame, sequence_length: int, target_col: str):
        """
        Args:
            data (pd.DataFrame): The dataframe with features and target.
            sequence_length (int): The length of the input sequence.
            target_col (str): The name of the target column to predict.
        """
        self.sequence_length = sequence_length

        # 1. Create a bucket for the answers (target)
        self.target_data = data[target_col].values.astype(np.float32)
        
        # 2. Create a bucket for the clues (features)
        self.feature_data = data.drop(columns=[target_col]).values.astype(np.float32)

    def __len__(self):
        """
        Returns the total number of possible sequences in the dataset.
        """
        return len(self.feature_data) - self.sequence_length

    def __getitem__(self, idx: int):
        """
        Retrieves a single sample (a sequence and its corresponding target) from the dataset.
        
        Args:
            idx (int): The index of the sample.
            
        Returns:
            A tuple of (sequence, target), both as PyTorch tensors.
        """
        # The input sequence (X) starts at the given index and spans 'sequence_length' timesteps.
        sequence = self.feature_data[idx : idx + self.sequence_length]
        
        # The target value (y) is the 'total_load_mw' value from the timestep immediately after the sequence ends.
        target = self.target_data[idx + self.sequence_length]

        return torch.from_numpy(sequence), torch.tensor(target).float()