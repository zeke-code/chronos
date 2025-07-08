import torch
import torch.nn as nn

class LoadForecastingLSTM(nn.Module):
    """
    LSTM model for load forecasting.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob):
        super(LoadForecastingLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM Layer
        # batch_first=True makes the input/output tensors have the batch dimension first,
        # which is more intuitive and works well with PyTorch's DataLoader.
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0 # Dropout is only applied between LSTM layers
        )

        # Fully Connected Layer
        # This layer takes the final LSTM output and maps it to our desired output size (1, for the load value).
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states for the LSTM
        # h0 and c0 are tensors of shape (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Pass the input through the LSTM layer
        # out: contains the output features from the last layer for each time step
        # _: contains the final hidden and cell states (h_n, c_n)
        out, _ = self.lstm(x, (h0, c0))

        # We only need the output of the very last time step from the sequence
        # out[:, -1, :] selects the output for all batches at the last time step
        out = self.fc(out[:, -1, :])

        return out