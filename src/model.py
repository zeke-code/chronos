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
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )

        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states for the LSTM
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Pass the input through the LSTM layer
        out, _ = self.lstm(x, (h0, c0))

        # We only need the output of the very last time step from the sequence
        out = self.fc(out[:, -1, :])

        return out


class LoadForecastingGRU(nn.Module):
    """
    GRU model for load forecasting (another RNN variant).
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob):
        super(LoadForecastingGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU Layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )

        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state for the GRU
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Pass the input through the GRU layer
        out, _ = self.gru(x, h0)

        # We only need the output of the very last time step from the sequence
        out = self.fc(out[:, -1, :])

        return out


def get_model(model_type, model_config, input_size):
    """
    Factory function to create models based on type.
    
    Args:
        model_type: Type of model ('lstm', 'gru')
        model_config: Model configuration dictionary
        input_size: Number of input features
        
    Returns:
        Initialized model
    """
    if model_type.lower() == 'lstm':
        return LoadForecastingLSTM(
            input_size=input_size,
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            output_size=model_config['output_size'],
            dropout_prob=model_config['dropout']
        )
    
    elif model_type.lower() == 'gru':
        return LoadForecastingGRU(
            input_size=input_size,
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            output_size=model_config['output_size'],
            dropout_prob=model_config['dropout']
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from: lstm, gru")