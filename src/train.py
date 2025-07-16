import argparse
import json
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import get_model
from dataset import EnergyLoadDataset

# Configure basic logging to the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Trainer:
    """
    A class to encapsulate the training and validation logic with TensorBoard logging.
    """
    def __init__(self, config_path: Path):
        """
        Initializes the Trainer, sets up unique directories, and the TensorBoard writer.
        
        Args:
            config_path (Path): Path to the JSON configuration file.
        """
        logging.info("Initializing Trainer.")
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self._load_config()
        self._setup_logging()
        self._prepare_data()
        self._build_model()
        self._log_initial_setup()

    def _load_config(self):
        """Loads parameters from the config file."""
        self.data_conf = self.config['data']
        self.train_conf = self.config['training']
        self.model_conf = self.config['model']
        self.log_conf = self.config['logging']
        
        self.sequence_length = self.train_conf['sequence_length']
        self.batch_size = self.train_conf['batch_size']
        self.epochs = self.train_conf['epochs']
        self.learning_rate = self.train_conf['learning_rate']
        
        self.early_stopping_conf = self.train_conf.get('early_stopping', {'enabled': False})
        self.target_loss_conf = self.train_conf.get('target_loss', {'enabled': False})

    def _setup_logging(self):
        """Creates a unique directory for the run and initializes SummaryWriter."""
        model_type = self.model_conf['type']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_name = f"{model_type}_{timestamp}"
        
        # All artifacts for this run will be stored here
        self.run_dir = Path(self.log_conf['log_dir']) / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Redefine checkpoint path to be inside the unique run directory
        self.best_model_path = self.run_dir / self.log_conf['best_model_name']
        
        # Initialize TensorBoard SummaryWriter
        self.writer = SummaryWriter(log_dir=self.run_dir)
        logging.info(f"Run artifacts will be saved in: {self.run_dir}")

    def _prepare_data(self):
        """Loads processed data and creates DataLoaders."""
        logging.info("Preparing data...")
        processed_df = pd.read_csv(self.data_conf['processed_path'], index_col='datetime', parse_dates=True)
        
        with open(self.data_conf['split_info_path'], 'r') as f:
            split_info = json.load(f)

        train_df = processed_df.loc[split_info['train_start_date']:split_info['train_end_date']]
        val_df = processed_df.loc[split_info['val_start_date']:split_info['val_end_date']]

        train_dataset = EnergyLoadDataset(train_df, self.sequence_length, target_col='total_load_mw')
        val_dataset = EnergyLoadDataset(val_df, self.sequence_length, target_col='total_load_mw')
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        self.input_size = train_df.shape[1] - 1
        logging.info(f"Data prepared with {self.input_size} features.")

    def _build_model(self):
        """Builds the model, loss function, and optimizer."""
        logging.info(f"Building model of type: {self.model_conf['type']}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Training on device: {self.device}")

        model_specific_config = self.model_conf[self.model_conf['type']]
        model_specific_config['output_size'] = self.model_conf['output_size']
        model_specific_config['dropout'] = self.model_conf['dropout']
        
        self.model = get_model(
            model_type=self.model_conf['type'],
            model_config=model_specific_config,
            input_size=self.input_size
        ).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _log_initial_setup(self):
        """Logs hyperparameters and model architecture to TensorBoard."""
        # Log hyperparameters as text
        hparams_dict = {
            'model_type': self.model_conf['type'],
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'sequence_length': self.sequence_length,
            'dropout': self.model_conf['dropout'],
            'hidden_size': self.model_conf[self.model_conf['type']]['hidden_size'],
            'num_layers': self.model_conf[self.model_conf['type']]['num_layers'],
        }
        # Use a Markdown-formatted table for nice display in TensorBoard
        hparams_md = "## Hyperparameters\n| Parameter | Value |\n|---|---|\n"
        for key, val in hparams_dict.items():
            hparams_md += f"| {key} | {val} |\n"
        self.writer.add_text('Configuration/Hyperparameters', hparams_md)

        # Log the model's text architecture
        model_arch_text = f"<pre>{self.model}</pre>"
        self.writer.add_text('Configuration/Model_Architecture', model_arch_text)

        # Log the model graph
        dummy_input = torch.randn(self.batch_size, self.sequence_length, self.input_size).to(self.device)
        self.writer.add_graph(self.model, dummy_input)
        
    def _train_epoch(self):
        """Performs one epoch of training."""
        self.model.train()
        total_loss = 0
        for X_batch, y_batch in self.train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device).unsqueeze(1)
            
            self.optimizer.zero_grad()
            y_pred = self.model(X_batch)
            loss = self.criterion(y_pred, y_batch)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def _validate_epoch(self):
        """Performs one epoch of validation."""
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device).unsqueeze(1)
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def train(self):
        """Runs the full training and validation loop."""
        logging.info("Starting training...")
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        for epoch in range(self.epochs):
            train_loss = self._train_epoch()
            val_loss = self._validate_epoch()
            
            logging.info(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            # --- TENSORBOARD LOGGING ---
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/validation', val_loss, epoch)
            self.writer.add_scalars('Loss/Combined', {'train': train_loss, 'validation': val_loss}, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_model_path)
                logging.info(f"Validation loss improved. Saving model to {self.best_model_path}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # 1. Early stopping based on patience
            if self.early_stopping_conf['enabled'] and epochs_no_improve >= self.early_stopping_conf['patience']:
                logging.info(f"Early stopping triggered after {epochs_no_improve} epochs with no improvement.")
                break

            # 2. Target loss stopping
            if self.target_loss_conf.get('enabled', False):
                metric = self.target_loss_conf.get('metric', 'validation')
                target = self.target_loss_conf.get('target_val_loss') if metric == 'validation' else self.target_loss_conf.get('target_train_loss')
                current_loss = val_loss if metric == 'validation' else train_loss
                
                if target and current_loss <= target:
                    logging.info(f"Target {metric} loss of {target} reached. Stopping training.")
                    if val_loss < best_val_loss:
                         torch.save(self.model.state_dict(), self.best_model_path)
                    break
                
        logging.info("Training finished.")
        self.writer.close()


def main():
    """Main function to run the training pipeline."""
    parser = argparse.ArgumentParser(description="Train a forecasting model.")
    parser.add_argument('--config', type=str, help="Path to the configuration JSON file.", default="config/config.json")
    
    args = parser.parse_args()

    # Initialize and run the trainer
    trainer = Trainer(config_path=Path(args.config))
    trainer.train()

if __name__ == '__main__':
    main()