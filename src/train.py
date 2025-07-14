import pandas as pd
import json
import jsonschema
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import joblib

# Import our custom modules
from dataset import create_sequences, TimeSeriesDataset
from model import get_model

# --- 1. SETUP ---
print("--- Starting Training Script ---")

# Load configuration and validate against schema
try:
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    with open('config/config_schema.json', 'r') as f:
        schema = json.load(f)

    jsonschema.validate(instance=config, schema=schema)
    print("Configuration file is valid.")
except FileNotFoundError as e:
    print(f"Error: Configuration or schema file not found. {e}")
    exit()
except jsonschema.exceptions.ValidationError as e:
    print(f"Error: Configuration file is invalid. {e.message}")
    exit()
except json.JSONDecodeError as e:
    print(f"Error: Could not decode JSON from config or schema file. {e}")
    exit()

# Get model type
model_type = config['model']['type']
print(f"Model type selected: {model_type.upper()}")

# Create necessary directories from config
# Add model type to directory names for organization
checkpoint_dir = Path(config['logging']['checkpoint_dir']) / model_type
log_dir = Path(config['logging']['log_dir']) / model_type
checkpoint_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)

# Set up device (use GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize TensorBoard writer for logging
writer = SummaryWriter(log_dir=log_dir)

# --- 2. DATA LOADING AND PREPARATION ---
print("--- Loading and Preparing Data ---")
df = pd.read_csv(config['data']['processed_path'], index_col='Date', parse_dates=True)

# Load split information
try:
    with open(config['data']['split_info_path'], 'r') as f:
        split_info = json.load(f)
    print("Loaded data split information")
except FileNotFoundError:
    print("Error: Data split information file not found. Run data_preprocessing.py first.")
    exit()

# Extract train and validation data based on split indices
train_start, train_end = split_info['train_start'], split_info['train_end']
val_start, val_end = split_info['val_start'], split_info['val_end']

# Split the data
df_train = df.iloc[train_start:train_end]
df_val = df.iloc[val_start:val_end]

print(f"Train data: {len(df_train)} samples ({split_info['train_date_range'][0]} to {split_info['train_date_range'][1]})")
print(f"Validation data: {len(df_val)} samples ({split_info['val_date_range'][0]} to {split_info['val_date_range'][1]})")

# Separate features and target
target_col = 'actual_load'
X_train = df_train.drop(columns=[target_col])
y_train = df_train[[target_col]]
X_val = df_val.drop(columns=[target_col])
y_val = df_val[[target_col]]

# Update the input_size in our config based on the number of features
input_size = X_train.shape[1]
print(f"Number of features (input_size): {input_size}")

# Scale the data
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)

joblib.dump(scaler_X, checkpoint_dir / 'scaler_X.gz')
joblib.dump(scaler_y, checkpoint_dir / 'scaler_y.gz')
print("Scalers fitted and saved.")

seq_length = config['training']['sequence_length']
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled.flatten(), seq_length)
X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled.flatten(), seq_length)

print(f"Training sequences: {len(X_train_seq)}")
print(f"Validation sequences: {len(X_val_seq)}")

# Create datasets and dataloaders
train_dataset = TimeSeriesDataset(X_train_seq, y_train_seq)
val_dataset = TimeSeriesDataset(X_val_seq, y_val_seq)

train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
print("DataLoaders created.")

# --- 3. MODEL, LOSS, AND OPTIMIZER ---
print("--- Initializing Model ---")

# Get model-specific config
model_config = config['model'].copy()
if model_type in config['model']:
    # Merge model-specific parameters
    model_specific_config = config['model'][model_type]
    model_config.update(model_specific_config)

# Create model using factory function
model = get_model(model_type, model_config, input_size).to(device)

# Calculate total parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}")

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

# --- 4. RESUME FROM CHECKPOINT LOGIC ---
start_epoch = 0
best_val_loss = float('inf')
patience_counter = 0
resume_checkpoint_path = checkpoint_dir / f"latest_checkpoint_{model_type}.pth"

if resume_checkpoint_path.exists():
    print(f"--- Resuming training from checkpoint: {resume_checkpoint_path} ---")
    checkpoint = torch.load(resume_checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    patience_counter = checkpoint.get('patience_counter', 0)
    
    print(f"Resumed from Epoch {start_epoch}. Best validation loss so far: {best_val_loss:.6f}")
    if patience_counter > 0:
        print(f"Early stopping patience counter: {patience_counter}")
else:
    print("--- Starting training from scratch ---")

print(model)

# --- 5. TRAINING LOOP ---
print(f"--- Starting Training from Epoch {start_epoch + 1} ---")
epochs = config['training']['epochs']

# Get early stopping configuration
early_stopping_config = config['training'].get('early_stopping', {})
early_stopping_enabled = early_stopping_config.get('enabled', False)
patience = early_stopping_config.get('patience', 10)
min_delta = early_stopping_config.get('min_delta', 0.0001)
verbose = early_stopping_config.get('verbose', True)

if early_stopping_enabled:
    print(f"Early stopping enabled with patience={patience}, min_delta={min_delta}")

for epoch in range(start_epoch, epochs):
    # Training Phase
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_pred = model(X_batch)
        loss = loss_function(y_pred, y_batch.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
    avg_train_loss = train_loss / len(train_loader.dataset)

    # Validation Phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = loss_function(y_pred, y_batch.unsqueeze(1))
            val_loss += loss.item() * X_batch.size(0)
    avg_val_loss = val_loss / len(val_loader.dataset)

    # Logging
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
    writer.add_scalar('Loss/Train', avg_train_loss, epoch)
    writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
    writer.flush()

    # --- CHECKPOINTING ---
    # Save the latest state for resuming
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'patience_counter': patience_counter,
        'model_type': model_type,
        'model_config': model_config,
        'split_info': split_info  # Save split info in checkpoint
    }, resume_checkpoint_path)
    
    # Save the best model for evaluation
    if avg_val_loss < best_val_loss - min_delta:
        best_val_loss = avg_val_loss
        best_model_path = checkpoint_dir / f"best_model_{model_type}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': model_type,
            'model_config': model_config,
            'input_size': input_size,
            'best_val_loss': best_val_loss,
            'split_info': split_info  # Save split info in best model too
        }, best_model_path)
        print(f"Validation loss decreased. Saving best model to {best_model_path}")
        patience_counter = 0
    else:
        patience_counter += 1
        if early_stopping_enabled and verbose:
            print(f"Early stopping patience: {patience_counter}/{patience}")
    
    # Check early stopping condition
    if early_stopping_enabled and patience_counter >= patience:
        print(f"\n--- Early stopping triggered at epoch {epoch+1} ---")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"No improvement for {patience} epochs.")
        break

writer.close()
print("--- Training Finished ---")
print(f"Model type: {model_type.upper()}")
print(f"Best validation loss: {best_val_loss:.6f}")