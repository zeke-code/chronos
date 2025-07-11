import pandas as pd
import numpy as np
import json
import jsonschema
import torch
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import our custom modules
from dataset import create_sequences
from model import get_model

# --- 1. SETUP AND CONFIGURATION ---
print("--- Starting Evaluation Script ---")

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

model_type = config['model']['type']
print(f"Evaluating model type: {model_type.upper()}")

# Set up device (use GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 2. LOAD SAVED ARTIFACTS (MODEL AND SCALERS) ---
print("--- Loading Model and Scalers ---")
checkpoint_dir = Path(config['logging']['checkpoint_dir']) / model_type

# Load the scalers
try:
    scaler_X = joblib.load(checkpoint_dir / 'scaler_X.gz')
    scaler_y = joblib.load(checkpoint_dir / 'scaler_y.gz')
except FileNotFoundError:
    print("Error: Scaler files not found. Make sure you have run the training script first.")
    exit()

# Load the trained model
model_path = checkpoint_dir / f"best_model_{model_type}.pth"
try:
    checkpoint = torch.load(model_path, map_location=device)
    model_state = checkpoint['model_state_dict']
    model_config = checkpoint['model_config']
    input_size = checkpoint['input_size']
    best_val_loss = checkpoint.get('best_val_loss', 'N/A')
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}. Make sure the best model was saved during training.")
    exit()

# Create model using factory function
model = get_model(model_type, model_config, input_size).to(device)
model.load_state_dict(model_state)

# Set the model to evaluation mode
model.eval()
print("Model and scalers loaded successfully.")
print(f"Best validation loss during training: {best_val_loss}")

# --- 3. PREPARE THE TEST DATA ---
print("--- Preparing Test Data (from validation split) ---")
df_full = pd.read_csv(config['data']['processed_path'], index_col='Date', parse_dates=True)
val_split_point = int(len(df_full) * (1 - config['data']['validation_split']))
df_test = df_full[val_split_point:]

# Separate features (X) and target (y)
target_col = 'actual_load'
X_test = df_test.drop(columns=[target_col])
y_test = df_test[[target_col]]

# Scale the test data using the already-fitted scalers
X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test)

# Create sequences
seq_length = config['training']['sequence_length']
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled.flatten(), seq_length)

# Convert to PyTorch tensors
X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32).to(device)

# --- 4. MAKE PREDICTIONS ---
print("--- Making Predictions on Test Data ---")
predictions_scaled = []
batch_size = config['training']['batch_size']

with torch.no_grad():
    # Process in batches for efficiency
    for i in range(0, len(X_test_tensor), batch_size):
        batch = X_test_tensor[i:i+batch_size]
        pred = model(batch)
        predictions_scaled.extend(pred.cpu().numpy().flatten())

# Inverse-transform the predictions to get the actual predicted MW values
predictions = scaler_y.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))
print(f"Generated {len(predictions)} predictions.")

# --- 5. EVALUATE AND VISUALIZE ---
# Align actual values with predictions
actuals = y_test.iloc[seq_length:].values

mae = mean_absolute_error(actuals, predictions)
rmse = np.sqrt(mean_squared_error(actuals, predictions))
mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

print(f"\n--- Evaluation Metrics for {model_type.upper()} ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} MW")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} MW")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f} %")

# Create a results DataFrame for easy plotting
results_df = pd.DataFrame({
    'Actual Load': actuals.flatten(),
    'Predicted Load': predictions.flatten()
}, index=df_test.index[seq_length:])

# Plot the results
print("\n--- Generating Visualization ---")
plt.style.use('seaborn-v0_8-whitegrid')

# Create two plots: one for a subset and one for error distribution
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))

# Plot 1: Time series comparison
plot_subset = results_df.head(500)  # Plot the first ~5 days
ax1.plot(plot_subset.index, plot_subset['Actual Load'], label='Actual Load', color='blue', linewidth=2)
ax1.plot(plot_subset.index, plot_subset['Predicted Load'], label=f'Predicted Load ({model_type.upper()})', 
         color='red', linestyle='--', alpha=0.8)
ax1.set_title(f'Load Forecast vs. Actual Load - {model_type.upper()} Model (Test Set)', fontsize=16)
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Load (MW)', fontsize=12)
ax1.legend(fontsize=12)
ax1.grid(True)

# Plot 2: Error distribution
errors = actuals.flatten() - predictions.flatten()
ax2.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='green')
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax2.set_title(f'Prediction Error Distribution - {model_type.upper()} Model', fontsize=16)
ax2.set_xlabel('Error (MW)', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Add error statistics text
error_stats = f'Mean Error: {np.mean(errors):.2f} MW\nStd Error: {np.std(errors):.2f} MW'
ax2.text(0.02, 0.95, error_stats, transform=ax2.transAxes, fontsize=11,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save the figure
figure_dir = Path('results/figures') / model_type
figure_dir.mkdir(parents=True, exist_ok=True)
figure_path = figure_dir / f'evaluation_plot_{model_type}.png'
plt.savefig(figure_path, dpi=300)
print(f"Evaluation plot saved to {figure_path}")
plt.show()

# Additional analysis: Performance by hour of day
print("\n--- Performance Analysis by Hour of Day ---")
results_df['Hour'] = results_df.index.hour
results_df['Error'] = results_df['Actual Load'] - results_df['Predicted Load']
results_df['Absolute Error'] = np.abs(results_df['Error'])

hourly_mae = results_df.groupby('Hour')['Absolute Error'].mean()
hourly_mape = results_df.groupby('Hour').apply(
    lambda x: np.mean(np.abs(x['Error'] / x['Actual Load'])) * 100
)

# Plot hourly performance
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(hourly_mae.index, hourly_mae.values, alpha=0.7, label='MAE')
ax2 = ax.twinx()
ax2.plot(hourly_mape.index, hourly_mape.values, color='red', marker='o', label='MAPE')

ax.set_xlabel('Hour of Day', fontsize=12)
ax.set_ylabel('Mean Absolute Error (MW)', fontsize=12)
ax2.set_ylabel('Mean Absolute Percentage Error (%)', fontsize=12, color='red')
ax.set_title(f'Prediction Performance by Hour - {model_type.upper()} Model', fontsize=14)
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
hourly_figure_path = figure_dir / f'hourly_performance_{model_type}.png'
plt.savefig(hourly_figure_path, dpi=300)
print(f"Hourly performance plot saved to {hourly_figure_path}")
plt.show()

# Save evaluation results to CSV
results_summary = {
    'Model Type': model_type.upper(),
    'MAE (MW)': mae,
    'RMSE (MW)': rmse,
    'MAPE (%)': mape,
    'Total Parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
    'Test Samples': len(predictions)
}

results_file = figure_dir / f'evaluation_results_{model_type}.csv'
pd.DataFrame([results_summary]).to_csv(results_file, index=False)
print(f"\nEvaluation results saved to {results_file}")