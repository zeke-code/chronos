import pandas as pd
import numpy as np
import json
import jsonschema
import torch
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
    split_info_from_model = checkpoint.get('split_info', None)
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
print("--- Preparing Test Data ---")

# Load the full dataset
df_full = pd.read_csv(config['data']['processed_path'], index_col='Date', parse_dates=True)

# Load split information
try:
    with open(config['data']['split_info_path'], 'r') as f:
        split_info = json.load(f)
    print("Loaded data split information")
except FileNotFoundError:
    # Fallback to split info from model if available
    if split_info_from_model:
        split_info = split_info_from_model
        print("Using split information from model checkpoint")
    else:
        print("Error: Data split information not found. Run data_preprocessing.py first.")
        exit()

# Extract test data based on split indices
test_start = split_info['test_start']
test_end = split_info['test_end']
df_test = df_full.iloc[test_start:test_end]

print(f"Test data: {len(df_test)} samples ({split_info['test_date_range'][0]} to {split_info['test_date_range'][1]})")

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

print(f"Test sequences created: {len(X_test_seq)}")

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

print(f"\n--- Test Set Evaluation Metrics for {model_type.upper()} ---")
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

# Create three plots
fig, axes = plt.subplots(3, 1, figsize=(18, 16))

# Plot 1: Time series comparison (first 500 points)
plot_subset = results_df.head(500)  # Plot the first ~5 days
axes[0].plot(plot_subset.index, plot_subset['Actual Load'], label='Actual Load', color='blue', linewidth=2)
axes[0].plot(plot_subset.index, plot_subset['Predicted Load'], label=f'Predicted Load ({model_type.upper()})', 
         color='red', linestyle='--', alpha=0.8)
axes[0].set_title(f'Load Forecast vs. Actual Load - {model_type.upper()} Model (Test Set - First 500 Points)', fontsize=16)
axes[0].set_xlabel('Date', fontsize=12)
axes[0].set_ylabel('Load (MW)', fontsize=12)
axes[0].legend(fontsize=12)
axes[0].grid(True)

# Add text box with metrics
metrics_text = f'MAE: {mae:.2f} MW\nRMSE: {rmse:.2f} MW\nMAPE: {mape:.2f}%'
axes[0].text(0.02, 0.95, metrics_text, transform=axes[0].transAxes, fontsize=11,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 2: Scatter plot of predictions vs actuals
axes[1].scatter(actuals, predictions, alpha=0.5, s=10)
axes[1].plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Load (MW)', fontsize=12)
axes[1].set_ylabel('Predicted Load (MW)', fontsize=12)
axes[1].set_title(f'Predicted vs Actual Load - {model_type.upper()} Model (Test Set)', fontsize=16)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

r2 = r2_score(actuals, predictions)
axes[1].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[1].transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# Plot 3: Error distribution
errors = actuals.flatten() - predictions.flatten()
axes[2].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='green')
axes[2].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
axes[2].set_title(f'Prediction Error Distribution - {model_type.upper()} Model (Test Set)', fontsize=16)
axes[2].set_xlabel('Error (MW)', fontsize=12)
axes[2].set_ylabel('Frequency', fontsize=12)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# Add error statistics text
error_stats = f'Mean Error: {np.mean(errors):.2f} MW\nStd Error: {np.std(errors):.2f} MW\nMin Error: {np.min(errors):.2f} MW\nMax Error: {np.max(errors):.2f} MW'
axes[2].text(0.02, 0.95, error_stats, transform=axes[2].transAxes, fontsize=11,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save the figure
figure_dir = Path('results/figures') / model_type
figure_dir.mkdir(parents=True, exist_ok=True)
figure_path = figure_dir / f'test_evaluation_plot_{model_type}.png'
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
ax.set_title(f'Test Set Prediction Performance by Hour - {model_type.upper()} Model', fontsize=14)
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
hourly_figure_path = figure_dir / f'test_hourly_performance_{model_type}.png'
plt.savefig(hourly_figure_path, dpi=300)
print(f"Hourly performance plot saved to {hourly_figure_path}")
plt.show()

# Save evaluation results to CSV
results_summary = {
    'Model Type': model_type.upper(),
    'Dataset': 'Test Set',
    'MAE (MW)': mae,
    'RMSE (MW)': rmse,
    'MAPE (%)': mape,
    'R² Score': r2,
    'Total Parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
    'Test Samples': len(predictions),
    'Test Date Range': f"{split_info['test_date_range'][0]} to {split_info['test_date_range'][1]}"
}

results_file = figure_dir / f'test_evaluation_results_{model_type}.csv'
pd.DataFrame([results_summary]).to_csv(results_file, index=False)
print(f"\nTest evaluation results saved to {results_file}")

# Print final summary
print("\n" + "="*60)
print("EVALUATION COMPLETE")
print("="*60)
print(f"Model: {model_type.upper()}")
print(f"Evaluated on: {len(predictions)} test samples (completely unseen during training)")
print(f"Performance: MAE={mae:.2f} MW, RMSE={rmse:.2f} MW, MAPE={mape:.2f}%")
print("="*60)