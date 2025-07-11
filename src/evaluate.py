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
from utils import create_sequences
from model import LoadForecastingLSTM

# --- 1. SETUP AND CONFIGURATION ---
print("--- Starting Evaluation Script ---")

# Load configuration and validate against schema
try:
    # Assuming config files are in a 'config' subdirectory
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

# Set up device (use GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 2. LOAD SAVED ARTIFACTS (MODEL AND SCALERS) ---
print("--- Loading Model and Scalers ---")
checkpoint_dir = Path(config['logging']['checkpoint_dir'])

# Load the scalers
try:
    scaler_X = joblib.load(checkpoint_dir / 'scaler_X.gz')
    scaler_y = joblib.load(checkpoint_dir / 'scaler_y.gz')
except FileNotFoundError:
    print("Error: Scaler files not found. Make sure you have run the training script first.")
    exit()

# Load the trained model
# First, instantiate the model with the same architecture as during training
model_params = config['model']

# The input_size was determined during training. We need to recalculate it.
df_full = pd.read_csv(config['data']['processed_path'], index_col='Date', parse_dates=True)
input_size = df_full.shape[1] - 1 # All columns except the target
model_params['input_size'] = input_size

model = LoadForecastingLSTM(
    input_size=model_params['input_size'],
    hidden_size=model_params['hidden_size'],
    num_layers=model_params['num_layers'],
    output_size=model_params['output_size'],
    dropout_prob=model_params['dropout']
).to(device)

# Now, load the saved weights (the "state dictionary")
model_path = checkpoint_dir / config['logging']['best_model_name']
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}. Make sure the best model was saved during training.")
    exit()

# Set the model to evaluation mode. This is crucial as it disables dropout.
model.eval()
print("Model and scalers loaded successfully.")

# --- 3. PREPARE THE TEST DATA ---
# For a fair evaluation, we should use the validation set, which the model
# was not directly trained on (it was only used to check performance).
print("--- Preparing Test Data (from validation split) ---")
val_split_point = int(len(df_full) * (1 - config['data']['validation_split']))
df_test = df_full[val_split_point:]

# Separate features (X) and target (y)
target_col = 'actual_load'
X_test = df_test.drop(columns=[target_col])
y_test = df_test[[target_col]]

# Scale the test data using the *already-fitted* scalers
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
with torch.no_grad(): # We don't need to calculate gradients for inference
    for i in range(len(X_test_tensor)):
        sample = X_test_tensor[i].unsqueeze(0)
        pred = model(sample)
        predictions_scaled.append(pred.item())

# Inverse-transform the predictions to get the actual predicted MW values.
predictions = scaler_y.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))
print(f"Generated {len(predictions)} predictions.")

# --- 5. EVALUATE AND VISUALIZE ---
# Align actual values with predictions.
actuals = y_test.iloc[seq_length:].values

mae = mean_absolute_error(actuals, predictions)
rmse = np.sqrt(mean_squared_error(actuals, predictions))
mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

print("\n--- Evaluation Metrics ---")
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
fig, ax = plt.subplots(figsize=(18, 8))

# We'll plot a subset of the data to keep the plot readable
plot_subset = results_df.head(500) # Plot the first ~5 days

ax.plot(plot_subset.index, plot_subset['Actual Load'], label='Actual Load', color='blue', linewidth=2)
ax.plot(plot_subset.index, plot_subset['Predicted Load'], label='Predicted Load', color='red', linestyle='--', alpha=0.8)

ax.set_title('Load Forecast vs. Actual Load (Test Set)', fontsize=16)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Load (MW)', fontsize=12)
ax.legend(fontsize=12)
ax.grid(True)
plt.tight_layout()

# Save the figure
figure_path = Path('results/figures') / 'evaluation_plot.png'
figure_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(figure_path, dpi=300)
print(f"Evaluation plot saved to {figure_path}")
plt.show()