import argparse
import json
import logging
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from joblib import load as joblib_load
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from model import get_model
from dataset import EnergyLoadDataset

# Configure basic logging for clear output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_full_timeline(full_df_unscaled: pd.DataFrame, test_preds_mw: np.ndarray, split_info: dict, results_dir: Path):
    """Generates the main plot of the full timeline with colored splits."""
    logging.info("Generating full timeline plot...")
    fig, ax = plt.subplots(figsize=(20, 8))
    
    train_end_dt = pd.to_datetime(split_info['train_end_date'])
    val_end_dt = pd.to_datetime(split_info['val_end_date'])

    # Plot each data split in a different color
    ax.plot(full_df_unscaled.loc[:train_end_dt].index, full_df_unscaled.loc[:train_end_dt]['total_load_mw'], label='Actual Load (Train)', color='dodgerblue', linewidth=1)
    ax.plot(full_df_unscaled.loc[train_end_dt:val_end_dt].index, full_df_unscaled.loc[train_end_dt:val_end_dt]['total_load_mw'], label='Actual Load (Validation)', color='darkorange', linewidth=1)
    ax.plot(full_df_unscaled.loc[val_end_dt:].index, full_df_unscaled.loc[val_end_dt:]['total_load_mw'], label='Actual Load (Test)', color='forestgreen', linewidth=1.5)
    
    test_preds_series = pd.Series(test_preds_mw, index=full_df_unscaled.loc[val_end_dt:].index[:len(test_preds_mw)])
    ax.plot(test_preds_series.index, test_preds_series, label='Predicted Load (Test)', color='red', linestyle='--', linewidth=1.5)
    
    ax.axvline(train_end_dt, color='red', linestyle=':', linewidth=2)
    ax.axvline(val_end_dt, color='red', linestyle=':', linewidth=2)
    
    ax.set_title('Full Timeline: Model Performance', fontsize=18, pad=20)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Energy Load (MW)', fontsize=14)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.tight_layout()
    
    plot_path = results_dir / 'full_timeline_predictions_plot.png'
    plt.savefig(plot_path, dpi=150)
    logging.info(f"Full timeline plot saved to {plot_path}")
    plt.close(fig)

def plot_test_set_zoom(test_actuals_series: pd.Series, test_preds_series: pd.Series, results_dir: Path):
    """Generates a detailed, zoomed-in plot of the test set predictions."""
    logging.info("Generating zoomed-in plot for the test set...")
    fig, ax = plt.subplots(figsize=(20, 8))

    ax.plot(test_actuals_series.index, test_actuals_series, label='Actual Load (Test)', color='forestgreen', linewidth=2)
    ax.plot(test_preds_series.index, test_preds_series, label='Predicted Load (Test)', color='red', linestyle='--', linewidth=2)
    
    ax.set_title('Zoom-In: Test Set Performance', fontsize=18, pad=20)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Energy Load (MW)', fontsize=14)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.tight_layout()
    
    plot_path = results_dir / 'test_set_zoom_plot.png'
    plt.savefig(plot_path, dpi=150)
    logging.info(f"Zoomed-in test set plot saved to {plot_path}")
    plt.close(fig)

def evaluate(config_path: Path, run_dir: Path):
    """
    Loads a trained model from a specific run directory and evaluates its performance.
    """
    logging.info(f"--- Starting Evaluation for run: {run_dir.name} ---")
    
    with open(config_path, 'r') as f:
        config = json.load(f)

    # --- 1. Load Configuration & Define Paths ---
    data_conf, model_conf, train_conf, log_conf = config['data'], config['model'], config['training'], config['logging']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # The path to the model is now inside the specific run directory
    best_model_path = run_dir / log_conf['best_model_name']
    if not best_model_path.exists():
        logging.error(f"Model checkpoint not found at {best_model_path}")
        raise FileNotFoundError(f"Model checkpoint not found at {best_model_path}")

    # --- 2. Load Data and Scalers ---
    logging.info("Loading processed data and scalers...")
    processed_df = pd.read_csv(data_conf['processed_path'], index_col='datetime', parse_dates=True)
    
    # Load the two separate scaler files
    scaler_dir = Path(data_conf['processed_path']).parent
    feature_scaler = joblib_load(scaler_dir / 'feature_scaler.joblib')
    target_scaler = joblib_load(scaler_dir / 'target_scaler.joblib')

    with open(data_conf['split_info_path'], 'r') as f:
        split_info = json.load(f)
    test_df = processed_df.loc[split_info['test_start_date']:split_info['test_end_date']]
    
    test_dataset = EnergyLoadDataset(test_df, train_conf['sequence_length'], target_col='total_load_mw')
    test_loader = DataLoader(test_dataset, batch_size=train_conf['batch_size'], shuffle=False)
    
    # --- 3. Load Trained Model ---
    logging.info(f"Loading best model from {best_model_path}")
    # Input size must be the number of features (total columns - 1)
    input_size = test_df.shape[1] - 1
    model_specific_config = model_conf[model_conf['type']]
    model_specific_config['output_size'] = model_conf['output_size']
    model_specific_config['dropout'] = 0.0 # Set dropout to 0 for evaluation
    
    model = get_model(model_type=model_conf['type'], model_config=model_specific_config, input_size=input_size)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    # --- 4. Make Predictions on Test Set ---
    logging.info("Making predictions on the test set...")
    all_preds_scaled, all_actuals_scaled = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pred_scaled = model(X_batch)
            all_preds_scaled.append(y_pred_scaled.cpu().numpy())
            # We use y_batch directly, which is already the scaled target
            all_actuals_scaled.append(y_batch.cpu().numpy())
            
    # Reshape predictions and actuals to be 1D arrays
    all_preds_scaled = np.concatenate(all_preds_scaled).flatten()

    # --- 5. Inverse Transform Data ---
    logging.info("Inverse transforming predictions and actuals...")
    test_preds_mw = target_scaler.inverse_transform(all_preds_scaled.reshape(-1, 1)).flatten()
    
    # Inverse transform the entire dataframe for plotting
    # Separate features and target, inverse transform them, then combine.
    features_scaled = processed_df.drop(columns='total_load_mw')
    target_scaled = processed_df[['total_load_mw']]
    
    features_unscaled = feature_scaler.inverse_transform(features_scaled)
    target_unscaled = target_scaler.inverse_transform(target_scaled)
    
    full_df_unscaled = pd.DataFrame(features_unscaled, index=processed_df.index, columns=features_scaled.columns)
    full_df_unscaled['total_load_mw'] = target_unscaled
    
    # Prepare data specifically for plotting and metrics
    test_actuals_series = full_df_unscaled.loc[split_info['test_start_date']:]['total_load_mw']
    # Align lengths, as the last few items in test set can't form a full sequence
    test_actuals_series = test_actuals_series.iloc[:len(test_preds_mw)]
    test_preds_series = pd.Series(test_preds_mw, index=test_actuals_series.index)
    
    # --- 6. Calculate and Log Interpretable Metrics ---
    mae_mw = np.mean(np.abs(test_preds_series.values - test_actuals_series.values))
    rmse_mw = np.sqrt(np.mean((test_preds_series.values - test_actuals_series.values)**2))
    
    logging.info("--- Test Set Performance ---")
    logging.info(f"Mean Absolute Error (MAE): {mae_mw:.2f} MW")
    logging.info(f"Root Mean Squared Error (RMSE): {rmse_mw:.2f} MW")
    logging.info("--------------------------")

    # Save metrics to a file inside the run directory
    metrics = {'MAE_MW': mae_mw, 'RMSE_MW': rmse_mw}
    with open(run_dir / 'test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    # --- 7. Visualize Results ---
    # Plots are now saved directly into the run directory
    plot_full_timeline(full_df_unscaled, test_preds_mw, split_info, results_dir=run_dir)
    plot_test_set_zoom(test_actuals_series, test_preds_series, results_dir=run_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained forecasting model.")
    parser.add_argument('--config', type=str, help="Path to the configuration JSON file.", default="config/config.json")
    parser.add_argument('--run_dir', type=str, help="(Optional) Path to a specific run directory to evaluate. If not provided, the latest run will be used.")
    args = parser.parse_args()

    run_dir_to_evaluate = None
    if args.run_dir:
        # Use the directory specified by the user
        run_dir_to_evaluate = Path(args.run_dir)
    else:
        # Automatically find the latest run directory
        with open(args.config, 'r') as f:
            config = json.load(f)
        log_dir = Path(config['logging']['log_dir'])
        
        if not log_dir.exists():
            logging.error(f"Log directory not found at {log_dir}. Please train a model first.")
            exit()
            
        all_runs = [d for d in log_dir.iterdir() if d.is_dir()]
        if not all_runs:
            logging.error(f"No training runs found in {log_dir}. Please train a model first.")
            exit()
            
        latest_run = max(all_runs, key=lambda d: d.stat().st_mtime)
        run_dir_to_evaluate = latest_run

    evaluate(config_path=Path(args.config), run_dir=run_dir_to_evaluate)