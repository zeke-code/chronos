import argparse
import json
import logging
import pandas as pd
import numpy as np
import holidays
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from joblib import dump

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPreprocessor:
    """
    A class to preprocess time-series energy load data for LSTM models.
    """
    def __init__(self, config_path: Path):
        """
        Initializes the preprocessor with configuration.
        
        Args:
            config_path (Path): Path to the JSON configuration file.
        """
        logging.info("Initializing DataPreprocessor.")
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.data_config = self.config.get('data', {})
        self.raw_data = None
        self.processed_data = None
        # Initialize two separate scalers for features and target ---
        # This is a more robust approach that prevents any data leakage between
        # the scale of the features and the scale of the target.
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()

    def load_data(self, raw_data_dir: Path) -> 'DataPreprocessor':
        """
        Loads and merges all raw CSV data from a specified directory.
        
        Args:
            raw_data_dir (Path): The directory path containing the raw CSV files.
        
        Returns:
            self: The instance of the DataPreprocessor.
        """
        # This method is correct and remains unchanged.
        logging.info(f"Loading and merging raw data from directory: {raw_data_dir}...")
        
        csv_files = list(raw_data_dir.glob('*.csv'))
        if not csv_files:
            logging.error(f"No CSV files found in directory {raw_data_dir}.")
            raise FileNotFoundError(f"No CSV files found in {raw_data_dir}")

        all_dfs = []
        try:
            for file_path in sorted(csv_files):
                logging.info(f"Reading file: {file_path.name}")
                df = pd.read_csv(
                    file_path,
                    usecols=[0, 1, 2, 3],
                    on_bad_lines='skip',
                    engine='python',
                    skipfooter=2
                )
                all_dfs.append(df)
            
            self.raw_data = pd.concat(all_dfs, ignore_index=True)
            logging.info(f"Successfully loaded and merged {len(all_dfs)} files. Total rows: {len(self.raw_data)}")
        except Exception as e:
            logging.error(f"An error occurred while loading data: {e}")
            raise
        return self

    def clean_and_prepare_data(self) -> 'DataPreprocessor':
        """
        Cleans the raw data by renaming columns, converting data types,
        and handling the datetime index.
        
        Returns:
            self: The instance of the DataPreprocessor.
        """
        # This method is correct and remains unchanged.
        if self.raw_data is None:
            logging.error("Raw data is not loaded. Please call load_data() first.")
            return self

        logging.info("Cleaning and preparing data...")
        
        self.raw_data.columns = [
            'datetime', 'total_load_mw', 'forecast_total_load_mw', 'bidding_zone'
        ]

        logging.info("Dropping 'bidding_zone' and 'forecast_total_load_mw' columns.")
        self.raw_data = self.raw_data.drop(columns=['bidding_zone', 'forecast_total_load_mw'])
        
        self.raw_data['datetime'] = pd.to_datetime(self.raw_data['datetime'], format='%d/%m/%Y %H:%M:%S')

        for col in ['total_load_mw']:
            self.raw_data[col] = self.raw_data[col].str.replace(',', '.', regex=False).astype(float)
            
        self.raw_data = self.raw_data.set_index('datetime').sort_index()
        self.raw_data.interpolate(method='time', inplace=True)
        
        self.processed_data = self.raw_data.copy()
        logging.info("Data cleaning and preparation complete.")
        return self
        
    def create_features(self) -> 'DataPreprocessor':
        """
        Engineers time-based, cyclical, lag, and rolling features from the datetime index.
        
        Returns:
            self: The instance of the DataPreprocessor.
        """
        if self.processed_data is None:
            logging.error("Data not cleaned. Please run create_features() first.")
            return self

        logging.info("Starting advanced feature engineering...")
        df = self.processed_data
        target_col = 'total_load_mw'
        
        # 1. Calendar-based features
        logging.info("Creating calendar features (weekend, holiday)...")
        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # --- NEW: Add holiday feature --- (Your logic was good, just integrated here)
        it_holidays = holidays.country_holidays('IT', years=df.index.year.unique())
        df['is_holiday'] = df.index.to_series().dt.date.isin(it_holidays).astype(int)
        
        # 2. Cyclical time-based features
        logging.info("Creating cyclical features for time components...")
        df['hour'] = df.index.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # We no longer need the original integer-based time columns
        df.drop(columns=['hour', 'day_of_week'], inplace=True)
        
        # 3. Lag features (Provides historical context)
        logging.info("Creating lag features (past load values)...")
        # 96 intervals = 1 day (24h * 4), 672 intervals = 1 week (96 * 7)
        lag_steps = [96, 672] 
        for lag in lag_steps:
            df[f'load_lag_{lag}'] = df[target_col].shift(lag)

        # 4. Rolling Window features (Provides recent trend context)
        logging.info("Creating rolling window features (recent trend and volatility)...")
        window_size = 96 # 24-hour window
        df[f'load_rolling_mean_{window_size}'] = df[target_col].rolling(window=window_size).mean()
        df[f'load_rolling_std_{window_size}'] = df[target_col].rolling(window=window_size).std()
        
        # 5. Drop rows with NaN values created by lag/rolling features
        logging.info(f"Shape before dropping NaNs: {df.shape}")
        df.dropna(inplace=True)
        logging.info(f"Shape after dropping NaNs: {df.shape}")

        self.processed_data = df
        logging.info("Advanced feature engineering complete.")
        return self

    def split_and_scale_data(self) -> 'DataPreprocessor':
        """
        Splits data chronologically into training, validation, and test sets
        and scales the features and target based on the training set.
        
        Returns:
            self: The instance of the DataPreprocessor.
        """
        if self.processed_data is None:
            logging.error("Processed data not available. Please run previous steps.")
            return self
        
        logging.info("Splitting and scaling data with separate scalers...")
        
        # Define which columns are features and which is the target
        target_col = 'total_load_mw'
        feature_cols = [col for col in self.processed_data.columns if col != target_col]
        
        n = len(self.processed_data)
        train_end_idx = int(n * self.data_config['train_ratio'])
        val_end_idx = train_end_idx + int(n * self.data_config['val_ratio'])

        train_df = self.processed_data.iloc[:train_end_idx]
        val_df = self.processed_data.iloc[train_end_idx:val_end_idx]
        test_df = self.processed_data.iloc[val_end_idx:]

        logging.info(f"Data split into: Train ({len(train_df)}), Validation ({len(val_df)}), Test ({len(test_df)}) samples.")

        # Fit the feature scaler ONLY on the training data's features
        self.feature_scaler.fit(train_df[feature_cols])
        # Transform the features of the entire dataset
        self.processed_data[feature_cols] = self.feature_scaler.transform(self.processed_data[feature_cols])

        # Fit the target scaler ONLY on the training data's target
        # Scaler expects a 2D array, so we reshape with [[]]
        self.target_scaler.fit(train_df[[target_col]])
        # Transform the target of the entire dataset
        self.processed_data[target_col] = self.target_scaler.transform(self.processed_data[[target_col]])

        logging.info("Data scaling complete.")

        # Save both scalers to disk for later use during inference
        output_dir = Path(self.data_config['processed_path']).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        feature_scaler_path = output_dir / 'feature_scaler.joblib'
        target_scaler_path = output_dir / 'target_scaler.joblib'
        dump(self.feature_scaler, feature_scaler_path)
        dump(self.target_scaler, target_scaler_path)
        logging.info(f"Feature and Target scalers saved to {output_dir}")
        
        # This part for saving the split info remains the same
        split_info = {
            "train_start_date": str(train_df.index.min()), "train_end_date": str(train_df.index.max()),
            "val_start_date": str(val_df.index.min()), "val_end_date": str(val_df.index.max()),
            "test_start_date": str(test_df.index.min()), "test_end_date": str(test_df.index.max()),
            "train_size": len(train_df), "val_size": len(val_df), "test_size": len(test_df)
        }
        split_info_path = Path(self.data_config['split_info_path'])
        split_info_path.parent.mkdir(parents=True, exist_ok=True)
        with open(split_info_path, 'w') as f:
            json.dump(split_info, f, indent=4)
            
        return self

    def save_processed_data(self):
        """
        Saves the fully processed and scaled DataFrame to a CSV file.
        """
        # This method is correct and remains unchanged.
        if self.processed_data is None:
            logging.error("No processed data to save.")
            return

        processed_path = Path(self.data_config['processed_path'])
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        self.processed_data.to_csv(processed_path)
        logging.info(f"Processed data successfully saved to {processed_path}.")


def main():
    """Main function to run the data preprocessing pipeline."""
    # This function is correct and remains unchanged.
    parser = argparse.ArgumentParser(description="Preprocess energy load data for LSTM models.")
    parser.add_argument('--raw_data_dir', type=str, help="Path to the directory containing raw data CSV files.", default="data/raw/")
    parser.add_argument('--config_path', type=str, help="Path to the configuration JSON file.", default="config/config.json")
    
    args = parser.parse_args()

    preprocessor = DataPreprocessor(config_path=Path(args.config_path))
    preprocessor.load_data(raw_data_dir=Path(args.raw_data_dir)) \
                .clean_and_prepare_data() \
                .create_features() \
                .split_and_scale_data() \
                .save_processed_data()
    
    logging.info("Data preprocessing pipeline finished successfully.")

if __name__ == '__main__':
    main()