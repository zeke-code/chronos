import pandas as pd
import numpy as np
import os
import holidays
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt


def load_ang_merge_data(file_paths: List[str]) -> pd.DataFrame:
    """
    Load and merge multiple CSV files
    
    Args:
        file_paths: List of paths to CSV files
        
    Returns:
        Merged DataFrame with all data
    """
    dfs = []
    
    for file_path in file_paths:
        print(f"Loading {file_path}...")
        
        # Skipfooter and engine to ignore garbage lines at the end of our datasets
        df = pd.read_csv(
            file_path,
            parse_dates=['Date'],
            dayfirst=True,
            decimal=',',
            thousands='.',
            skipfooter=2,  # Ignore the last 2 lines of the file as they're garbage
            engine='python' # skipfooter requires the python engine
        )
        
        # Clean column names as we want to rename them
        df.columns = df.columns.str.strip()
        
        # Rename columns for consistency
        column_mapping = {
            'Total Load [MW]': 'actual_load',
            'Forecast Total Load [MW]': 'forecast_load',
            'Bidding Zone': 'bidding_zone'
        }
        df = df.rename(columns=column_mapping)

        # Drop useless bidding zone column
        df = df.drop(columns=['bidding_zone'])
        
        # Convert to numeric (in case any issues with decimal conversion)
        for col in ['actual_load', 'forecast_load']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        dfs.append(df)
    
    # Concatenate all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Handle any rows that might have failed numeric conversion
    merged_df.dropna(inplace=True)

    # Sort by date and set as index
    merged_df = merged_df.sort_values('Date').set_index('Date')
    
    # Remove duplicates if any
    merged_df = merged_df[~merged_df.index.duplicated(keep='first')]
    
    print(f"\nMerged data shape: {merged_df.shape}")
    print(f"Date range: {merged_df.index.min()} to {merged_df.index.max()}")
    print(f"Missing values: {merged_df.isnull().sum().sum()}")
    
    return merged_df


def create_load_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features specifically for load forecasting
    """
    df = df.copy()
    
    # Time-based features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    
    # Use the holidays library for an automated list of holidays and mark them
    start_year = df.index.min().year
    end_year = df.index.max().year
    it_holidays = holidays.Italy(years=range(start_year, end_year + 1))
    df['is_holiday'] = df.index.normalize().isin(it_holidays).astype(int)
    
    # Cyclical encoding for time features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Lag features (previous load values)
    lag_steps = [1, 2, 3, 6, 12, 24, 48, 96, 168]
    for lag in lag_steps:
        df[f'load_lag_{lag}h'] = df['actual_load'].shift(lag)
    
    # Rolling statistics
    rolling_windows = [24, 48, 168]
    for window in rolling_windows:
        df[f'load_rolling_mean_{window}h'] = df['actual_load'].rolling(window=window).mean()
        df[f'load_rolling_std_{window}h'] = df['actual_load'].rolling(window=window).std()
        df[f'load_rolling_min_{window}h'] = df['actual_load'].rolling(window=window).min()
        df[f'load_rolling_max_{window}h'] = df['actual_load'].rolling(window=window).max()
    
    # Load from same hour previous days
    for days_back in [1, 7]:
        df[f'load_same_hour_{days_back}d_ago'] = df['actual_load'].shift(24 * days_back)
    
    # Drop rows with NaN values created by lag/rolling features
    df = df.dropna()
    
    return df


def analyze_load_patterns(df: pd.DataFrame, save_path: str = 'results/figures/'):
    """
    Analyze and visualize load patterns
    """
    os.makedirs(save_path, exist_ok=True)

    # Create a figure and a grid of subplots (3 rows, 2 columns).
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Panel (0, 0): Full Time Series Plot
    # Shows the overall trend and seasonality over the entire dataset duration.
    axes[0, 0].plot(df.index, df['actual_load'], linewidth=0.5, alpha=0.7)
    axes[0, 0].set_title('Electricity Load Over Time')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Load (MW)')
    
    # Panel (0, 1): Histogram of Load Values
    # Shows the distribution of the energy load, revealing common and extreme values.
    axes[0, 1].hist(df['actual_load'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Load Distribution')
    axes[0, 1].set_xlabel('Load (MW)')
    axes[0, 1].set_ylabel('Frequency')
    
    # Panel (1, 0): Average Daily Profile
    # Groups data by hour to show the typical 24-hour energy consumption cycle.
    hourly_avg = df.groupby('hour')['actual_load'].mean()
    hourly_std = df.groupby('hour')['actual_load'].std()
    axes[1, 0].plot(hourly_avg.index, hourly_avg.values, marker='o', label='Average')
    axes[1, 0].fill_between(hourly_avg.index, 
                           hourly_avg - hourly_std, 
                           hourly_avg + hourly_std, 
                           alpha=0.3, label='±1 STD')
    axes[1, 0].set_title('Average Load by Hour of Day')
    axes[1, 0].set_xlabel('Hour')
    axes[1, 0].set_ylabel('Load (MW)')
    axes[1, 0].set_xticks(range(0, 24, 2))
    axes[1, 0].legend()
    
    # Panel (1, 1): Average Weekly Profile
    # Groups data by day of the week to show the difference between weekdays and weekends.
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    daily_avg = df.groupby('day_of_week')['actual_load'].mean()
    daily_std = df.groupby('day_of_week')['actual_load'].std()
    x_pos = range(7)
    axes[1, 1].bar(x_pos, daily_avg.values, yerr=daily_std.values, capsize=5)
    axes[1, 1].set_title('Average Load by Day of Week')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(days)
    axes[1, 1].set_ylabel('Load (MW)')
    
    # Panel (2, 0): Average Monthly (Seasonal) Profile
    # Groups data by month to visualize the annual seasonal pattern (e.g., summer/winter peaks).
    monthly_avg = df.groupby('month')['actual_load'].mean()
    axes[2, 0].plot(monthly_avg.index, monthly_avg.values, marker='o')
    axes[2, 0].set_title('Average Load by Month')
    axes[2, 0].set_xlabel('Month')
    axes[2, 0].set_ylabel('Load (MW)')
    axes[2, 0].set_xticks(range(1, 13))
    
    # Panel (2, 1): Weekday vs. Weekend Daily Profile
    # Overlays the hourly profiles for weekdays and weekends for direct comparison.
    weekend_hourly = df[df['is_weekend'] == 1].groupby('hour')['actual_load'].mean()
    weekday_hourly = df[df['is_weekend'] == 0].groupby('hour')['actual_load'].mean()
    axes[2, 1].plot(weekend_hourly.index, weekend_hourly.values, marker='o', label='Weekend')
    axes[2, 1].plot(weekday_hourly.index, weekday_hourly.values, marker='s', label='Weekday')
    axes[2, 1].set_title('Load Pattern: Weekday vs Weekend')
    axes[2, 1].set_xlabel('Hour')
    axes[2, 1].set_ylabel('Load (MW)')
    axes[2, 1].legend()
    axes[2, 1].set_xticks(range(0, 24, 2))
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/load_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n=== Load Statistics ===")
    print(f"Average load: {df['actual_load'].mean():.2f} MW")
    print(f"Peak load: {df['actual_load'].max():.2f} MW")
    print(f"Minimum load: {df['actual_load'].min():.2f} MW")
    print(f"Standard deviation: {df['actual_load'].std():.2f} MW")
    
    if 'forecast_load' in df.columns:
        mae = np.abs(df['forecast_load'] - df['actual_load']).mean()
        mape = (np.abs(df['forecast_load'] - df['actual_load']) / df['actual_load']).mean() * 100
        print(f"\nTerna Forecast Accuracy:")
        print(f"MAE: {mae:.2f} MW")
        print(f"MAPE: {mape:.2f}%")


def prepare_data_for_training(df: pd.DataFrame, target_col: str = 'actual_load') -> pd.DataFrame:
    """
    This function selects the final columns for the model and appends them in the final dataset.

    We can add or remove column names if we want to modify the final dataset.
    """
    feature_columns = [
        # Time features
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
        # Calendar features
        'is_weekend', 'is_holiday',
        # Memory features (lags)
        'load_lag_1h', 'load_lag_2h', 'load_lag_3h', 'load_lag_6h', 
        'load_lag_12h', 'load_lag_24h', 'load_lag_48h', 'load_lag_168h',
        # Rolling stats
        'load_rolling_mean_24h', 'load_rolling_std_24h',
        'load_rolling_mean_48h', 'load_rolling_std_48h',
        'load_rolling_mean_168h', 'load_rolling_std_168h',
        # Past memory features
        'load_same_hour_1d_ago', 'load_same_hour_7d_ago',
    ]
    
    # Append all the feature columns to our final dataset.
    feature_columns.append(target_col)
    
    # Final list that includes columns that are both in my list and in the dataframe to avoid typos.
    available_features = [col for col in feature_columns if col in df.columns]
    
    return df[available_features]


if __name__ == "__main__":
    
    # Define file paths
    file_paths = [
        'data/raw/italy_energy_consumption_2022.csv',
        'data/raw/italy_energy_consumption_2023.csv',
        'data/raw/italy_energy_consumption_2024.csv',
        'data/raw/italy_energy_consumption_2025.csv'
    ]
    
    # Create the necessary directories if they don't exist
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    Path('results/figures').mkdir(parents=True, exist_ok=True)
    
    df = load_ang_merge_data(file_paths)
    df = create_load_features(df)
    
    analyze_load_patterns(df)
    train_df = prepare_data_for_training(df)
    
    # Save processed data
    train_df.to_csv('data/processed/load_data_processed.csv')
    print(f"\nProcessed data saved. Shape: {train_df.shape}")