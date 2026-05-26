#!/usr/bin/env python3
"""Preprocesses raw BTC minute-level data into normalized hourly prices."""
import pandas as pd
import numpy as np


COINBASE = 'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'
BITSTAMP = 'bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'
OUTPUT = 'btc_hourly.npy'


def load_exchange(filepath):
    """Load and clean a single exchange CSV file.

    Parses timestamps, drops rows missing a Close price, and returns
    a DataFrame indexed by datetime with only the Close column.

    Args:
        filepath: path to the raw CSV file

    Returns:
        pandas.DataFrame with a DatetimeIndex and a single 'Close' column
    """
    df = pd.read_csv(filepath)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df = df.set_index('Timestamp')
    df = df[['Close']].dropna()
    return df


def resample_hourly(df):
    """Resample a minute-level DataFrame to hourly close prices.

    Takes the last available close price within each hour, then
    forward-fills any remaining gaps up to 24 hours.

    Args:
        df: pandas.DataFrame with DatetimeIndex and 'Close' column

    Returns:
        pandas.Series of hourly close prices with no NaN values
    """
    hourly = df['Close'].resample('1H').last()
    hourly = hourly.fillna(method='ffill', limit=24).dropna()
    return hourly


def normalize(series):
    """Apply min-max normalization to a pandas Series.

    Args:
        series: pandas.Series of numeric values

    Returns:
        tuple of (normalized Series, min value, max value)
    """
    min_val = series.min()
    max_val = series.max()
    return (series - min_val) / (max_val - min_val), min_val, max_val


def preprocess():
    """Run the full preprocessing pipeline and save output.

    Loads both exchange datasets, combines them (bitstamp primary,
    coinbase fills gaps), resamples to hourly, normalizes, and saves
    the result as a NumPy array.
    """
    print("Loading coinbase data...")
    coinbase = load_exchange(COINBASE)
    print("Loading bitstamp data...")
    bitstamp = load_exchange(BITSTAMP)

    # Merge: bitstamp as primary, fill gaps with coinbase
    combined = bitstamp.combine_first(coinbase)

    print("Resampling to hourly windows...")
    hourly = resample_hourly(combined)

    print("Normalizing...")
    normalized, min_val, max_val = normalize(hourly)

    data = normalized.values.astype(np.float32)
    np.save(OUTPUT, data)

    print("Saved {} hourly data points to {}".format(len(data), OUTPUT))
    print("Price range: ${:.2f} - ${:.2f}".format(min_val, max_val))


if __name__ == '__main__':
    preprocess()
