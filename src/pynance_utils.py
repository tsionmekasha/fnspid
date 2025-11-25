# src/pynance_utils.py

import pandas as pd

def calculate_daily_returns_pynance(df: pd.DataFrame, close_col='close') -> pd.Series:
    """
    Calculate daily returns using PyNance.

    Parameters:
        df (pd.DataFrame): Stock data with a 'close' column.
        close_col (str): Column name for closing prices.

    Returns:
        pd.Series: Daily returns.
    """
    # `pynance` version installed in the venv doesn't expose a `Stock` class.
    # Use pandas to calculate daily returns directly to avoid import errors.
    returns = df[close_col].pct_change()
    # Ensure the series has a DatetimeIndex for correct slicing and alignment
    if 'date' in df.columns:
        returns.index = pd.to_datetime(df['date'])
    elif not isinstance(df.index, pd.DatetimeIndex):
        try:
            returns.index = pd.to_datetime(df.index)
        except Exception:
            # Leave index as-is if conversion fails
            pass
    return returns

def calculate_volatility(df: pd.DataFrame, window=14, close_col='close') -> pd.Series:
    """
    Calculate rolling volatility (standard deviation of daily returns).

    Parameters:
        df (pd.DataFrame): Stock data with 'close' column.
        window (int): Rolling window size.
        close_col (str): Column name for closing prices.

    Returns:
        pd.Series: Rolling volatility.
    """
    returns = df[close_col].pct_change()
    volatility = returns.rolling(window=window).std()
    # Ensure the index is datetime if possible
    if 'date' in df.columns:
        volatility.index = pd.to_datetime(df['date'])
    elif not isinstance(df.index, pd.DatetimeIndex):
        try:
            volatility.index = pd.to_datetime(df.index)
        except Exception:
            pass
    return volatility

def calculate_moving_average(df: pd.DataFrame, window=7, close_col='close') -> pd.Series:
    """
    Calculate moving average using PyNance Stock object.

    Parameters:
        df (pd.DataFrame): Stock data with 'close' column.
        window (int): Window size for moving average.
        close_col (str): Column name for closing prices.

    Returns:
        pd.Series: Moving average.
    """
    # Use pandas' rolling mean for moving average to avoid depending on
    # a `Stock` class from `pynance` (not exported in installed version).
    ma = df[close_col].rolling(window=window).mean()
    if 'date' in df.columns:
        ma.index = pd.to_datetime(df['date'])
    elif not isinstance(df.index, pd.DatetimeIndex):
        try:
            ma.index = pd.to_datetime(df.index)
        except Exception:
            pass
    return ma
