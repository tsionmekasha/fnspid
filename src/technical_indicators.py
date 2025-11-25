import talib

def moving_average(series, timeperiod=7):
    """
    Compute simple moving average (SMA) for a price series.

    Args:
        series (pd.Series): Price series.
        timeperiod (int): Window size for moving average.

    Returns:
        pd.Series: SMA values.
    """
    return talib.SMA(series, timeperiod=timeperiod)

def rsi(series, timeperiod=14):
    """
    Compute Relative Strength Index (RSI).

    Args:
        series (pd.Series): Price series.
        timeperiod (int): Lookback period for RSI.

    Returns:
        pd.Series: RSI values.
    """
    return talib.RSI(series, timeperiod=timeperiod)

def macd(series, fastperiod=12, slowperiod=26, signalperiod=9):
    """
    Compute MACD, signal line, and histogram.

    Args:
        series (pd.Series): Price series.
        fastperiod (int): Fast EMA period.
        slowperiod (int): Slow EMA period.
        signalperiod (int): Signal line period.

    Returns:
        tuple: (macd_line, signal_line, histogram)
    """
    macd_line, signal_line, hist = talib.MACD(series, fastperiod, slowperiod, signalperiod)
    return macd_line, signal_line, hist
